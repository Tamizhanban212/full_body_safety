#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <danger_fields/danger_calc.hpp>
#include <algorithm>
#include <memory>
#include <vector>
#include <chrono>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

class DangerFieldsNode : public rclcpp::Node {
    public:
        DangerFieldsNode() : Node("danger_field_node"), danger_field_(5.0, 1.0, 1.0, 20)
        {
            desc_sub_ = create_subscription<std_msgs::msg::String>(
                "/robot_description",
                rclcpp::QoS(1).transient_local(),
                std::bind(&DangerFieldsNode::robotDescriptionCallback, this, std::placeholders::_1)
            );

            joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
                "/joint_states",
                10,
                std::bind(&DangerFieldsNode::jointStateCallback, this, std::placeholders::_1)
            );

            obstacle_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
                "/clicked_point",
                10,
                std::bind(&DangerFieldsNode::obstacleCallback, this, std::placeholders::_1)
            );

            goal_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
                "/danger_fields/joint_goal", 10);

            // Timer — continuously publishes at 50Hz once all data is available
            timer_ = create_wall_timer(
                std::chrono::milliseconds(20),
                [this]() {
                    if (model_loaded_ && state_received_ && obstacle_set_)
                        computeAndPublish();
                });
        }

    private:
        void computeAndPublish() {
            if (!model_loaded_ || !state_received_) return;

            // FK
            pinocchio::forwardKinematics(model_, *data_, q_, dq_);
            pinocchio::updateFramePlacements(model_, *data_);
            pinocchio::computeJointJacobians(model_, *data_, q_);

            int nv = model_.nv;

            // ── Step 1: Primary task Jacobian (end effector) ──────────
            Eigen::MatrixXd J_task = Eigen::MatrixXd::Zero(6, nv);
            pinocchio::getJointJacobian(model_, *data_, model_.njoints - 1,
                                        pinocchio::LOCAL_WORLD_ALIGNED, J_task);
            Eigen::MatrixXd J_task_lin = J_task.topRows(3);   // (3 x nv)

            // Pseudoinverse of task Jacobian
            Eigen::MatrixXd J_pinv = J_task_lin.completeOrthogonalDecomposition()
                                                .pseudoInverse();   // (nv x 3)

            // Nullspace projector: N = I - J†J
            Eigen::MatrixXd N = Eigen::MatrixXd::Identity(nv, nv)
                                - J_pinv * J_task_lin;              // (nv x nv)

            // ── Step 2: q0_dot from danger field ──────────────────────
            Eigen::VectorXd q0_dot = Eigen::VectorXd::Zero(nv);
            const double kv = 0.1;   // tuning gain

            if (obstacle_set_) {
                const double kv = 1.0;   // Tuning gain for repulsion speed

                for (int i = 1; i < model_.njoints; ++i) {
                    Eigen::Vector3d r_i   = data_->oMi[i].translation();
                    Eigen::Vector3d r_ip1 = (i + 1 < model_.njoints)
                                            ? data_->oMi[i+1].translation()
                                            : r_i;

                    int n_pts = 10;
                    for (int k = 0; k < n_pts; ++k) {
                        double s = static_cast<double>(k) / (n_pts - 1);
                        Eigen::Vector3d r_s = r_i + s * (r_ip1 - r_i);

                        Eigen::MatrixXd J_i = Eigen::MatrixXd::Zero(6, nv);
                        pinocchio::getJointJacobian(model_, *data_, i,
                                                    pinocchio::LOCAL_WORLD_ALIGNED, J_i);

                        Eigen::MatrixXd J_s_lin = J_i.topRows(3) - pinocchio::skew(r_s - r_i) * J_i.bottomRows(3);
                        Eigen::Vector3d v_s  = J_i.topRows(3) * dq_;
                        
                        // 1. Get scalar danger magnitude (CDF)
                        double danger_mag = danger_field_.pointDanger(obstacle_pos_, r_s, v_s);

                        // 2. Get danger gradient for direction (∇CDF)
                        Eigen::Vector3d grad = danger_field_.computePointGradient(obstacle_pos_, r_s, v_s);
                        double grad_norm = grad.norm();

                        if (grad_norm < 1e-9) continue;

                        // 3. Eq 10: Vector Danger Field = danger_mag * (grad / grad_norm)
                        Eigen::Vector3d CDF_bar = danger_mag * (grad / grad_norm);

                        // Safety cap: Prevent math explosions if the obstacle is perfectly on the link
                        if (CDF_bar.norm() > 20.0) {
                            CDF_bar = CDF_bar.normalized() * 20.0;
                        }

                        double w  = (k == 0 || k == n_pts - 1) ? 0.5 : 1.0;
                        double ds = 1.0 / (n_pts - 1);

                        q0_dot += w * ds * kv * J_s_lin.transpose() * CDF_bar;
                    }
                }
            }

            // ── Step 3: Project into nullspace ────────────────────────
            // q_dot = q_dot_task + N * q0_dot
            // q_dot_task = 0 (impedance controller handles task)
            Eigen::VectorXd q_dot = N * q0_dot;

            // ── Step 4: Integrate to get q_goal ───────────────────────
            double dt = 0.02;   // matches 50Hz timer
            Eigen::VectorXd q_goal = q_ + dt * q_dot;

            // ── Step 5: Publish ───────────────────────────────────────
            std_msgs::msg::Float64MultiArray msg;
            for (int i = 0; i < nv; ++i)
                msg.data.push_back(q_goal(i));
            goal_pub_->publish(msg);

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                "q0_dot norm: %.6f  q_dot norm: %.6f",
                q0_dot.norm(), q_dot.norm());
        }

        void robotDescriptionCallback(const std_msgs::msg::String::SharedPtr msg) {
            if (model_loaded_) {
                RCLCPP_WARN(get_logger(), "Robot description already loaded, ignoring.");
                return;
            }
            try {
                pinocchio::urdf::buildModelFromXML(msg->data, model_);
                data_ = std::make_unique<pinocchio::Data>(model_);
                q_  = Eigen::VectorXd::Zero(model_.nq);
                dq_ = Eigen::VectorXd::Zero(model_.nv);
                model_loaded_ = true;
                RCLCPP_INFO(get_logger(), "Pinocchio model loaded: %d joints", model_.njoints);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(get_logger(), "Failed to load robot description: %s", e.what());
            }
        }

        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
            if (!model_loaded_) return;

            for (size_t i = 0; i < msg->name.size(); i++) {
                auto it = std::find(model_.names.begin(), model_.names.end(), msg->name[i]);
                if (it != model_.names.end()) {
                    int idx = static_cast<int>(std::distance(model_.names.begin(), it)) - 1;
                    if (idx >= 0 && idx < model_.nv) {
                        q_(idx)  = msg->position[i];
                        dq_(idx) = msg->velocity[i];
                    }
                }
            }
            state_received_ = true;
            // timer handles computeAndPublish — no call needed here
        }

        void obstacleCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
            obstacle_pos_ << msg->point.x, msg->point.y, msg->point.z;
            obstacle_set_ = true;
            RCLCPP_INFO(get_logger(), "Obstacle set at: (%.2f, %.2f, %.2f)",
                        obstacle_pos_.x(), obstacle_pos_.y(), obstacle_pos_.z());

            // Trigger immediately on first obstacle set
            if (model_loaded_ && state_received_)
                computeAndPublish();
        }

        // Members
        pinocchio::Model model_;
        std::unique_ptr<pinocchio::Data> data_;
        bool model_loaded_{false};

        Eigen::VectorXd q_;
        Eigen::VectorXd dq_;
        bool state_received_{false};

        Eigen::Vector3d obstacle_pos_{0.0, 0.0, 0.0};
        bool obstacle_set_{false};

        DangerField danger_field_;

        rclcpp::TimerBase::SharedPtr timer_;

        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr            desc_sub_;
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr     joint_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr obstacle_sub_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr    goal_pub_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DangerFieldsNode>());
    rclcpp::shutdown();
    return 0;
}