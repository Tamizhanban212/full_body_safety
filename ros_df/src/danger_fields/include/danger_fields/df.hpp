#pragma once

#include <string>
#include <Eigen/Eigen>
#include <cassert>
#include <cmath>
#include <mutex>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <controller_interface/controller_interface.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace danger_fields {

/**
 * The pick place impedance controller moves joint 4 and 5 in a very compliant periodic movement.
 */
class DFController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string robot_type_;
  std::string robot_description_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d q_goal_;
  std::mutex q_goal_mutex_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  void updateJointStates();

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_goal_sub_;
  rclcpp::Time last_goal_time_;
};

}  // namespace danger_fields