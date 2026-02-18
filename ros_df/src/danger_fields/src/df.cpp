#include <controller_interface/controller_interface.hpp>
#include <danger_fields/df.hpp>
#include <danger_fields/robot_utils.hpp>

namespace danger_fields
{
    controller_interface::InterfaceConfiguration
    DFController::command_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        for (int i = 1; i <= num_joints; ++i)
        {
            config.names.push_back(robot_type_ + "_joint" + std::to_string(i) + "/effort");
        }
        return config; 
    }

    controller_interface::InterfaceConfiguration
    DFController::state_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        for (int i = 1; i <= num_joints; ++i)
        {
            config.names.push_back(robot_type_ + "_joint" + std::to_string(i) + "/position");
            config.names.push_back(robot_type_ + "_joint" + std::to_string(i) + "/velocity");
        }
        return config; 
    }

 controller_interface::return_type DFController::update(
      const rclcpp::Time & /*time*/,
      const rclcpp::Duration & /*period*/)
  {
    updateJointStates();


    Vector7d q_goal;
    {
      std::lock_guard<std::mutex> lock(q_goal_mutex_);
      q_goal = q_goal_;
    }

    const double kAlpha = 0.99;
    dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
    Vector7d tau_d_calculated =
        k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);
    for (int i = 0; i < num_joints; ++i)
    {
      command_interfaces_[i].set_value(tau_d_calculated(i));
    }
    return controller_interface::return_type::OK;
  }

  CallbackReturn DFController::on_init()
  {
    try
    {
      auto_declare<std::string>("robot_type", "");
      auto_declare<std::vector<double>>("k_gains", {});
      auto_declare<std::vector<double>>("d_gains", {});
    }
    catch (const std::exception &e)
    {
      fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
      return CallbackReturn::ERROR;
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn DFController::on_configure(
      const rclcpp_lifecycle::State & /*previous_state*/)
  {
    robot_type_ = get_node()->get_parameter("robot_type").as_string();
    auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
    auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
    if (k_gains.empty())
    {
      RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
      return CallbackReturn::FAILURE;
    }
    if (k_gains.size() != static_cast<uint>(num_joints))
    {
      RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
                   num_joints, k_gains.size());
      return CallbackReturn::FAILURE;
    }
    if (d_gains.empty())
    {
      RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
      return CallbackReturn::FAILURE;
    }
    if (d_gains.size() != static_cast<uint>(num_joints))
    {
      RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
                   num_joints, d_gains.size());
      return CallbackReturn::FAILURE;
    }
    for (int i = 0; i < num_joints; ++i)
    {
      d_gains_(i) = d_gains.at(i);
      k_gains_(i) = k_gains.at(i);
    }
    dq_filtered_.setZero();

    auto parameters_client =
        std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
    parameters_client->wait_for_service();

    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty())
    {
      robot_description_ = result[0].value_to_string();
    }
    else
    {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
    }

    robot_type_ =
        robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
      
    joint_goal_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/danger_fields/joint_goal", 10,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
          if (static_cast<int>(msg->data.size()) != num_joints) {
            RCLCPP_WARN(get_node()->get_logger(),
                        "Received joint_goal with wrong size %zu, expected %d",
                        msg->data.size(), num_joints);
            return;
          }
          std::lock_guard<std::mutex> lock(q_goal_mutex_);
          for (int i = 0; i < num_joints; ++i)
            q_goal_(i) = msg->data[i];
          last_goal_time_ = get_node()->now();   // update timestamp
        });

    // ── ADD 2: Initialize timestamp so timeout doesn't trigger immediately ───
    last_goal_time_ = get_node()->now();


    return CallbackReturn::SUCCESS;
  }

  CallbackReturn DFController::on_activate(
      const rclcpp_lifecycle::State & /*previous_state*/)
  {
    updateJointStates();
    dq_filtered_.setZero();
    {
      std::lock_guard<std::mutex> lock(q_goal_mutex_);
      q_goal_ = q_;
    }
    return CallbackReturn::SUCCESS;
  }

  void DFController::updateJointStates()
  {
    for (auto i = 0; i < num_joints; ++i)
    {
      const auto &position_interface = state_interfaces_.at(2 * i);
      const auto &velocity_interface = state_interfaces_.at(2 * i + 1);

      assert(position_interface.get_interface_name() == "position");
      assert(velocity_interface.get_interface_name() == "velocity");

      q_(i) = position_interface.get_value();
      dq_(i) = velocity_interface.get_value();
    }
  }

} 
// namespace danger_fields
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(danger_fields::DFController,
                       controller_interface::ControllerInterface)


