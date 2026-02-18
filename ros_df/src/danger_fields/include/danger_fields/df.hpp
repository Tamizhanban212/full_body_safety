#pragma once

#include <string>
#include <Eigen/Eigen>
#include <cassert>
#include <cmath>


#include <controller_interface/controller_interface.hpp>

namespace danger_fields
{
    controller_interface::InterfaceConfiguration
    DF::command_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        for (int i = 0; i <= num_joints; ++i)
        {
            config.names.push_back(robot_type_ + "_joint" + std::to_string(i) + "/effort");
        }
        return config; 
    }

    contrioller_interface::InterfaceConfiguration
    DF::state_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        for (int i = 0; i <= num_joints; ++i)
        {
            config.names.push_back(robot_type_ + "_joint" + std::to_string(i) + "/position");
            config.names.push_back(robot_type_ + "_joint" + std::to_string(i) + "/velocity");
        }
        return config; 
    }

    controller
}

