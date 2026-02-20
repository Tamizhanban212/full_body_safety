cmake_minimum_required(VERSION 3.8)
project(danger_fields)

# Default to C++17 (often required for modern ROS 2 / Pinocchio)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------
# Find Dependencies
# ---------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)    # Required for JointState
find_package(geometry_msgs REQUIRED)  # Required for PointStamped
find_package(Eigen3 REQUIRED)
find_package(franka_ros2 REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(pinocchio REQUIRED)

# ---------------------------------------------------------
# 1. Build the Executable (The Node)
# ---------------------------------------------------------
add_executable(danger_fields_node src/danger_fields_node.cpp)

target_include_directories(danger_fields_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

# Link dependencies for the node
ament_target_dependencies(danger_fields_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  pinocchio
)

# ---------------------------------------------------------
# 2. Build the Library (The Controller Plugin)
# ---------------------------------------------------------
add_library(danger_fields SHARED src/df.cpp)

target_include_directories(danger_fields PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

# Link dependencies for the controller library
ament_target_dependencies(danger_fields
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  franka_ros2
  franka_msgs
)

# Register the plugin description file
pluginlib_export_plugin_description_file(controller_interface controller.xml)

# ---------------------------------------------------------
# 3. Install Rules
# ---------------------------------------------------------
# Install both the Library AND the Executable

install(
  TARGETS danger_fields_node
  DESTINATION lib/${PROJECT_NAME} 
)

install(
  TARGETS danger_fields
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ---------------------------------------------------------
# 4. Export Dependencies
# ---------------------------------------------------------
ament_export_libraries(danger_fields)
ament_export_dependencies(
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  pinocchio
)

ament_package()