cmake_minimum_required(VERSION 3.8)
project(nemesis_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(rc_to_wheels src/rc_to_wheels.cpp)
ament_target_dependencies(rc_to_wheels rclcpp std_msgs)
install(TARGETS rc_to_wheels DESTINATION lib/${PROJECT_NAME})

add_executable(rc_to_wheels_opponent src/rc_to_wheels_opponent.cpp)
ament_target_dependencies(rc_to_wheels_opponent rclcpp std_msgs)
install(TARGETS rc_to_wheels_opponent DESTINATION lib/${PROJECT_NAME})

add_executable(control_to_wheels src/control_to_wheels.cpp)
ament_target_dependencies(control_to_wheels rclcpp std_msgs)
install(TARGETS control_to_wheels DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(PROGRAMS src/gz_pose_to_centroid.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gz_pose_to_centroid
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

# Robot models (URDF + meshes): Nemesis and the two generic opponents
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
