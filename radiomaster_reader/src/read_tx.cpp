#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <SDL2/SDL.h>
#include <chrono>

float ch1=0, ch2=0, ch3=0, ch4=0;

float norm(int16_t v) {
  return v >= 0 ? v / 32767.0f : v / 32768.0f;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("radiomaster_reader");

  // SDL init
  if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
    RCLCPP_ERROR(node->get_logger(), "SDL_Init failed: %s", SDL_GetError());
    rclcpp::shutdown();
    return 1;
  }

  SDL_Joystick* js = SDL_JoystickOpen(0);
  if (!js) {
    RCLCPP_ERROR(node->get_logger(), "SDL_JoystickOpen failed: %s", SDL_GetError());
    SDL_Quit();
    rclcpp::shutdown();
    return 1;
  }

  // Publisher
  auto pub = node->create_publisher<std_msgs::msg::Float32MultiArray>(
    "/radiomaster/channels", 10);

  auto timer = node->create_wall_timer(
    std::chrono::milliseconds(5),   // 200 Hz
    [&]() {
      if (!js) return;
      SDL_JoystickUpdate();

      ch1 = norm(SDL_JoystickGetAxis(js, 0));
      ch2 = norm(SDL_JoystickGetAxis(js, 1));
      ch3 = norm(SDL_JoystickGetAxis(js, 2));
      ch4 = norm(SDL_JoystickGetAxis(js, 3));

      // Publish
      std_msgs::msg::Float32MultiArray msg;
      msg.data = {ch1, ch2, ch3, ch4};
      pub->publish(msg);

      // Debug print
      RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 5,
        "CH %.2f %.2f %.2f %.2f", ch1, ch2, ch3, ch4);
    });

  rclcpp::spin(node);

  SDL_JoystickClose(js);
  SDL_Quit();
  rclcpp::shutdown();
  return 0;
}
