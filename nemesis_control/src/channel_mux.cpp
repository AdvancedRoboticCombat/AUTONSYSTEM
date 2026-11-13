// channel_mux.cpp
// Forwards manual RC input (radiomaster/channels) to control/channels when
// override is active. In autonomous mode the policy_engine publishes directly
// to control/channels, so this node does nothing.
//
// override = false  →  policy_engine owns /control/channels (autonomous mode)
// override = true   →  forwards radiomaster/channels to /control/channels (manual / driver mode)

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

class ChannelMux : public rclcpp::Node
{
public:
    ChannelMux() : Node("channel_mux"), override_active_(false)
    {
        // Output – forwarded to control_to_wheels only in manual mode
        pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
            "/control/channels", 10);

        // Manual input – from radiomaster_reader
        manual_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            "/radiomaster/channels", 10,
            [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                if (override_active_) {
                    pub_->publish(*msg);
                }
            });

        // Override switch – published by dashboard
        override_sub_ = create_subscription<std_msgs::msg::Bool>(
            "/nemesis/override", 10,
            [this](std_msgs::msg::Bool::SharedPtr msg) {
                override_active_ = msg->data;
            });
    }

private:
    bool override_active_;

    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr manual_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr override_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ChannelMux>());
    rclcpp::shutdown();
    return 0;
}
