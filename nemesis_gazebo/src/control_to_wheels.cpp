// For use to convert control to motor velocity commands (NEMESIS PUBLISHER)

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <algorithm>
#include <cmath>

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("control_to_wheels");

    // 0-based indices into your channels array
    const int ch_left  = n->declare_parameter("ch_left",  0); // CH1
    const int ch_right = n->declare_parameter("ch_right", 1); // CH2
    const int ch_weap = n->declare_parameter("ch_weap", 2); // CH3
    const int ch_arm = n->declare_parameter("ch_arm", 3); // CH4

    // RC input is assumed normalized to [-1, 1]
    const double w_max = n->declare_parameter("w_max", 136.9036265);                // [rad/s] max wheel speed
    const double w_arm_max = n->declare_parameter("w_arm_max", 8.4);                // [rad/s] arm speed: the 2.95 rad swing takes 0.35 s on the robot
                                                                                    // (37.9, the joint's limit, did it in 0.08 s and threw the robot over)
    const double w_weap_max = n->declare_parameter("w_weap_max", 387);              // [rad/s] max weapon speed
    const double db    = n->declare_parameter("deadband", 0.05);  // unitless
    const bool inv_l   = n->declare_parameter("invert_left",  false);
    const bool inv_r   = n->declare_parameter("invert_right", false);
    const bool inv_weap = n->declare_parameter("invert_weap", false);
    const bool inv_arm  = n->declare_parameter("invert_arm",  false);

    auto pubL = n->create_publisher<std_msgs::msg::Float64>("/nemesis/left_wheels/cmd_vel", 10);
    auto pubR = n->create_publisher<std_msgs::msg::Float64>("/nemesis/right_wheels/cmd_vel", 10);
    auto pubArm = n->create_publisher<std_msgs::msg::Float64>("/nemesis/arm_motor/cmd_vel", 10);
    auto pubWeap = n->create_publisher<std_msgs::msg::Float64>("/nemesis/weapon_motor/cmd_vel", 10);

    auto sub = n->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/control/channels", 10,
        [&](std_msgs::msg::Float32MultiArray::SharedPtr m){
        if ((int)m->data.size() <= std::max(ch_left, ch_right)) return;

        double l = m->data[ch_left];
        double r = m->data[ch_right];
        double a = m->data[ch_arm];
        double w = m->data[ch_weap];

        if (inv_l) l = -l;
        if (inv_r) r = -r;
        if (inv_arm) a = -a;
        if (inv_weap) w = -w;

        if (std::abs(l) < db) l = 0.0;
        if (std::abs(r) < db) r = 0.0;
        if (std::abs(a) < db) a = 0.0;
        if (std::abs(w) < db) w = 0.0;

        l = std::clamp(l, -1.0, 1.0);
        r = std::clamp(r, -1.0, 1.0);
        a = std::clamp(a, -1.0, 1.0);
        w = 0.5 * (w + 1.0);
        w = std::clamp(w, 0.0, 1.0);

        std_msgs::msg::Float64 wl, wr, wa, ww;
        wl.data = w_max * l;
        wr.data = w_max * r;
        wa.data = w_arm_max * a;
        ww.data = w_weap_max * w;

        pubL->publish(wl);
        pubR->publish(wr);
        pubArm->publish(wa);
        pubWeap->publish(ww);
        }
    );

    rclcpp::spin(n);
    rclcpp::shutdown();
    return 0;
}
