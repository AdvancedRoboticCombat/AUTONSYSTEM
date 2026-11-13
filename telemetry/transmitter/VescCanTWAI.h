#ifndef VESC_CAN_TWAI_H
#define VESC_CAN_TWAI_H

#include <Arduino.h>
#include <driver/twai.h>

class VescCanTWAI {

public:
  // CAN message data
  long unsigned int rxId;
  unsigned char len = 0;
  unsigned char rxBuf[8];
  char msgString[128]; // Array to store serial string

  // VESC telemetry data
  float inpVoltage, dutyCycleNow, avgInputCurrent, avgMotorCurrent, tempFET, tempMotor;
  long erpm, WattHours;
  
  // Initialization state
  bool isInitialized;

  // Constructor
  VescCanTWAI();

  // Initialize TWAI driver with specified TX/RX pins
  bool initialize(gpio_num_t tx_pin, gpio_num_t rx_pin, uint32_t bitrate = 500);
  
  // Main spin function - call this frequently to process incoming CAN messages
  void spin();
  
  // Get latest CAN frame (returns true if message received)
  bool get_frame(uint32_t timeout_ms = 10);
  
  // Print raw CAN data to serial (for debugging)
  void print_raw_can_data();

  // VESC control commands
  void vesc_set_duty(float duty);
  void vesc_set_current(float current);
  void vesc_set_erpm(float erpm);

private:
  // Helper functions
  float process_data_frame_vesc(char datatype, unsigned char byte1, unsigned char byte2);
  int hex2int(char buf[]);
};

#endif // VESC_CAN_TWAI_H