#include "VescCanTWAI.h"

VescCanTWAI::VescCanTWAI() {
  // Initialize telemetry variables
  inpVoltage = 0.0;
  dutyCycleNow = 0.0;
  avgInputCurrent = 0.0;
  avgMotorCurrent = 0.0;
  tempFET = 0.0;
  tempMotor = 0.0;
  erpm = 0;
  WattHours = 0;
  rxId = 0;
  len = 0;
  isInitialized = false;
}

bool VescCanTWAI::initialize(gpio_num_t tx_pin, gpio_num_t rx_pin, uint32_t bitrate) {
  // Configure TWAI with specified TX/RX pins
  twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(tx_pin, rx_pin, TWAI_MODE_NORMAL);
  
  // Set timing rate based on bitrate parameter (default 500kbps for VESC)
  twai_timing_config_t t_config;
  switch(bitrate) {
    case 250:
      t_config = TWAI_TIMING_CONFIG_250KBITS();
      break;
    case 500:
      t_config = TWAI_TIMING_CONFIG_500KBITS();
      break;
    case 1000:
      t_config = TWAI_TIMING_CONFIG_1MBITS();
      break;
    default:
      t_config = TWAI_TIMING_CONFIG_500KBITS();
  }
  
  twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
    Serial.println("VESC CAN: driver install failed");
    return false;
  }

  if (twai_start() != ESP_OK) {
    Serial.println("VESC CAN: start failed");
    return false;
  }

  Serial.println("VESC CAN: initialized successfully");
  isInitialized = true;
  return true;
}

void VescCanTWAI::spin() {
  if (!isInitialized) {
    return; // Not initialized yet
  }
  
  if (!get_frame()) {
    return; // No message received
  }

  // Process VESC CAN messages based on ID
  if (rxId == 0x8000090A) {
    dutyCycleNow = process_data_frame_vesc('D', rxBuf[6], rxBuf[7]);
    avgMotorCurrent = process_data_frame_vesc('C', rxBuf[4], rxBuf[5]);
    
    // Process ERPM (4 bytes, big endian)
    unsigned char erpmvals[4];
    erpmvals[0] = rxBuf[3];
    erpmvals[1] = rxBuf[2];
    erpmvals[2] = rxBuf[1];
    erpmvals[3] = rxBuf[0];
    erpm = *(long *)erpmvals;
  }
  
  if (rxId == 0x80000F0A) {
    // Process Watt Hours (4 bytes, big endian)
    unsigned char WHvals[4];
    WHvals[0] = rxBuf[3];
    WHvals[1] = rxBuf[2];
    WHvals[2] = rxBuf[1];
    WHvals[3] = rxBuf[0];
    WattHours = *(long *)WHvals;
  }
  
  if (rxId == 0x8000100A) {
    tempFET = process_data_frame_vesc('F', rxBuf[0], rxBuf[1]);
    tempMotor = process_data_frame_vesc('T', rxBuf[2], rxBuf[3]);
    avgInputCurrent = process_data_frame_vesc('I', rxBuf[4], rxBuf[5]);
  }
  
  if (rxId == 0x80001B0A) {
    char receivedByte[4];
    sprintf(receivedByte, "%02X%02X", rxBuf[4], rxBuf[5]);
    inpVoltage = hex2int(receivedByte) * 0.1;
  }

  // Uncomment to see raw CAN messages
  // print_raw_can_data();
}

bool VescCanTWAI::get_frame(uint32_t timeout_ms) {
  if (!isInitialized) {
    return false; // Not initialized yet
  }
  
  twai_message_t message;
  
  esp_err_t result = twai_receive(&message, pdMS_TO_TICKS(timeout_ms));
  
  if (result == ESP_OK) {
    // Populate class variables with received data
    rxId = message.identifier;
    len = message.data_length_code;
    
    for (int i = 0; i < len && i < 8; i++) {
      rxBuf[i] = message.data[i];
    }
    
    return true;
  }
  
  return false; // No message or error
}

void VescCanTWAI::print_raw_can_data() {
  sprintf(msgString, "Standard ID: 0x%.3lX       DLC: %1d  Data:", rxId, len);
  Serial.print(msgString);
  for (byte i = 0; i < len; i++) {
    sprintf(msgString, " 0x%.2X", rxBuf[i]);
    Serial.print(msgString);
  }
  Serial.println();
}

float VescCanTWAI::process_data_frame_vesc(char datatype, unsigned char byte1, unsigned char byte2) {
  char receivedByte[4];
  sprintf(receivedByte, "%02X%02X", byte1, byte2);
  float output = hex2int(receivedByte);

  switch (datatype) {
    case 'D': output *= 0.001; break; // dutyCycleNow
    case 'C': output *= 0.1; break;   // avgMotorCurrent
    case 'F': output *= 0.1; break;   // tempFET
    case 'T': output *= 0.1; break;   // tempMotor
    case 'I': output *= 0.1; break;   // avgInputCurrent
    case 'V': output *= 0.1; break;   // inpVoltage
  }
  return output;
}

int VescCanTWAI::hex2int(char buf[]) {
  return (short)strtol(buf, NULL, 16);
}

void VescCanTWAI::vesc_set_duty(float duty) {
  uint32_t set_value = duty * 100000;
  uint8_t buffer[4];
  buffer[0] = (set_value >> 24) & 0xFF;
  buffer[1] = (set_value >> 16) & 0xFF;
  buffer[2] = (set_value >> 8) & 0xFF;
  buffer[3] = set_value & 0xFF;
  
  twai_message_t message;
  message.identifier = 0x0000000A;
  message.extd = 1; // Extended frame
  message.data_length_code = 4;
  for (int i = 0; i < 4; i++) {
    message.data[i] = buffer[i];
  }
  
  twai_transmit(&message, pdMS_TO_TICKS(100));
}

void VescCanTWAI::vesc_set_current(float current) {
  uint32_t set_value = current * 1000;
  uint8_t buffer[4];
  buffer[0] = (set_value >> 24) & 0xFF;
  buffer[1] = (set_value >> 16) & 0xFF;
  buffer[2] = (set_value >> 8) & 0xFF;
  buffer[3] = set_value & 0xFF;
  
  twai_message_t message;
  message.identifier = 0x0000010A;
  message.extd = 1; // Extended frame
  message.data_length_code = 4;
  for (int i = 0; i < 4; i++) {
    message.data[i] = buffer[i];
  }
  
  twai_transmit(&message, pdMS_TO_TICKS(100));
}

void VescCanTWAI::vesc_set_erpm(float erpm) {
  uint32_t set_value = erpm;
  uint8_t buffer[4];
  buffer[0] = (set_value >> 24) & 0xFF;
  buffer[1] = (set_value >> 16) & 0xFF;
  buffer[2] = (set_value >> 8) & 0xFF;
  buffer[3] = set_value & 0xFF;
  
  twai_message_t message;
  message.identifier = 0x0000030A;
  message.extd = 1; // Extended frame
  message.data_length_code = 4;
  for (int i = 0; i < 4; i++) {
    message.data[i] = buffer[i];
  }
  
  twai_transmit(&message, pdMS_TO_TICKS(100));
}