# VescCanTWAI Library

A library for communicating with VESC motor controllers over CAN bus using the ESP32's native TWAI (Two-Wire Automotive Interface) driver.

## Overview

This library replaces the MCP2515-based `vesc_can_bus_arduino` library with a native ESP32 TWAI implementation. It provides:
- Reading VESC telemetry data (RPM, current, voltage, temperature, etc.)
- Sending control commands to VESC (duty cycle, current, ERPM)
- Clean object-oriented API

## Features

- **Native ESP32 TWAI support** - No external CAN controller needed
- **Easy to use** - Simple initialization and data access
- **Non-blocking** - Designed to work with FreeRTOS tasks
- **Configurable bitrate** - Supports 250kbps, 500kbps, or 1Mbps (default: 500kbps)

## Hardware Requirements

- ESP32 with CAN transceiver (e.g., SN65HVD230)
- VESC motor controller with CAN bus enabled

## Installation

Simply place `VescCanTWAI.h` and `VescCanTWAI.cpp` in your project directory.

## Usage

### Basic Example

```cpp
#include "VescCanTWAI.h"

#define CAN_TX 36
#define CAN_RX 34

VescCanTWAI vesc;

void setup() {
  Serial.begin(115200);
  
  // Initialize with TX/RX pins and bitrate (500kbps default)
  if (!vesc.initialize((gpio_num_t)CAN_TX, (gpio_num_t)CAN_RX, 500)) {
    Serial.println("VESC CAN initialization failed!");
    while(1);
  }
}

void loop() {
  // Process incoming CAN messages
  vesc.spin();
  
  // Read telemetry data
  Serial.printf("RPM: %ld\n", vesc.erpm);
  Serial.printf("Voltage: %.2fV\n", vesc.inpVoltage);
  Serial.printf("Motor Current: %.2fA\n", vesc.avgMotorCurrent);
  Serial.printf("Duty Cycle: %.3f\n", vesc.dutyCycleNow);
  Serial.printf("FET Temp: %.1f°C\n", vesc.tempFET);
  Serial.printf("Motor Temp: %.1f°C\n", vesc.tempMotor);
  
  delay(100);
}
```

### FreeRTOS Task Example

```cpp
VescCanTWAI vesc;

void vescTask(void *pvParameters) {
  while (true) {
    vesc.spin();
    vTaskDelay(pdMS_TO_TICKS(1)); // Process messages as fast as possible
  }
}

void setup() {
  vesc.initialize((gpio_num_t)CAN_TX, (gpio_num_t)CAN_RX);
  
  xTaskCreatePinnedToCore(vescTask, "vescTask", 4096, NULL, 1, NULL, 0);
}
```

## API Reference

### Constructor

```cpp
VescCanTWAI()
```
Creates a new VescCanTWAI instance.

### Methods

#### `bool initialize(gpio_num_t tx_pin, gpio_num_t rx_pin, uint32_t bitrate = 500)`
Initializes the TWAI driver.
- **tx_pin**: GPIO pin for CAN TX
- **rx_pin**: GPIO pin for CAN RX
- **bitrate**: CAN bitrate in kbps (250, 500, or 1000). Default: 500
- **Returns**: `true` if successful, `false` otherwise

#### `void spin()`
Processes incoming CAN messages and updates telemetry data. Call this frequently in your main loop or FreeRTOS task.

#### `bool get_frame(uint32_t timeout_ms = 10)`
Receives a single CAN frame with specified timeout.
- **timeout_ms**: Timeout in milliseconds
- **Returns**: `true` if message received, `false` otherwise

#### `void print_raw_can_data()`
Prints the last received CAN frame to Serial (for debugging).

#### `void vesc_set_duty(float duty)`
Sets VESC duty cycle (-1.0 to 1.0).

#### `void vesc_set_current(float current)`
Sets VESC motor current in Amps.

#### `void vesc_set_erpm(float erpm)`
Sets VESC electrical RPM.

### Public Variables

All telemetry data is stored in public member variables:

- `float inpVoltage` - Input voltage (V)
- `float dutyCycleNow` - Current duty cycle (0.0-1.0)
- `float avgInputCurrent` - Average input current (A)
- `float avgMotorCurrent` - Average motor current (A)
- `float tempFET` - FET temperature (°C)
- `float tempMotor` - Motor temperature (°C)
- `long erpm` - Electrical RPM
- `long WattHours` - Energy consumed (Wh)

## CAN Message IDs

The library processes these VESC CAN message IDs:

- `0x8000090A` - ERPM, Motor Current, Duty Cycle
- `0x80000F0A` - Watt Hours
- `0x8000100A` - FET Temp, Motor Temp, Input Current
- `0x80001B0A` - Input Voltage

## Differences from Original Library

| Feature | Original (MCP2515) | VescCanTWAI |
|---------|-------------------|-------------|
| CAN Controller | External MCP2515 | Built-in ESP32 TWAI |
| SPI Required | Yes | No |
| Initialization | `CAN0.begin()` | `vesc.initialize()` |
| Default Bitrate | 250kbps | 500kbps |
| Blocking | Yes | Configurable timeout |

## Troubleshooting

### No messages received
- Check TX/RX pin connections
- Verify CAN transceiver is powered
- Ensure VESC CAN bus is enabled in VESC Tool
- Check bitrate matches VESC configuration (usually 500kbps)
- Make sure CAN bus has proper termination resistors (120Ω)

### Compilation errors
- Ensure you're using ESP32 board package (not ESP8266 or other)
- Include `<driver/twai.h>` before using the library

## License

Ported from the original `vesc_can_bus_arduino` library. Adapted for ESP32 TWAI driver.