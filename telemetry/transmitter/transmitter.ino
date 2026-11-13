#include <ArduinoJson.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <SPI.h>
#include <Adafruit_BNO08x.h>

#include "LoRaWan_APP.h"
#include "Arduino.h"

#include <math.h>

#include <driver/twai.h>

#include <FastLED.h>

#include "VescCanTWAI.h"


#define CAN_TX 36
#define CAN_RX 34

#define SOUL_SEGMENT_LENGTH 20
#define UNDERGLOW_HALF_LENGTH 46
//9 on each side, 20 in center, 38 total
#define NUM_LEDS 112
#define LED_DATA_PIN 46

CRGB leds[NUM_LEDS];


#define RF_FREQUENCY 433000000  // Hz

#define TX_OUTPUT_POWER 5  // dBm

#define LORA_BANDWIDTH 1         // [0: 125 kHz, 1: 250 kHz, 2: 500 kHz, 3: Reserved]
#define LORA_SPREADING_FACTOR 7  // [SF7..SF12]
#define LORA_CODINGRATE 1        // [1: 4/5, 2: 4/6, 3: 4/7, 4: 4/8]
#define LORA_PREAMBLE_LENGTH 8   // Same for Tx and Rx
#define LORA_SYMBOL_TIMEOUT 0    // Symbols
#define LORA_FIX_LENGTH_PAYLOAD_ON false
#define LORA_IQ_INVERSION_ON false


#define RX_TIMEOUT_VALUE 1000
#define BUFFER_SIZE 30  // Define the payload size here


char rxpacket[BUFFER_SIZE];

//List of all sensor readings from the IMU
// IMU 1
//  3 float accelerometer
//  3 float gyro
//  3 float magnetometer
//  3 float linear accel
//  3 float gravity
//  4 float rotation quaternion

// float sensorReadings [38] = {
//   0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7
// };


// IMU quaternion packet
struct QuaternionPacket {
  unsigned long timestamp;
  float qw;
  float qx;
  float qy;
  float qz;

  float gx;
  float gy;
  float gz;

  float ax;
  float ay;
  float az;
  
  float vescRPM;
};

QuaternionPacket imuPacket;
size_t tx_buffer_size = sizeof(imuPacket);
char txpacket[sizeof(imuPacket)];
//

// size_t tx_buffer_size = sizeof(sensorReadings);
// char txpacket [152]; //38 * 4
// char txpacket[sizeof(sensorReadings)];

double txNumber;

bool lora_idle = true;

static RadioEvents_t RadioEvents;
void OnTxDone(void);
void OnTxTimeout(void);

#define BNO08X_CS 4    //CS    //3   orig: 47
#define BNO08X_INT 38            //38  orig: 4
#define BNO08X_RST 37           //1   orig: 37
#define SPI_SCK 40      //SCL   //40  orig: 21
#define SPI_MISO 39     //SDA   //39  orig: 26
#define SPI_MOSI 2     //DI    //2   orig: 5

sh2_RotationVector_t GameRotationVector;
// sh2_Gyroscope_t GyroData;
// sh2_LinearAcceleration_t LinAccelData;

int timer = 0;
int timerLimit = 3;

float soul_frequency = 0.5;  // to be set by VESC data
float underglow_speed = 1;

void transmitData(void *pvParameters);
void readVESCData(void *pvParameters);

VescCanTWAI vesc;

SPIClass hspi(HSPI);
Adafruit_BNO08x bno08x(BNO08X_RST);
sh2_SensorValue_t sensorValue;

void setup() {
  Serial.begin(115200);

  Serial.println("=== LORA INIT ===");

  Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);

  txNumber = 0;

  RadioEvents.TxDone = OnTxDone;
  RadioEvents.TxTimeout = OnTxTimeout;

  Radio.Init(&RadioEvents);
  Radio.SetChannel(RF_FREQUENCY);
  Radio.SetTxConfig(MODEM_LORA, TX_OUTPUT_POWER, 0, LORA_BANDWIDTH,
                    LORA_SPREADING_FACTOR, LORA_CODINGRATE,
                    LORA_PREAMBLE_LENGTH, LORA_FIX_LENGTH_PAYLOAD_ON,
                    true, 0, 0, LORA_IQ_INVERSION_ON, 3000);


  delay(500);

  Serial.println();
  Serial.println("=== HELTEC BNO08x SPI INIT ===");

  hspi.begin(SPI_SCK, SPI_MISO, SPI_MOSI, BNO08X_CS);
  Serial.println("SPI started");

  if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT, &hspi)) {
    Serial.println(" BNO08x NOT FOUND");
    Serial.println("Check: CS, PS0, PS1, SCK, MISO, MOSI, INT");
    while (1) delay(100);
  }
  Serial.println(" BNO08x FOUND");

  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR)) {
    Serial.println("Failed to enable rotation vector");
  } else {
    Serial.println("Rotation vector enabled");
  }

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED)) {
    Serial.println("Failed to enable gyroscope");
  } else {
    Serial.println("Gyroscope enabled");
  }

  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION)) {
    Serial.println("Failed to enable linear acceleration");
  } else {
    Serial.println("Linear acceleration enabled");
  }


  Serial.println("=== FASTLED INIT ===");
  FastLED.addLeds<NEOPIXEL, LED_DATA_PIN>(leds, NUM_LEDS);


  Serial.println("=== VESC CAN/TWAI INIT ===");
  
  if (!vesc.initialize((gpio_num_t)CAN_TX, (gpio_num_t)CAN_RX, 500)) {
    Serial.println("VESC CAN initialization failed!");
  }

  Serial.println("=== TRANSMITTER RTOS INIT ===");

  TaskHandle_t taskTransmitData;
  xTaskCreatePinnedToCore(transmitData, "transmitData", 10000, NULL, 0, &taskTransmitData, 0);

  Serial.println("=== VESC CAN READER RTOS INIT ===");

  TaskHandle_t taskReadVESCData;
  xTaskCreatePinnedToCore(readVESCData, "readVESCData", 10000, NULL, 0, &taskReadVESCData, 0);

  Serial.println("=== LED CONTROL RTOS INIT ===");

  TaskHandle_t taskControlLEDs;
  xTaskCreatePinnedToCore(controlLEDs, "controlLEDs", 10000, NULL, 0, &taskControlLEDs, 1);  //second core for LEDs
}

//Loop reads the IMU data and stores it to a (temporary) variable
void loop() {
  if (!bno08x.getSensorEvent(&sensorValue)) return;

  if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {
    GameRotationVector = sensorValue.un.gameRotationVector;

    imuPacket.timestamp = millis();
    imuPacket.qw = GameRotationVector.real;
    imuPacket.qx = GameRotationVector.i;
    imuPacket.qy = GameRotationVector.j;
    imuPacket.qz = GameRotationVector.k;

    // Serial.printf("time: %lu\n", imuPacket.timestamp, 6);
    // Serial.printf("qw: %.6f\n", GameRotationVector.real, 6);
    // Serial.printf("qi: %.6f\n", GameRotationVector.i, 6);
    // Serial.printf("qj: %.6f\n", GameRotationVector.j, 6);
    // Serial.printf("qk: %.6f\n", GameRotationVector.k, 6);
    // Serial.println(" ");
    // sensorReadings[0] = GameRotationVector.real;
    // sensorReadings[1] = GameRotationVector.i;
    // sensorReadings[2] = GameRotationVector.j;
    // sensorReadings[3] = GameRotationVector.k;
  }

  if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
    imuPacket.gx = sensorValue.un.gyroscope.x;
    imuPacket.gy = sensorValue.un.gyroscope.y;
    imuPacket.gz = sensorValue.un.gyroscope.z;

    // Serial.printf("gx: %.6f\n", imuPacket.gx);
    // Serial.printf("gy: %.6f\n", imuPacket.gy);
    // Serial.printf("gz: %.6f\n", imuPacket.gz);
    // Serial.println(" ");
  }

  if (sensorValue.sensorId == SH2_LINEAR_ACCELERATION) {
    imuPacket.ax = sensorValue.un.linearAcceleration.x;
    imuPacket.ay = sensorValue.un.linearAcceleration.y;
    imuPacket.az = sensorValue.un.linearAcceleration.z;

    // Serial.printf("ax: %.6f\n", imuPacket.ax);
    // Serial.printf("ay: %.6f\n", imuPacket.ay);
    // Serial.printf("az: %.6f\n", imuPacket.az);
    // Serial.println(" ");
  }
}

//Transmits via LoRa every second
void transmitData(void *pvParameters) {
  while (true) {

    if (lora_idle == true) {
      txNumber += 0.01;

      memcpy(txpacket, &imuPacket, tx_buffer_size);
      Serial.printf("\r\nsending packet \"%s\" , length %d\r\n", txpacket, tx_buffer_size);
      Radio.Send((uint8_t *)txpacket, tx_buffer_size);  //send the package out

      lora_idle = false;
    }
    Radio.IrqProcess();

    vTaskDelay(pdMS_TO_TICKS(1));
  }
}


uint8_t gHue = 255;
uint8_t gHueDelta = 3;

float waveTime = 0.0;
float waveIncrement = .016;

void controlLEDs(void *pvParameters) {
  while (true) {
    // MANIPULATE SOUL SEGMENT
    float t = millis() / 1000.0;
    uint8_t brightness = (sin(2 * M_PI * soul_frequency * t) + 1.0) * 127.5;
    fill_solid(leds, SOUL_SEGMENT_LENGTH, CHSV(gHue, 255, brightness));


    float speed = underglow_speed;
    float direction = (speed >= 0) ? 1.0 : -1.0;
    float absSpeed = abs(speed);

    float waveDensity = 0.4;
    float waveFrequency = waveDensity;
    float baseWaveSpeed = 10.0;
    waveTime += waveIncrement * underglow_speed; 

    // UNDERGLOW FIRST HALF

    for (int i = 0; i < UNDERGLOW_HALF_LENGTH; i++) {
      float position = i;

      float wave = sin((position * waveFrequency) - (waveTime * baseWaveSpeed));

      uint8_t brightness = (wave + 1.0) * 127.5;

      leds[SOUL_SEGMENT_LENGTH + i] = CHSV(gHue, 255, brightness);
      leds[NUM_LEDS - i - 1] = CHSV(gHue, 255, brightness);
    }
    FastLED.show();
    vTaskDelay(pdMS_TO_TICKS(16));

    //realistically we could probably use fill_solid
  }
}


// Read VESC CAN data continuously
void readVESCData(void *pvParameters) {
  while (true) {
    // Process incoming CAN messages and update VESC telemetry
    vesc.spin();
    
    // Uncomment to see raw CAN messages for debugging:
    // vesc.print_raw_can_data();
    
    // Update LED effects based on VESC telemetry
    // Map ERPM to underglow speed (adjust scaling as needed)
    float rpm = abs(vesc.erpm) / 7.0; // Convert ERPM to approximate RPM (adjust motor poles)
    imuPacket.vescRPM = vesc.erpm;
    underglow_speed = rpm / 1000.0; // Scale to reasonable speed value
    
    // Map motor current or duty cycle to soul frequency
    soul_frequency = 0.5 + (abs(vesc.avgMotorCurrent) / 20.0); // Scale 0-20A to 0.5-1.5Hz
    
    // Optional: Print telemetry data periodically
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 1000) {
      Serial.printf("VESC: RPM=%ld Current=%.1fA Voltage=%.1fV Duty=%.3f\n", 
                    vesc.erpm, vesc.avgMotorCurrent, vesc.inpVoltage, vesc.dutyCycleNow);
      lastPrint = millis();
    }
    
    // Small delay to prevent spinning too fast
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}


void OnTxDone(void) {
  // Serial.println("TX done......");
  lora_idle = true;
}

void OnTxTimeout(void) {
  Radio.Sleep();
  // Serial.println("TX Timeout......");
  lora_idle = true;
}
