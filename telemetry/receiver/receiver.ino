#include "LoRaWan_APP.h"
#include "Arduino.h"
#include "SerialTransfer.h"

SerialTransfer serialTransfer;


#define RF_FREQUENCY 433000000 //MHz
#define LORA_BANDWIDTH 1 // 125 kHz
#define LORA_SPREADING_FACTOR 7 // SF7 -> SF12 // ^SF, then ^sensitivity, but slower (^SF if RSSI < ~-115 dB; SNR < ~12, or if need more range)
#define LORA_CODINGRATE 1 // 4/5 (4/5 -> 4/8) // ^CR for slower data rate, but less RX Errors
#define LORA_PREAMBLE_LENGTH 8
#define LORA_SYMBOL_TIMEOUT 0
#define LORA_FIX_LENGTH_PAYLOAD_ON false
#define LORA_IQ_INVERSION_ON false
// #define BUFFER_SIZE 12 // test package // 152 // 38 floats × 4 bytes
// #define BUFFER_SIZE 20 // 38 floats × 4 bytes

static RadioEvents_t RadioEvents;


// float receivedFloats[3];
// float receivedFloats[5];

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

  float vescRPM;   // sent by the transmitter; without it every packet fails the size check below
};
QuaternionPacket receivedPacket;
#define BUFFER_SIZE sizeof(receivedPacket)
uint8_t rxBuffer[BUFFER_SIZE];

void OnRxDone(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr);
void OnRxTimeout(void);
void OnRxError(void);

void setup(){
  Serial.begin(115200);
  serialTransfer.begin(Serial);
  // Serial.println("=== START LORA RECEIVER ===");

  Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);

  RadioEvents.RxDone = OnRxDone;
  RadioEvents.RxTimeout = OnRxTimeout;
  RadioEvents.RxError = OnRxError;

  Radio.Init(&RadioEvents);
  Radio.SetChannel(RF_FREQUENCY);

  Radio.SetRxConfig( MODEM_LORA, LORA_BANDWIDTH, LORA_SPREADING_FACTOR,
      LORA_CODINGRATE, 0, LORA_PREAMBLE_LENGTH, LORA_SYMBOL_TIMEOUT, 
      LORA_FIX_LENGTH_PAYLOAD_ON, 0, true, 0, 0, LORA_IQ_INVERSION_ON, true);

  Serial.println("listening... :3");
  Radio.Rx(0); // continuous receiving
}

void loop() {
  // Serial.println("{'a':'a'}");
  Radio.IrqProcess();
}

void OnRxDone(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr) {
  // Serial.println("\n--- PACKET RECEIVED ---");

  if (size != sizeof(receivedPacket))
  {
    // Serial.print("Unexpected packet size: ");
    // Serial.println(size);
    Radio.Rx(0);
    return;
  }

  memcpy(rxBuffer, payload, size);
  memcpy(&receivedPacket, payload, sizeof(receivedPacket));
  
  // Write raw binary packet to serial
  // Serial.write('&');
  // Serial.write(payload, size);
  // Serial.write(';');
  serialTransfer.txObj(receivedPacket);
  serialTransfer.sendData(sizeof(receivedPacket));
  // Serial.println();
  // Serial.println("asdfasfdasfdsadfasfdasfd"); 

  // Serial.print("RSSI: ");
  // Serial.println(rssi);
  // Serial.print("SNR: ");
  // Serial.println(snr);

  // // disp sensor readings i j k
  // Serial.print("Time (ms): ");
  // Serial.println(receivedPacket.timestamp);
  // Serial.print("qw: ");
  // Serial.println(receivedPacket.qw, 6);
  // Serial.print("qx: ");
  // Serial.println(receivedPacket.qx, 6);
  // Serial.print("qy: ");
  // Serial.println(receivedPacket.qy, 6);
  // Serial.print("qz: ");
  // Serial.println(receivedPacket.qz, 6);

  // Serial.print("gx: ");
  // Serial.println(receivedPacket.gx, 6);
  // Serial.print("gy: ");
  // Serial.println(receivedPacket.gy, 6);
  // Serial.print("gz: ");
  // Serial.println(receivedPacket.gz, 6);

  // Serial.print("ax: ");
  // Serial.println(receivedPacket.ax, 6);
  // Serial.print("ay: ");
  // Serial.println(receivedPacket.ay, 6);
  // Serial.print("az: ");
  // Serial.println(receivedPacket.az, 6);

  // Serial.println("=======================");

  Radio.Rx(0); // keep listening
}

void OnRxTimeout(void) {
  // Serial.println("RX Timeout");
  Radio.Rx(0);
}

void OnRxError(void) {
  // Serial.println("RX Error");
  Radio.Rx(0);
}
