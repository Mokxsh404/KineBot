/**
 * @file xiao_esp32s3_balancing_car_web.ino
 * @brief Self-Balancing Human-Following Robot — Shared WiFi Station with Video Dashboard.
 *
 * Architecture:
 * - XIAO connects to shared WiFi (SSID: FAST_AND_FURIOUS).
 * - Hosts a web dashboard with live MJPEG camera stream at port 80.
 * - MJPEG stream server on port 81 (/stream).
 * - Registers as kinebot.local via mDNS.
 * - PID balancing loop runs on Core 0 at 200Hz (dedicated RTOS task).
 * - WebServer runs on Core 1 (main loop) with no delays.
 * - Always-Balancing: PID active at all times, commands shift target angle.
 * - Arduino Uno Q connects to the same WiFi, resolves kinebot.local,
 *   fetches the stream, runs YOLOv8, and sends /track commands back.
 * - Speed capped at 180 with deadband for reliable balance.
 */

#include <WebServer.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <Wire.h>

// Camera libraries
#include "esp_camera.h"
#include "esp_http_server.h"

// Camera pin mapping for Seeed Studio XIAO ESP32S3 Sense
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ==========================================
// 1. PIN DEFINITIONS  (L298N Motor Driver)
// ==========================================
#define I2C_SDA D4
#define I2C_SCL D5

// L298N wiring
#define PIN_ENA   D0   // PWM speed control — Motor A
#define PIN_IN1   D1   // Motor A direction
#define PIN_IN2   D2   // Motor A direction
#define PIN_IN3   D3   // Motor B direction
#define PIN_IN4   D9   // Motor B direction
#define PIN_ENB   D8   // PWM speed control — Motor B

#define INVERT_MOTOR_A true
#define INVERT_MOTOR_B true

// ==========================================
// 2. LIMITS
// ==========================================
#define MAX_SPEED_CAP 180
#define FALL_ANGLE 35.0     // Max degrees AWAY from setpoint before stall
#define MOTOR_DEADBAND 25   // Minimum PWM to overcome static friction

// ==========================================
// 3. MPU-6500 & SENSOR ORIENTATION
// ==========================================
const uint8_t MPU6500_ADDR       = 0x68;
const uint8_t REG_CONFIG         = 0x1A;
const uint8_t REG_GYRO_CONFIG    = 0x1B;
const uint8_t REG_ACCEL_CONFIG   = 0x1C;
const uint8_t REG_ACCEL_CONFIG_2 = 0x1D;
const uint8_t REG_ACCEL_XOUT_H   = 0x3B;
const uint8_t REG_GYRO_XOUT_H    = 0x43;
const uint8_t REG_PWR_MGMT_1     = 0x6B;
const uint8_t REG_PWR_MGMT_2     = 0x6C;
const uint8_t REG_WHO_AM_I       = 0x75;

const float GYRO_SENSITIVITY  = 65.5;
const float ACCEL_SENSITIVITY = 8192.0;

// Offsets
float accelOffsetX = 0, accelOffsetY = 0, accelOffsetZ = 0;
float gyroOffsetX  = 0, gyroOffsetY  = 0, gyroOffsetZ  = 0;

// --- SENSOR MOUNTING CONFIGURATION ---
// Adjust these to match how your MPU sensor is physically placed on your robot:
#define USE_X_AXIS          false // Set to true if tilting forward/backward rotates around X axis instead of Y axis
#define INVERT_TILT_ANGLE   false // Set to true if tilting forward shows a negative angle instead of positive
#define INVERT_GYRO_SIGN    false // Set to true if the angle drifts/climbs when holding the robot tilted
// -------------------------------------

volatile double pitch = 0.0;
volatile double debug_pa = 0.0;
volatile double debug_gyro = 0.0;
volatile unsigned long lastWebActivity = 0;  // Dashboard watchdog
volatile unsigned long lastAIActivity = 0;   // AI tracker watchdog

enum RobotState { STATE_STALL, STATE_BALANCING };
volatile RobotState currentState = STATE_STALL;

enum MoveCommand { CMD_NONE, CMD_FORWARD, CMD_BACKWARD, CMD_LEFT, CMD_RIGHT };
volatile MoveCommand currentCommand = CMD_NONE;
volatile int driveInput = 0; // -1 = backward, 0 = stop, 1 = forward
volatile int steerInput = 0; // -1 = left, 0 = stop, 1 = right

// Tracking mode: AI commands take priority over manual when active
enum TrackingSource { SRC_MANUAL, SRC_AI };
volatile TrackingSource activeSource = SRC_MANUAL;
volatile bool aiTrackingEnabled = true;  // Can be toggled from dashboard

// ==========================================
// 4. PID
// ==========================================
volatile double Kp = 16.5;
volatile double Ki = 0.10;
volatile double Kd = 0.85;
volatile double setpoint = -14.7;
volatile int steerPower = 45;      // Steering power (adjustable from dashboard)
volatile int drivePower = 40;      // Drive feedforward power (adjustable from dashboard)

double pidError = 0, lastError = 0, integral = 0, derivative = 0;

// ==========================================
// 5. WI-FI  (Station mode with AP fallback)
// ==========================================
const char* ssid         = "FAST_AND_FURIOUS";
const char* password     = "ravi@7341";
const char* fallback_ssid = "KINE-BOT-FALLBACK";
const char* fallback_pass = "root@123";

WebServer server(80);

// ==========================================
// 6. HTML DASHBOARD  (with live MJPEG video)
// ==========================================
const char html_page[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KINE-BOT Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
body {
  background: #090d16;
  color: #f0f6fc;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
  padding: 20px;
  overflow-x: hidden;
}
h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin-bottom: 20px;
  background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 1px;
  text-shadow: 0 0 20px rgba(79, 172, 254, 0.3);
}
.c {
  width: 100%;
  max-width: 480px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.card {
  background: rgba(21, 26, 38, 0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.ct {
  font-size: 1.2rem;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.badge {
  font-size: 0.8rem;
  padding: 4px 10px;
  border-radius: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.b-stall {
  background: rgba(255, 42, 133, 0.15);
  color: #ff2a85;
  border: 1px solid rgba(255, 42, 133, 0.3);
  box-shadow: 0 0 10px rgba(255, 42, 133, 0.2);
}
.b-bal {
  background: rgba(0, 242, 254, 0.15);
  color: #00f2fe;
  border: 1px solid rgba(0, 242, 254, 0.3);
  box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
}
/* Video feed */
.vid-wrap {
  position: