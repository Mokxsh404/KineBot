#include <Arduino.h>
#include <Wire.h>

#define I2C_SDA D4
#define I2C_SCL D5

#define PIN_PWMA D0
#define PIN_AIN1 D2
#define PIN_AIN2 D1
#define PIN_STBY D3
#define PIN_BIN1 D8
#define PIN_BIN2 D9
#define PIN_PWMB D10

#define INVERT_MOTOR_A true
#define INVERT_MOTOR_B false

#define MAX_SPEED_CAP 90

#define FALL_ANGLE 40.0

const uint8_t MPU6500_ADDR = 0x68;
const uint8_t REG_CONFIG = 0x1A;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_CONFIG_2 = 0x1D;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_GYRO_XOUT_H = 0x43;
const uint8_t REG_PWR_MGMT_1 = 0x6B;
const uint8_t REG_PWR_MGMT_2 = 0x6C;
const uint8_t REG_WHO_AM_I = 0x75;

const float GYRO_SENSITIVITY = 65.5;
const float ACCEL_SENSITIVITY = 8192.0;

float accelOffsetX = 0.0, accelOffsetY = 0.0, accelOffsetZ = 0.0;
float gyroOffsetX = 0.0, gyroOffsetY = 0.0, gyroOffsetZ = 0.0;

double pitch = 0.0;
unsigned long prevTime = 0;

double Kp = 15.0;
double Ki = 0.0;
double Kd = 0.8;
double setpoint = 0.0;

double error = 0.0;
double lastError = 0.0;
double integral = 0.0;
double derivative = 0.0;

bool isFallen = true;

void initMotors() {
  pinMode(PIN_PWMA, OUTPUT);
  pinMode(PIN_AIN1, OUTPUT);
  pinMode(PIN_AIN2, OUTPUT);
  pinMode(PIN_PWMB, OUTPUT);
  pinMode(PIN_BIN1, OUTPUT);
  pinMode(PIN_BIN2, OUTPUT);
  pinMode(PIN_STBY, OUTPUT);

  digitalWrite(PIN_STBY, LOW);
  analogWrite(PIN_PWMA, 0);
  analogWrite(PIN_PWMB, 0);

  digitalWrite(PIN_AIN1, LOW);
  digitalWrite(PIN_AIN2, LOW);
  digitalWrite(PIN_BIN1, LOW);
  digitalWrite(PIN_BIN2, LOW);
}

void setStandby(bool active) { digitalWrite(PIN_STBY, active ? LOW : HIGH); }

void driveMotorA(int speed) {
  if (INVERT_MOTOR_A)
    speed = -speed;
  speed = constrain(speed, -MAX_SPEED_CAP, MAX_SPEED_CAP);

  if (speed == 0) {
    digitalWrite(PIN_AIN1, LOW);
    digitalWrite(PIN_AIN2, LOW);
    analogWrite(PIN_PWMA, 0);
  } else if (speed > 0) {
    digitalWrite(PIN_AIN1, HIGH);
    digitalWrite(PIN_AIN2, LOW);
    analogWrite(PIN_PWMA, speed);
  } else {
    digitalWrite(PIN_AIN1, LOW);
    digitalWrite(PIN_AIN2, HIGH);
    analogWrite(PIN_PWMA, -speed);
  }
}

void driveMotorB(int speed) {
  if (INVERT_MOTOR_B)
    speed = -speed;
  speed = constrain(speed, -MAX_SPEED_CAP, MAX_SPEED_CAP);

  if (speed == 0) {
    digitalWrite(PIN_BIN1, LOW);
    digitalWrite(PIN_BIN2, LOW);
    analogWrite(PIN_PWMB, 0);
  } else if (speed > 0) {
    digitalWrite(PIN_BIN1, HIGH);
    digitalWrite(PIN_BIN2, LOW);
    analogWrite(PIN_PWMB, speed);
  } else {
    digitalWrite(PIN_BIN1, LOW);
    digitalWrite(PIN_BIN2, HIGH);
    analogWrite(PIN_PWMB, -speed);
  }
}

void drive(int speedA, int speedB) {
  setStandby(false);
  driveMotorA(speedA);
  driveMotorB(speedB);
}

void brake() {
  digitalWrite(PIN_AIN1, HIGH);
  digitalWrite(PIN_AIN2, HIGH);
  digitalWrite(PIN_BIN1, HIGH);
  digitalWrite(PIN_BIN2, HIGH);
  analogWrite(PIN_PWMA, MAX_SPEED_CAP);
  analogWrite(PIN_PWMB, MAX_SPEED_CAP);
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

bool readRegisters(uint8_t startReg, uint8_t *dest, uint8_t count) {
  Wire.beginTransmission(MPU6500_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0)
    return false;

  uint8_t bytesReceived = Wire.requestFrom(MPU6500_ADDR, count);
  if (bytesReceived != count)
    return false;

  for (uint8_t i = 0; i < count; i++) {
    dest[i] = Wire.read();
  }
  return true;
}

bool initMPU6500() {
  uint8_t whoAmI = 0;
  if (!readRegisters(REG_WHO_AM_I, &whoAmI, 1))
    return false;

  if (!writeRegister(REG_PWR_MGMT_1, 0x01))
    return false;
  delay(10);
  if (!writeRegister(REG_PWR_MGMT_2, 0x00))
    return false;
  delay(10);

  if (!writeRegister(REG_CONFIG, 0x03))
    return false;
  if (!writeRegister(REG_ACCEL_CONFIG_2, 0x03))
    return false;
  if (!writeRegister(REG_GYRO_CONFIG, 0x08))
    return false;
  if (!writeRegister(REG_ACCEL_CONFIG, 0x08))
    return false;

  return true;
}

void calibrateMPU6500() {
  Serial.println("\n=== MPU-6500 Balancing Calibration ===");
  Serial.println("Place the robot flat and upright. Do NOT move it!");
  for (int i = 5; i > 0; i--) {
    Serial.print("Calibrating in ");
    Serial.print(i);
    Serial.println("...");
    delay(1000);
  }

  long sumAccX = 0, sumAccY = 0, sumAccZ = 0;
  long sumGyrX = 0, sumGyrY = 0, sumGyrZ = 0;
  const int numSamples = 200;
  int validSamples = 0;
  uint8_t buffer[14];

  while (validSamples < numSamples) {
    if (readRegisters(REG_ACCEL_XOUT_H, buffer, 14)) {
      int16_t rawAccX = (buffer[0] << 8) | buffer[1];
      int16_t rawAccY = (buffer[2] << 8) | buffer[3];
      int16_t rawAccZ = (buffer[4] << 8) | buffer[5];
      int16_t rawGyrY = (buffer[10] << 8) | buffer[11];

      sumAccX += rawAccX;
      sumAccY += rawAccY;
      sumAccZ += (rawAccZ - ACCEL_SENSITIVITY);
      sumGyrY += rawGyrY;

      validSamples++;
      delay(5);
    }
  }

  accelOffsetX = (float)sumAccX / numSamples;
  accelOffsetY = (float)sumAccY / numSamples;
  accelOffsetZ = (float)sumAccZ / numSamples;
  gyroOffsetY = (float)sumGyrY / numSamples;

  Serial.println("[SUCCESS] Calibration complete!");
}

void parseSerialCommands() {
  if (Serial.available()) {
    char key = Serial.read();
    float val = Serial.parseFloat();

    switch (key) {
    case 'p':
      Kp = val;
      break;
    case 'i':
      Ki = val;
      break;
    case 'd':
      Kd = val;
      break;
    case 's':
      setpoint = val;
      break;
    case 'r':
      calibrateMPU6500();
      pitch = 0;
      integral = 0;
      lastError = 0;
      break;
    }

    Serial.print("[TUNING] Updated -> P: ");
    Serial.print(Kp);
    Serial.print(" | I: ");
    Serial.print(Ki);
    Serial.print(" | D: ");
    Serial.print(Kd);
    Serial.print(" | Setpoint: ");
    Serial.println(setpoint);
  }
}

void setup() {
  Serial.begin(115200);
  initMotors();

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  while (!initMPU6500()) {
    Serial.println("[ERROR] Failed to locate MPU-6500. Retrying in 2s...");
    delay(2000);
  }

  calibrateMPU6500();
  prevTime = micros();
  Serial.println("\nPID Tuning commands available in input field:");
  Serial.println("  p<val>  (e.g., p18.5) -> Update Kp");
  Serial.println("  i<val>  (e.g., i0.2)  -> Update Ki");
