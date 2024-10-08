/*
 * Atmospheric Parameters Forecasting with Neural Networks
 * 
 * This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
 * The project utilizes Arduino-compatible sensors to monitor temperature, humidity, atmospheric pressure, and light intensity,
 * with the collected data intended for use in machine learning models, specifically neural networks,
 * to predict atmospheric conditions.
 * Developed with assistance from AI-based chatbot tools such as ChatGPT by OpenAI
 * to streamline coding and implementation processes.
 * 
 * Copyright (C) 2024 Maurg1975
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * 
 * This program is compatible with the following Arduino models:
 * - Arduino Mega
 * - Arduino Nano
 * 
 * Sensors used:
 * - DHT11 or DHT22: Measures temperature and humidity.
 * - BMP180: Measures atmospheric pressure.
 * - MQ-2: Measures gas concentration (LPG, methane, smoke).
 * - MQ-135: Measures gas concentration (CO2, ammonia, alcohol, benzene).
 * - Anemometer: Measures wind speed.
 * - Wind Direction Sensor: Measures wind direction.
 * - BH1750: Measures light intensity in lux.
 * - Rain sensor
 */

#include <DHT.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085.h>
#include <BH1750.h>  // Include the BH1750 library

// Define the digital pin connected to the DHT sensor
#define DHTPIN 2     

// Define the type of DHT sensor used (DHT11 or DHT22)
#define DHTTYPE DHT11   

// Define the analog pins connected to the MQ-2, MQ-135, Anemometer, and Wind Direction sensors
#define ANEMOMETERPIN A0
#define WINDDIRECTIONPIN A1
#define MQ2PIN A2
#define MQ135PIN A3
#define RAINSENSORANALOG A6

// Initialize the DHT sensor with the specified pin and type
DHT dht(DHTPIN, DHTTYPE);

// Initialize the BMP180 sensor for pressure measurement (I2C protocol)
Adafruit_BMP085 bmp; 
bool disablePressureMeter = false;

// Initialize the BH1750 sensor for light intensity measurement (I2C protocol)
BH1750 lightMeter;
bool disableLightMeter = false;

// The setup function runs once when you press reset or power the board
void setup() {
  // Start the serial communication at 9600 baud rate
  Serial.begin(9600);

  // Start the DHT sensor
  dht.begin();

  // Start the BMP180 sensor and check if it's detected correctly
  if (!bmp.begin()) {
    Serial.println(F("Could not find a valid BMP180 sensor, check wiring!"));
    disablePressureMeter = true;
  }

  // Start the BH1750 sensor and check if it's detected correctly
  if (!lightMeter.begin()) {
    Serial.println(F("Could not find a valid BH1750 sensor, check wiring!"));
	  disableLightMeter = true;
  }

  // Optional: Wait for MQ-2 and MQ-135 sensor warm-up (e.g., 2 minutes)
//  Serial.println(F("Warming up MQ-2 and MQ-135 sensors..."));
//  delay(120000); // 2 minutes warm-up
//  Serial.println(F("MQ-2 and MQ-135 sensors are ready."));
}

// The loop function runs over and over again forever
void loop() {
  // Read temperature from the DHT sensor
  float temperature = dht.readTemperature();
  if (isnan(temperature)) {
    temperature = 0;
  }

  // Read humidity from the DHT sensor
  float humidity = dht.readHumidity();
  if (isnan(humidity)) {
    humidity = 0;
  }

  // Read pressure from the BMP180 sensor and convert it to hPa
  float pressure = 0;
  if (!disablePressureMeter) {
    pressure = bmp.readPressure();
    if (!isnan(pressure) && (pressure > 0)) {
	    pressure = pressure / 100.0F;
    }
  }

  // Read light intensity from the BH1750 sensor
  float lightLevel = 0;
  if (!disableLightMeter) {
	  lightLevel = lightMeter.readLightLevel();
	  if (lightLevel < 0) {
	    lightLevel = 0;
	  }
  }
  
  // Read gas concentration from the MQ-2 sensor
  int mq2Value = analogRead(MQ2PIN);
  float mq2Voltage = mq2Value * (5.0 / 1023.0); // Convert to voltage

  // Read gas concentration from the MQ-135 sensor
  int mq135Value = analogRead(MQ135PIN);
  float mq135Voltage = mq135Value * (5.0 / 1023.0); // Convert to voltage

  // Read wind speed from the anemometer
  int anemometerValue = analogRead(ANEMOMETERPIN);
  float windSpeed = (anemometerValue * (5.0 / 1023.0)) * (30.0 / 5.0); // Convert to m/s

  // Read wind direction from the wind direction sensor
  int windDirectionValue = analogRead(WINDDIRECTIONPIN);
  float windDirection = (windDirectionValue * (5.0 / 1023.0)) * (360.0 / 5.0); // Convert to degrees

  // Read value from the rain sensor
  int rainSensorValue = analogRead(RAINSENSORANALOG);
  float rainIntensity = (1023.0 - rainSensorValue) * (5.0 / 1023.0);

  // Print the temperature, humidity, pressure, light intensity, gas concentration, wind speed, and wind direction values to the serial monitor
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print(", Humidity: ");
  Serial.print(humidity);
  Serial.print(", Pressure: ");
  Serial.print(pressure);
  Serial.print(", Light Intensity: ");
  Serial.print(lightLevel);
  Serial.print(", MQ-2 Voltage: ");
  Serial.print(mq2Voltage);
  Serial.print(", MQ-135 Voltage: ");
  Serial.print(mq135Voltage);
  Serial.print(", Wind Speed: ");
  Serial.print(windSpeed);
  Serial.print(", Wind Direction: ");
  Serial.print(windDirection);
  Serial.print(", Rain Intensity: ");
  Serial.println(rainIntensity);

  // Wait for 2 seconds before taking another reading
  delay(2000);
}
