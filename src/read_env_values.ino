/*
 * Atmospheric Parameters Forecasting with Neural Networks
 * 
 * This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
 * The project utilizes Arduino-compatible sensors to monitor temperature, humidity, and atmospheric pressure,
 * with the collected data intended for use in machine learning models, specifically neural networks,
 * to predict atmospheric conditions.
 * Developed with assistance from AI-based chatbot tools such as ChatGPT by OpenAI
 * to streamline coding and implementation processes.
 * 
 * Copyright (C) 2024 Maurg1975
 * Licensed under the GNU General Public License v3.0 (GPLv3)
 * 
 * This program is compatible with the following Arduino models:
 * - Arduino Uno
 * - Arduino Mega
 * - Arduino Nano
 * 
 * Sensors used:
 * - DHT11 or DHT22: Measures temperature and humidity.
 * - BMP180: Measures atmospheric pressure.
 */

#include <DHT.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085.h>

// Define the digital pin connected to the DHT sensor
#define DHTPIN 2     

// Define the type of DHT sensor used (DHT11 or DHT22)
#define DHTTYPE DHT11   

// Initialize the DHT sensor with the specified pin and type
DHT dht(DHTPIN, DHTTYPE);

// Initialize the BMP180 sensor for pressure measurement (I2C protocol)
Adafruit_BMP085 bmp; 

// The setup function runs once when you press reset or power the board
void setup() {
  // Start the serial communication at 9600 baud rate
  Serial.begin(9600);

  // Start the DHT sensor
  dht.begin();

  // Start the BMP180 sensor and check if it's detected correctly
  if (!bmp.begin()) {
    Serial.println(F("Could not find a valid BMP180 sensor, check wiring!"));
    // If the sensor is not found, stop the program here
    while (1);
  }
}

// The loop function runs over and over again forever
void loop() {
  // Read temperature from the DHT sensor
  float temperature = dht.readTemperature();

  // Read humidity from the DHT sensor
  float humidity = dht.readHumidity();

  // Read pressure from the BMP180 sensor and convert it to hPa
  float pressure = bmp.readPressure() / 100.0F;

  // Check if the temperature or humidity readings are valid
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    // If readings are invalid, return to the start of the loop
    return;
  }

  // Print the temperature, humidity, and pressure values to the serial monitor
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print(", Humidity: ");
  Serial.print(humidity);
  Serial.print(", Pressure: ");
  Serial.println(pressure);

  // Wait for 2 seconds before taking another reading
  delay(2000);
}
