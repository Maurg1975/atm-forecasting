"""
Atmospheric Parameters Forecasting with Neural Networks

This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
The program establishes a serial connection with an Arduino device to collect atmospheric data,
and then sends this data to a remote server via a TCP/IP socket connection. This setup is intended
for scenarios where atmospheric data is gathered remotely and processed on a server, potentially 
feeding into machine learning models for atmospheric forecasting.
Developed with assistance from AI-based chatbot tools such as Claude by Anthropic, 
to streamline coding and implementation processes.

Copyright (C) 2024 Maurg1975
Licensed under the GNU General Public License v3.0 (GPLv3)

Compatible with:
- Windows, macOS, and Linux (depending on the serial port and server configurations)
- Arduino Uno, Arduino Mega, Arduino Nano
"""

import socket
import time
from datetime import datetime
import serial

###### REPLACE WITH YOUR ARDUINO AND NGROK DATA ######

# Configure the serial connection to Arduino
SERIAL_PORT = 'COM3'  # Replace with the correct serial port
BAUD_RATE = 9600

# Server address and port
SERVER_ADDRESS = "0.tcp.ngrok.io"  # Replace with your server's address
SERVER_PORT = 13212  # Replace with your server's port

######################################################

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Initialize the serial connection to Arduino
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)

try:
    # Connect to the server
    client_socket.connect((SERVER_ADDRESS, SERVER_PORT))
    print(f"Connected to server {SERVER_ADDRESS}:{SERVER_PORT}")

    while True:
        if arduino.in_waiting > 0:
            # Read data from Arduino
            arduino_data = arduino.readline().decode('utf-8').strip()
            
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create the message to send
            message = f"Timestamp: {timestamp}, {arduino_data}\n"

            # Send the message to the server
            client_socket.sendall(message.encode())
            print(f"Sent to server: {message}")

            # Wait for one second before sending the next message
            time.sleep(1)

except Exception as e:
    print(f"Error during server connection: {e}")

finally:
    # Close the socket
    client_socket.close()
    print("Connection closed with the server")
    
    # Close the serial connection to Arduino
    arduino.close()
    print("Connection closed with Arduino")
