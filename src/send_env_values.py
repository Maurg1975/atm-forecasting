"""
Atmospheric Parameters Forecasting with Neural Networks

This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
The program establishes a serial connection with an Arduino device to collect atmospheric data,
and then sends this data to a remote server via a TCP/IP socket connection. This setup is intended
for scenarios where atmospheric data is gathered remotely and processed on a server, potentially 
feeding into machine learning models for atmospheric forecasting.
Developed with assistance from AI-based chatbot tools such as Claude by Anthropic and ChatGPT by OpenAI, 
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

###### DEFAULT CONFIGURATION ######

DEFAULT_SERIAL_PORT = 'COM4'
DEFAULT_BAUD_RATE = 9600
DEFAULT_SERVER_ADDRESS = "4.tcp.ngrok.io"
DEFAULT_SERVER_PORT = 10512

###################################

# Prompt the user for the serial port and baud rate
serial_port = input(f"Enter the serial port [default: {DEFAULT_SERIAL_PORT}]: ") or DEFAULT_SERIAL_PORT
baud_rate = input(f"Enter the baud rate [default: {DEFAULT_BAUD_RATE}]: ") or DEFAULT_BAUD_RATE
baud_rate = int(baud_rate)

# Prompt the user for the server address and port
server_address = input(f"Enter the server address [default: {DEFAULT_SERVER_ADDRESS}]: ") or DEFAULT_SERVER_ADDRESS
server_port = input(f"Enter the server port [default: {DEFAULT_SERVER_PORT}]: ") or DEFAULT_SERVER_PORT
server_port = int(server_port)

######################################################

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Initialize the serial connection to Arduino
try:
    arduino = serial.Serial(serial_port, baud_rate)
except serial.SerialException as e:
    print(f"Failed to connect to Arduino on {serial_port} at {baud_rate} baud: {e}")
    exit(1)

try:
    # Connect to the server
    client_socket.connect((server_address, server_port))
    print(f"Connected to server {server_address}:{server_port}")

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
