"""
Atmospheric Parameters Forecasting with Neural Networks

This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
The program sets up a server to receive real-time atmospheric data from clients, 
processes this data using a Long Short-Term Memory (LSTM) neural network, and 
provides predictions on temperature, humidity, pressure, gas concentration, 
wind speed, and wind direction. The server is exposed to the public via ngrok, 
allowing remote devices to send data. The processed data and predictions are 
then visualized using Matplotlib.
Developed with assistance from AI-based chatbot tools such as ChatGPT by OpenAI
to streamline coding and implementation processes.

Copyright (C) 2024 Maurg1975
Licensed under the GNU General Public License v3.0 (GPLv3)
"""

import socket
import threading
from pyngrok import ngrok
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from IPython.display import clear_output

############ REPLACE WITH YOUR NGROK DATA ############

ngrok_auth_token = ""  # Replace with your ngrok auth token
ngrok_port = 8000

######################################################

# Set up ngrok authentication token and start the tunnel
ngrok.set_auth_token(ngrok_auth_token)
public_url = ngrok.connect(ngrok_port, proto="tcp")
print(f"Server publicly exposed at: {public_url}")

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server_socket.bind(("localhost", ngrok_port))
except socket.error as e:
    print(f"Failed to bind to port {ngrok_port}: {e}")
    exit(1)
server_socket.listen(5)
print("Server listening on localhost")

# Global variables for model training
seq_length = 10
columns = ['timestamp', 'temperature', 'humidity', 'pressure', 'light_intensity', 'mq2_voltage', 'mq135_voltage', 'wind_speed', 'wind_direction']
selected_sensors = ['temperature', 'humidity', 'pressure', 'light_intensity', 'mq2_voltage', 'mq135_voltage', 'wind_speed', 'wind_direction']  # Default selection
data = pd.DataFrame(columns=columns)
scaler = MinMaxScaler()
model = None
predictions = []

# Function to select sensors
def select_sensors():
    global selected_sensors
    print("Available sensors: temperature, humidity, pressure, light_intensity, mq2_voltage, mq135_voltage, wind_speed, wind_direction")
    selected_sensors = input("Enter the sensors you want to use, separated by commas (e.g., temperature, humidity, pressure): ").split(", ")
    if 'timestamp' not in selected_sensors:
        selected_sensors.insert(0, 'timestamp')
    print(f"Selected sensors: {selected_sensors}")

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, input_shape)))
    model.add(LSTM(50))
    model.add(Dense(len(selected_sensors) - 1))  # Output neurons equal to the number of selected sensors minus the timestamp
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Initialize the LSTM model after selecting sensors
select_sensors()
model = build_model(len(selected_sensors) - 1)

# Function to create sequences from data
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to update the LSTM model with new data
def update_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=1, batch_size=10, verbose=0)
    return model

# Function to predict future data using the LSTM model
def predict_weather_lstm(model, recent_data):
    recent_data_df = pd.DataFrame(recent_data, columns=selected_sensors[1:])
    recent_data_scaled = scaler.transform(recent_data_df)
    recent_data_scaled = np.array([recent_data_scaled])
    prediction_scaled = model.predict(recent_data_scaled, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0]

# Function to plot real and predicted data
def plot_data(data, predictions):
    clear_output(wait=True)  # This will clear the previous output before plotting new graphs
    
    num_sensors = len(selected_sensors) - 1  # Exclude timestamp
    rows = (num_sensors + 1) // 2  # Calculate the number of rows needed (2 plots per row)
    
    plt.figure(figsize=(14, rows * 2))  # Adjust figure size based on number of rows

    for i, sensor in enumerate(selected_sensors[1:]):  # Skip timestamp
        plt.subplot(rows, 2, i + 1)  # Create subplot grid with 2 columns
        plt.plot(data['timestamp'], data[sensor], label=f'Real {sensor.capitalize()}', color='blue')
        plt.plot(data['timestamp'][seq_length:], [p[i] for p in predictions], label=f'Predicted {sensor.capitalize()}', color='red')
        plt.title(sensor.capitalize())
        plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Function to handle client connections
def handle_client(client_socket, client_address):
    global data, model, predictions  # Declare model and predictions as global variables
    buffer = ""
    try:
        print(f"Connection accepted from {client_address}")
        while True:
            data_received = client_socket.recv(1024).decode()
            if not data_received:
                break
            buffer += data_received
            while "\n" in buffer:
                message, buffer = buffer.split("\n", 1)
                try:
                    parts = message.split(', ')
                    if len(parts) != len(columns):  # Now expecting parts matching all columns
                        print(f"Invalid data format (incorrect number of parts): {message}")
                        continue
                    timestamp_str, temp_str, hum_str, pres_str, light_str, mq2_str, mq135_str, wind_speed_str, wind_dir_str = parts
                    timestamp = datetime.strptime(timestamp_str.split(": ")[1], "%Y-%m-%d %H:%M:%S")
                    data_dict = {
                        'timestamp': timestamp,
                        'temperature': float(temp_str.split(": ")[1]),
                        'humidity': float(hum_str.split(": ")[1]),
                        'pressure': float(pres_str.split(": ")[1]),
                        'light_intensity': float(light_str.split(": ")[1].split(' ')[0]),
                        'mq2_voltage': float(mq2_str.split(": ")[1]),
                        'mq135_voltage': float(mq135_str.split(": ")[1]),
                        'wind_speed': float(wind_speed_str.split(": ")[1].split(' ')[0]),
                        'wind_direction': float(wind_dir_str.split(": ")[1].split(' ')[0])
                    }
                    selected_data = [data_dict[sensor] for sensor in selected_sensors]

                except (ValueError, IndexError) as e:
                    print(f"Error processing message: {message} ({e})")
                    continue

                # Append new data to the DataFrame
                new_data = pd.DataFrame([selected_data], columns=selected_sensors)
                data = pd.concat([data, new_data], ignore_index=True)

                # If enough data is available, train the model and make predictions
                if len(data) > seq_length:
                    # Select only the relevant columns for prediction
                    data_for_training = data[selected_sensors[1:]]  # Skip timestamp

                    # Normalize the data
                    data_normalized = scaler.fit_transform(data_for_training)

                    # Create sequences
                    X, y = create_sequences(data_normalized, seq_length)

                    # Update the model
                    model = update_model(model, X, y)

                    # Predict the next value
                    recent_data = data_for_training[-seq_length:].values
                    predicted_weather = predict_weather_lstm(model, recent_data)

                    predictions.append(predicted_weather)

#                    print(f"Predicted {selected_sensors[1]}: {predicted_weather[0]}")
#                    for i, sensor in enumerate(selected_sensors[2:]):
#                        print(f"Predicted {sensor.capitalize()}: {predicted_weather[i+1]}")

                    # Visualize the data
                    plot_data(data, predictions)

    except Exception as e:
        print(f"Error during connection with {client_address}: {e}")
    finally:
        client_socket.close()
        print(f"Connection closed with {client_address}")

# Main server loop
while True:
    print("Waiting for connections...")
    client_socket, client_address = server_socket.accept()
    client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
    client_thread.start()
