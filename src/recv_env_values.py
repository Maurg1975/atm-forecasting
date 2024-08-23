"""
Atmospheric Parameters Forecasting with Neural Networks

This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
The program sets up a server to receive real-time atmospheric data from clients, 
processes this data using a Long Short-Term Memory (LSTM) neural network, and 
provides predictions on temperature, humidity, and pressure. The server is exposed 
to the public via ngrok, allowing remote devices to send data. The processed data and 
predictions are then visualized using Matplotlib.
Developed with assistance from AI-based chatbot tools such as ChatGPT by OpenAI
to streamline coding and implementation processes.

Copyright (C) 2024 Maurg1975
Licensed under the GNU General Public License v3.0 (GPLv3)

Requirements:
- Python environment with packages: socket, threading, pyngrok, numpy, pandas, sklearn, tensorflow, matplotlib
- ngrok account for tunneling
"""

#### UNCOMMENT NEXT LINE IF YOU RUN ON GOOGLE COLAB ####

# !pip install pyngrok

########################################################

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
server_socket.bind(("localhost", ngrok_port))
server_socket.listen(5)
print("Server listening on localhost")

# Global variables for model training
seq_length = 10
columns = ['timestamp', 'temperature', 'humidity', 'pressure']
data = pd.DataFrame(columns=columns)
scaler = MinMaxScaler()
model = None
predictions = []

# Function to build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 3)))
    model.add(LSTM(50))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Initialize the LSTM model
model = build_model()

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
    recent_data_df = pd.DataFrame(recent_data, columns=['temperature', 'humidity', 'pressure'])
    recent_data_scaled = scaler.transform(recent_data_df)
    recent_data_scaled = np.array([recent_data_scaled])
    prediction_scaled = model.predict(recent_data_scaled)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0]

# Function to plot real and predicted data
def plot_data(data, predictions):
    plt.figure(figsize=(14, 7))

    # Plot real and predicted temperature
    plt.subplot(3, 1, 1)
    plt.plot(data['timestamp'], data['temperature'], label='Real Temperature', color='blue')
    plt.plot(data['timestamp'][seq_length:], [p[0] for p in predictions], label='Predicted Temperature', color='red')
    plt.title('Temperature')
    plt.legend()

    # Plot real and predicted humidity
    plt.subplot(3, 1, 2)
    plt.plot(data['timestamp'], data['humidity'], label='Real Humidity', color='blue')
    plt.plot(data['timestamp'][seq_length:], [p[1] for p in predictions], label='Predicted Humidity', color='red')
    plt.title('Humidity')
    plt.legend()

    # Plot real and predicted pressure
    plt.subplot(3, 1, 3)
    plt.plot(data['timestamp'], data['pressure'], label='Real Pressure', color='blue')
    plt.plot(data['timestamp'][seq_length:], [p[2] for p in predictions], label='Predicted Pressure', color='red')
    plt.title('Pressure')
    plt.legend()

    plt.tight_layout()
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
                    if len(parts) != 4:
                        print(f"Invalid data format (incorrect number of parts): {message}")
                        continue
                    timestamp_str, temp_str, hum_str, pres_str = parts
                    timestamp = datetime.strptime(timestamp_str.split(": ")[1], "%Y-%m-%d %H:%M:%S")
                    temperature = float(temp_str.split(": ")[1])
                    humidity = float(hum_str.split(": ")[1])
                    pressure = float(pres_str.split(": ")[1])
                except (ValueError, IndexError) as e:
                    print(f"Error processing message: {message} ({e})")
                    continue

                # Append new data to the DataFrame
                new_data = pd.DataFrame([[timestamp, temperature, humidity, pressure]], columns=columns)
                data = pd.concat([data, new_data], ignore_index=True)

                # If enough data is available, train the model and make predictions
                if len(data) > seq_length:
                    # Select only the relevant columns for prediction
                    data_for_training = data[['temperature', 'humidity', 'pressure']]

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

                    print(f"Predicted Temperature: {predicted_weather[0]} °C")
                    print(f"Predicted Humidity: {predicted_weather[1]} %")
                    print(f"Predicted Pressure: {predicted_weather[2]} hPa")

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