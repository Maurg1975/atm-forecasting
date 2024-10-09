"""
Atmospheric Parameters Forecasting with Neural Networks

This file is part of the "Atmospheric Parameters Forecasting with Neural Networks" project.
The program sets up a server to receive real-time atmospheric data from clients,
processes this data using a Long Short-Term Memory (LSTM) neural network, and
provides predictions on a single selected parameter based on selected input parameters.
The server is exposed to the public via ngrok, allowing remote devices to send data.
The processed data and predictions are then visualized using Matplotlib.
Developed with assistance from AI-based chatbot tools such as ChatGPT by OpenAI
to streamline coding and implementation processes.

Copyright (C) 2024 Maurg1975
Licensed under the GNU General Public License v3.0 (GPLv3)
"""

#!pip install pyngrok

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
from matplotlib.dates import DateFormatter, AutoDateLocator
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
columns = ['timestamp', 'temperature', 'humidity', 'pressure', 'light_intensity',
           'mq2_voltage', 'mq135_voltage', 'wind_speed', 'wind_direction', 'rain_intensity']
available_sensors = ['temperature', 'humidity', 'pressure', 'light_intensity',
                     'mq2_voltage', 'mq135_voltage', 'wind_speed', 'wind_direction', 'rain_intensity']
data = pd.DataFrame(columns=columns)

# Create separate scalers for X and y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

model = None
predictions = {}

# Variables to store the selected target sensors and input features
selected_sensors = []
input_features = []

# Variables to control model update frequency
new_data_count = 0  # Counter for new data points since last model update
model_trained = False  # Flag to indicate if the model has been trained at least once

# Function to select the sensors to predict and input features
def select_sensors():
    global selected_sensors, input_features
    print("Available sensors:")
    for idx, sensor in enumerate(available_sensors):
        print(f"{idx + 1}. {sensor}")
    # Select the target sensors
    target_choice = input("Enter the number corresponding to the sensor you want to predict (target sensor). Press Enter without typing anything to select all sensors as targets: ")
    if target_choice.strip() == '':
        selected_sensors = available_sensors.copy()
        print("All sensors selected as targets.")
    else:
        try:
            target_choice_idx = int(target_choice) - 1
            if 0 <= target_choice_idx < len(available_sensors):
                selected_sensors = [available_sensors[target_choice_idx]]
                print(f"Selected sensor to predict (target): {selected_sensors[0]}")
            else:
                print("Invalid selection. Please try again.")
                select_sensors()
                return
        except ValueError:
            print("Invalid input. Please enter a number.")
            select_sensors()
            return
    # Select the input features
    print("Select input features (you can include the target sensors if desired):")
    print("Enter the numbers corresponding to the sensors you want to use as input features, separated by commas.")
    for idx, sensor in enumerate(available_sensors):
        print(f"{idx + 1}. {sensor}")
    input_choices = input("Enter your choices: ")
    try:
        input_indices = [int(idx.strip()) - 1 for idx in input_choices.split(',')]
        if all(0 <= idx < len(available_sensors) for idx in input_indices):
            input_features = [available_sensors[idx] for idx in input_indices]
            print(f"Selected input features: {input_features}")
        else:
            print("Invalid selection. Please try again.")
            select_sensors()
            return
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        select_sensors()
        return

# Function to build the LSTM model
def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(25, return_sequences=True, input_shape=(seq_length, input_shape)))
    model.add(LSTM(25))
    model.add(Dense(output_shape))  # Output neurons for the predicted parameters
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Initialize the LSTM model after selecting the sensors
select_sensors()
update_interval = len(input_features) + 2  # Adjust the interval as needed
model = build_model(len(input_features), len(selected_sensors))

# Function to create sequences from data
def create_sequences(X_data, y_data, seq_length):
    X = []
    y = []
    for i in range(len(X_data) - seq_length):
        X.append(X_data[i:i+seq_length])
        y.append(y_data[i+seq_length])
    return np.array(X), np.array(y)

# Function to update the LSTM model with new data
def update_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=1, batch_size=10, verbose=0)
    return model

# Function to predict future data using the LSTM model
def predict_weather_lstm(model, recent_data):
    # recent_data is a NumPy array with shape (seq_length, num_features)
    recent_data_scaled = scaler_X.transform(recent_data)
    recent_data_scaled = np.array([recent_data_scaled])  # Shape: (1, seq_length, num_features)
    prediction_scaled = model.predict(recent_data_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return prediction[0]

# Function to plot real and predicted data
def plot_data(data, predictions, max_points=50):
    clear_output(wait=True)

    # Exclude the target sensors from input features when plotting
    features_to_plot = [feature for feature in input_features if feature not in selected_sensors]

    num_input_features = len(features_to_plot)
    total_plots = num_input_features + len(selected_sensors)
    cols = 3
    rows = (total_plots + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    # Limit data to most recent max_points
    data_to_plot = data.iloc[-max_points:].copy()
    data_to_plot.reset_index(drop=True, inplace=True)
    num_predictions = len(data_to_plot) - seq_length
    predictions_to_plot = {sensor: preds[-num_predictions:] for sensor, preds in predictions.items()}

    # Plot target sensors
    for idx, sensor in enumerate(selected_sensors):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.plot(data_to_plot['timestamp'], data_to_plot[sensor], label=f'Real {sensor.capitalize()}', color='blue')
        if len(predictions_to_plot.get(sensor, [])) > 0:
            ax.plot(data_to_plot['timestamp'][seq_length:], predictions_to_plot[sensor], label=f'Predicted {sensor.capitalize()}', color='red')
        ax.set_title(f"{sensor.capitalize()} Prediction")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(sensor.capitalize())
        ax.legend()
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot input features
    for idx, feature in enumerate(features_to_plot):
        ax = plt.subplot(rows, cols, len(selected_sensors) + idx + 1)
        ax.plot(data_to_plot['timestamp'], data_to_plot[feature], label=feature)
        ax.set_title(f"Input Sensor: {feature.capitalize()}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

# Function to handle client connections
def handle_client(client_socket, client_address):
    global data, model, predictions, new_data_count, model_trained  # Declare as global variables
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
                    if len(parts) != len(columns):
                        print(f"Invalid data format (incorrect number of parts): {message}")
                        continue
                    # Parse the data
                    timestamp_str, temp_str, hum_str, pres_str, light_str, mq2_str, mq135_str, \
                    wind_speed_str, wind_dir_str, rain_intensity_str = parts
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
                        'wind_direction': float(wind_dir_str.split(": ")[1].split(' ')[0]),
                        'rain_intensity': float(rain_intensity_str.split(": ")[1])
                    }

                    # Append new data to the DataFrame
                    new_data = pd.DataFrame([data_dict], columns=columns)
                    data = pd.concat([data, new_data], ignore_index=True)

                    # Increment the counter for new data points
                    new_data_count += 1

                    # If enough data is available, process it
                    if len(data) > seq_length:
                        # Prepare the input features and target variables
                        X_data = data[input_features].values
                        y_data = data[selected_sensors].values

                        # Fit the scalers on the training data
                        scaler_X.fit(X_data)
                        scaler_y.fit(y_data)

                        # Normalize the data
                        X_scaled = scaler_X.transform(X_data)
                        y_scaled = scaler_y.transform(y_data)

                        # Create sequences
                        X, y = create_sequences(X_scaled, y_scaled, seq_length)

                        if new_data_count >= update_interval:
                            # Reset new_data_count
                            new_data_count = 0

                            # Update the model
                            model = update_model(model, X, y)
                            model_trained = True

                        if model_trained:
                            # Predict the next values
                            recent_data = X_data[-seq_length:]
                            predicted_values = predict_weather_lstm(model, recent_data)

                            # Store predictions for each sensor
                            for idx, sensor in enumerate(selected_sensors):
                                if sensor not in predictions:
                                    predictions[sensor] = []
                                predictions[sensor].append(predicted_values[idx])

                            # Visualize the data
                            plot_data(data, predictions)
                except (ValueError, IndexError) as e:
                    print(f"Error processing message: {message} ({e})")
                    continue

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

#server_socket.close()
#ngrok.kill()