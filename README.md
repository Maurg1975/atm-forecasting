# Atmospheric Parameters Forecasting with Neural Networks

## Dedication

This project is dedicated to the loving memory of Michela Dattolo, Camil Demetrescu, Donato Griesi and Francesco Vietri.

## Overview

This project is designed for real-time collection, transmission, and forecasting of atmospheric parameters (temperature, humidity, pressure, gas concentration, wind speed, wind direction and rain intensity) using a combination of Arduino, Python, and Long Short-Term Memory (LSTM) neural networks.
The project is composed of three main components:

1. **Arduino Program**: Collects environmental data from sensors.
2. **Data Transmission Script**: Sends the collected data to a remote server via ngrok.
3. **Data Reception and Forecasting Script**: Receives the data, trains an LSTM model, and forecasts future values.

More information:
https://www.youtube.com/watch?v=-h2XQh71tjQ

## Project Structure

- **`read_env_values.ino`**: The Arduino program that reads temperature, humidity, pressure and other data from the sensors.
- **`send_env_values.py`**: A Python script that runs on a local computer, reading data from the Arduino and sending it to a remote server through ngrok.
- **`recv_env_values.py`**: A Python script that runs on a server or a cloud platform (e.g., Google Colab), receiving data, training an LSTM model, and forecasting future atmospheric values.

## Requirements

### Hardware

- **Arduino Board**: Compatible models include Arduino Uno, Nano, and Mega.
- **Sensors**:
  - **Temperature and Humidity**: DHT11 or DHT22
  - **Pressure**: BMP180 or BMP280
  - **Light intensity**: BH1750
  - **Gas concentration**: MQ-2 and MQ-135
  - **Wind speed**: Anemometer
  - **Wind direction**: Wind Direction Sensor
  - **Rain intensity**: Rain intensity Sensor
- **Computer**: To run the data transmission and reception scripts.

### Software

- **Arduino IDE**: To upload the `read_env_values.ino` sketch to your Arduino board.
- **Python 3.x**: Required to run the Python scripts.
- **Python Packages**:
  - `pyngrok`
  - `socket`
  - `threading`
  - `numpy`
  - `pandas`
  - `sklearn`
  - `tensorflow`
  - `matplotlib`
  - `IPython`

### Additional Requirements

- **ngrok**: Required to expose the local server to the internet.

## Installation and Setup

### 1. Arduino Setup

1. **Connect the Sensors**: Wire your sensors to the Arduino (see `schematics` folder for details).
2. **Upload the Sketch**:
   - Open the `read_env_values.ino` file in the Arduino IDE.
   - Select the appropriate board and port from the Tools menu.
   - Upload the sketch to the Arduino.

### 2. Python Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Maurg1975/atm-forecasting.git
   cd atm-forecasting
   ```
2. **Install Python Dependencies**:
   - if you run `recv_env_values.py` on the Server:
     ```bash
     pip install -r requirements.txt
     ```
   - if you run `recv_env_values.py` on Google Colab:
     ```bash
     pip install -r requirements_colab.txt
     ```
3. **Configure ngrok**:
   - Sign up at ngrok and obtain an authentication token.
   - Replace the ngrok.set_auth_token value in recv_env_values.py with your token.

### 3. Running the Scripts

a. **Run `recv_env_values.py` on the Server or Colab**
   1. **Expose the Server**:
      - if you run `recv_env_values.py` on Google Colab:
        * Uncomment line `!pip install pyngrok` in `recv_env_values.py`
      - if you run `recv_env_values.py` on Server:
	    * Start ngrok to expose the server to the internet.
	    * Copy the public URL provided by ngrok.
   2. **Run the Script**:
      - if you run `recv_env_values.py` on the Server:
        ```bash
        python ./src/recv_env_values.py
        ```
      - The script will start listening for incoming data, update the LSTM model, and make predictions.

b. **Run `send_env_values.py` on the Local Computer**
   1. **Connect Arduino**: Ensure your Arduino is connected to the computer via USB.
   2. **Start the Script**:
      ```bash
      python ./src/send_env_values.py
      ```

## Usage

   1. **Real-Time Data Collection**: The Arduino reads environmental data and sends it to the server.
   2. **Data Transmission**: The send_env_values.py script transmits the data from the Arduino to the server.
   3. **Data Reception and Forecasting**: The recv_env_values.py script receives the data, trains the LSTM model, and makes predictions.
   4. **Visualization**: The script will plot the real-time data alongside the predicted values.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See the LICENSE file for more details.

## Contribution

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## Acknowledgements

**TensorFlow**: For the deep learning framework.
**ngrok**: For the tunneling service.
**Arduino**: For providing the hardware platform.

This project was developed with assistance from AI chatbot tools, specifically **ChatGPT** by OpenAI and **Claude** by Anthropic, which provided guidance and suggestions for coding and project implementation.

## Contact

For any questions or issues, feel free to open an issue on GitHub or contact me at <galeonet@tiscali.it>.


### Key Sections:

- **Overview**: Introduces the purpose and scope of the project.
- **Project Structure**: Describes the main components of the project.
- **Requirements**: Lists hardware and software prerequisites.
- **Installation and Setup**: Provides detailed instructions for setting up and running the project.
- **Usage**: Explains how to use the project and what each script does.
- **License**: Specifies the project's license.
- **Contribution**: Encourages collaboration.
- **Acknowledgements**: Credits tools and platforms used.
- **Contact**: Provides contact information for further inquiries.
