from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import random # Imported just to simulate sensor data
import time

# --- REAL SENSOR LIBRARY IMPORTS (Uncomment when on Pi) ---
# import RPi.GPIO as GPIO
# import adafruit_dht
# import board
# ----------------------------------------------------------

app = Flask(__name__)

# Load the trained model
with open('crop_yield_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ==========================================
#  SENSOR READING HARDWARE SECTION
# ==========================================

# Setup GPIO pin numbering (Uncomment on Pi)
# GPIO.setmode(GPIO.BCM)

# Define GPIO Pins (Example pins - adjust to your wiring!)
SOIL_PIN = 21  # Example GPIO pin for digital soil sensor
WATER_LEVEL_PIN = 20 # Example pin
DHT_SENSOR = adafruit_dht.DHT11(board.D4) # Temperature/Humidity on GPIO 4

# Setup Input Pins (Uncomment on Pi)
# GPIO.setup(SOIL_PIN, GPIO.IN)
# GPIO.setup(WATER_LEVEL_PIN, GPIO.IN)


def get_real_time_sensor_data():
    """
    Reads data from actual sensors connected to Raspberry Pi.
    CURRENTLY MOCKED WITH RANDOM NUMBERS FOR TESTING OFF-PI.
    """
    data = {}

    # --- MOCK DATA (DELETE THIS BLOCK WHEN REAL SENSORS WORK) ---
    # We simulate data so the frontend has something to show right now.
    # Soil moisture: often analog but simpler sensors give 0 (wet) or 1 (dry) digital signal.
    # Let's assume a percentage simulation for now.
    data['soil_moisture'] = round(random.uniform(30.0, 85.0), 1)
    # Water level: similar simulation
    data['water_level'] = round(random.uniform(10.0, 95.0), 1)
    # Temp and Humidity
    data['temperature'] = round(random.uniform(24.0, 32.0), 1)
    data['humidity'] = round(random.uniform(50.0, 80.0), 1)
    # -----------------------------------------------------------


    # --- REAL SENSOR IMPLEMENTATION GUIDE (UNCOMMENT & ADAPT ON PI) ---
    
    # 1. Temperature & Humidity (DHT11/22)
    # try:
    #     # Print to console to debug on Pi
    #     # print("Reading DHT sensor...")
    #     temperature_c = DHT_SENSOR.temperature
    #     humidity_reading = DHT_SENSOR.humidity
    #     if temperature_c is not None and humidity_reading is not None:
    #          data['temperature'] = round(temperature_c, 1)
    #          data['humidity'] = round(humidity_reading, 1)
    #     else:
    #          # Handle failed read gracefully
    #          data['temperature'] = 25.0 # Default fallback
    #          data['humidity'] = 60.0 # Default fallback
    # except RuntimeError as error:
    #     print(f"DHT Error: {error.args[0]}")
    #     data['temperature'] = 25.0
    #     data['humidity'] = 60.0

    # 2. Soil Moisture & Water Level (Digital "Is Wet/Is Dry" type sensors)
    # Note: If using analog sensors, you need an ADC chip like MCP3008.
    # This example assumes simple digital GPIO input (HIGH/LOW).
    
    # soil_status = GPIO.input(SOIL_PIN)
    # If sensor returns 0 for wet, 1 for dry:
    # if soil_status == 0:
    #      data['soil_moisture'] = 80.0 # High value for wet
    # else:
    #      data['soil_moisture'] = 20.0 # Low value for dry

    # water_status = GPIO.input(WATER_LEVEL_PIN)
    # data['water_level'] = 90.0 if water_status == 1 else 10.0

    # ------------------------------------------------------------------

    return data


# ==========================================
#  FLASK ROUTES
# ==========================================

@app.route('/')
def home():
    # When loading the page initially, grab one set of data
    initial_sensor_data = get_real_time_sensor_data()
    return render_template('index.html', sensor_data=initial_sensor_data)


@app.route('/api/sensor_readings')
def api_sensor_readings():
    """
    This route is called by JavaScript every few seconds.
    It returns the latest sensor data as JSON.
    """
    sensor_data = get_real_time_sensor_data()
    return jsonify(sensor_data)


@app.route('/predict', methods=['POST'])
def predict():
    # Get manual inputs
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Get sensor inputs (these will be populated in hidden fields in HTML)
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        soil_moisture = float(request.form['soil_moisture'])
        water_level = float(request.form['water_level'])

        # Prepare features for the model
        final_features = [np.array([N, P, K, temperature, humidity, ph, rainfall, soil_moisture, water_level])]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        prediction_text = 'Predicted Crop Yield: {} tons/hectare'.format(output)

    except ValueError:
        prediction_text = "Error: Please ensure all fields contain valid numbers."
    except Exception as e:
        prediction_text = f"An error occurred: {str(e)}"

    # Re-fetch sensor data so the dashboard doesn't look empty after prediction
    current_sensor_data = get_real_time_sensor_data()

    return render_template('index.html', 
                           prediction_text=prediction_text, 
                           sensor_data=current_sensor_data)

if __name__ == "__main__":
    # host='0.0.0.0' makes it accessible on the local network
    app.run(host='0.0.0.0', port=5000, debug=True)