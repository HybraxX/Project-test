import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import serial
import time
import threading

# --- SERIAL CONFIGURATION ---
# Usually '/dev/ttyACM0' or '/dev/ttyUSB0'. 
# Run 'ls /dev/tty*' in terminal to double check.
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600

# Global variable to store latest sensor data
latest_sensor_data = {
    "Temperature": 0.0,
    "Humidity": 0.0,
    "Soil_Moisture": 0.0,
    "Water_Level": 0.0
}

# --- BACKGROUND THREAD TO READ ARDUINO ---
def read_serial_data():
    global latest_sensor_data
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flush()
        print(f"✅ Connected to Arduino on {SERIAL_PORT}")
        
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').rstrip()
                    # Expecting format: "Temp,Hum,Soil,Water" e.g. "25.5,60.0,45,80"
                    data = line.split(',')
                    if len(data) == 4:
                        latest_sensor_data["Temperature"] = float(data[0])
                        latest_sensor_data["Humidity"] = float(data[1])
                        latest_sensor_data["Soil_Moisture"] = float(data[2])
                        latest_sensor_data["Water_Level"] = float(data[3])
                except ValueError:
                    continue # Skip corrupted packets
            time.sleep(0.1)
    except Exception as e:
        print(f"⚠️ Serial Error: {e}. Running in cached mode.")

# Start the background thread
thread = threading.Thread(target=read_serial_data)
thread.daemon = True # Ensures thread dies when main app quits
thread.start()


# --- FASTAPI APP SETUP ---
app = FastAPI()

# Mount static folder for HTML/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
try:
    model = joblib.load('crop_yield_prediction_model.pkl')
except:
    model = None

class CropData(BaseModel):
    N: float
    P: float
    K: float
    Soil_Moisture: float
    Temperature: float
    Humidity: float
    pH: float
    FN: float
    FP: float
    FK: float

# --- ROUTES ---

@app.get("/")
def serve_index():
    # Assumes index.html is inside a folder named 'static'
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/api/sensors")
def get_sensors():
    """API Endpoint called by the frontend to get real-time values"""
    return latest_sensor_data

@app.post("/predict/")
def predict_yield(data: CropData):
    if not model:
        return {"error": "Model not loaded"}
    
    features = [
        data.N, data.P, data.K, data.Soil_Moisture, 
        data.Temperature, data.Humidity, data.pH, 
        data.FN, data.FP, data.FK
    ]
    
    prediction = model.predict([features])
    return {"predicted_yield": round(prediction[0], 2)}