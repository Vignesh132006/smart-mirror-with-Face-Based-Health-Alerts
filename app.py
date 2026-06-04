import cv2
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime
import os
import json
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global status tracking variables
current_user = "Unknown"
current_confidence = 0.0

# =======================
#  LOAD CONFIGURATION
# =======================
CONFIG_PATH = "config.json"
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    # Default fallback config
    return {
        "Unknown": {
            "greeting": "Welcome to Smart Mirror!",
            "alert": "Keep smiling and stay healthy today!"
        },
        "default": {
            "greeting": "Welcome!",
            "alert": "Stay healthy, positive, and active!"
        }
    }

def save_config(config_data):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

# =======================
#  LOAD TRAINED MODEL
# =======================
model_path = "model/face_model.h5"
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        # Get persons folders
        persons = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]
        print(f"Model loaded successfully. Registered classes: {persons}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        persons = []
else:
    print(f"Warning: Model not found at '{model_path}'. Please run train_model.py first.")
    model = None
    persons = []

# Load face cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =======================
#  WEATHER API SETTINGS
# =======================
API_KEY = "27c5d0050b12c89a764f5df909f5385a"   # Weather API key
LAT = 11.37  # Kannampalayam
LON = 76.97  # Kannampalayam

temp_buffer = deque(maxlen=5)
humidity_buffer = deque(maxlen=5)
weather_desc_buffer = deque(maxlen=5)
last_api_call = 0
API_CALL_INTERVAL = 30

def get_weather():
    global last_api_call
    current_time = datetime.now().timestamp()
    
    if current_time - last_api_call < API_CALL_INTERVAL:
        if temp_buffer and humidity_buffer and weather_desc_buffer:
            return round(np.mean(list(temp_buffer)), 1), round(np.mean(list(humidity_buffer)), 1), weather_desc_buffer[-1]
        else:
            return "N/A", "N/A", "Clear Sky"
            
    last_api_call = current_time
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    try:
        data = requests.get(url, timeout=5).json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        desc = data["weather"][0]["description"].title()
        
        temp_buffer.append(temp)
        humidity_buffer.append(humidity)
        weather_desc_buffer.append(desc)
        
        return round(np.mean(list(temp_buffer)), 1), round(np.mean(list(humidity_buffer)), 1), desc
    except Exception as e:
        print(f"Weather API error: {e}")
        if temp_buffer and humidity_buffer and weather_desc_buffer:
            return round(np.mean(list(temp_buffer)), 1), round(np.mean(list(humidity_buffer)), 1), weather_desc_buffer[-1]
        return "N/A", "N/A", "N/A"

# Helper for health alerts and greetings
def get_health_alert_and_greeting(name):
    config_data = load_config()
    user_key = name if name in config_data else "default"
    user_data = config_data.get(user_key, config_data.get("default"))
    return user_data.get("greeting", "Welcome!"), user_data.get("alert", "Stay Healthy!")

# =======================
#  WEBCAM VIDEO FEED GENERATOR
# =======================
def generate_frames():
    global current_user, current_confidence
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Horizontal flip for natural mirror experience
        frame = cv2.flip(frame, 1)
        
        # Preprocessing & Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        
        name = "Unknown"
        confidence = 0.0
        
        if len(faces) > 0 and model is not None:
            # Predict for the largest face detected
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            (x, y, w, h) = largest_face
            
            face_crop = frame[y:y+h, x:x+w]
            try:
                # Preprocess cropped color face image
                img = cv2.resize(face_crop, (100, 100))
                img = np.expand_dims(img / 255.0, axis=0)
                
                # Predict
                pred = model.predict(img, verbose=0)
                idx = np.argmax(pred)
                confidence = float(pred[0][idx])
                
                # Cutoff for recognized users
                if confidence > 0.65:
                    name = persons[idx]
                else:
                    name = "Unknown"
            except Exception as e:
                print(f"Prediction logic error: {e}")
            
            current_user = name
            current_confidence = confidence
            
            # Draw overlay bounding box on video
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.1%})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            current_user = "Unknown"
            current_confidence = 0.0
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    camera.release()

# =======================
#  FLASK ROUTING
# =======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    global current_user, current_confidence
    greeting, alert = get_health_alert_and_greeting(current_user)
    temp, hum, desc = get_weather()
    
    return jsonify({
        "user": current_user,
        "confidence": current_confidence,
        "greeting": greeting,
        "alert": alert,
        "time": datetime.now().strftime("%I:%M:%S %p"),
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "weather": {
            "temp": temp,
            "humidity": hum,
            "description": desc
        }
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        new_config = request.json
        if save_config(new_config):
            return jsonify({"status": "success", "message": "Configuration updated successfully."})
        else:
            return jsonify({"status": "error", "message": "Failed to update configuration."}), 500
    else:
        return jsonify(load_config())

if __name__ == '__main__':
    # Start webserver on port 5000
    print("Starting Flask Web Smart Mirror on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
