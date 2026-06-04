import cv2
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime
import os
import json
from collections import deque

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

config = load_config()

# =======================
#  LOAD TRAINED MODEL
# =======================
model_path = "model/face_model.h5"
if not os.path.exists(model_path):
    print(f"Warning: Model not found at '{model_path}'. Please run train_model.py first.")
    model = None
    persons = []
else:
    model = tf.keras.models.load_model(model_path)
    # Filter dataset directories to get persons names
    persons = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]

# Load face cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# =======================
#  WEATHER API SETTINGS
# =======================
API_KEY = "27c5d0050b12c89a764f5df909f5385a"   # your real API key
LAT = 11.37  # Kannampalayam
LON = 76.97  # Kannampalayam

# Temperature smoothing buffer (last 5 readings)
temp_buffer = deque(maxlen=5)
humidity_buffer = deque(maxlen=5)
last_api_call = 0
API_CALL_INTERVAL = 30  # Call API every 30 seconds to reduce noise

def get_weather():
    global last_api_call
    
    current_time = datetime.now().timestamp()
    
    # Only call API every 30 seconds to reduce fluctuations
    if current_time - last_api_call < API_CALL_INTERVAL:
        if temp_buffer and humidity_buffer:
            # Return average of buffered readings for stability
            return round(np.mean(list(temp_buffer)), 1), round(np.mean(list(humidity_buffer)), 1)
        else:
            return "N/A", "N/A"
    
    last_api_call = current_time
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

    try:
        data = requests.get(url, timeout=5).json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        
        # Add to buffers for smoothing
        temp_buffer.append(temp)
        humidity_buffer.append(humidity)
        
        # Return average of all buffered readings
        return round(np.mean(list(temp_buffer)), 1), round(np.mean(list(humidity_buffer)), 1)

    except Exception as e:
        print(f"Weather API error: {e}")
        if temp_buffer and humidity_buffer:
            return round(np.mean(list(temp_buffer)), 1), round(np.mean(list(humidity_buffer)), 1)
        return "N/A", "N/A"


# =======================
#  HEALTH ALERT LOGIC
# =======================
def get_health_alert_and_greeting(name):
    # Reload config dynamically to catch any runtime changes
    current_config = load_config()
    user_key = name if name in current_config else "default"
    user_data = current_config.get(user_key, current_config.get("default"))
    return user_data.get("greeting", "Welcome!"), user_data.get("alert", "Stay Healthy!")


# =======================
#  CAMERA START
# =======================
camera = cv2.VideoCapture(0)

print("Starting Smart Mirror...")
print("Press 'q' in the window to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # We want a mirroring effect (flip horizontally)
    frame = cv2.flip(frame, 1)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    name = "Unknown"
    confidence = 0.0

    if len(faces) > 0 and model is not None:
        # Predict for the largest face detected (closest person)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = largest_face

        # Crop face from color frame
        face_crop = frame[y:y+h, x:x+w]
        try:
            # Preprocess cropped face
            img = cv2.resize(face_crop, (100, 100))
            img = np.expand_dims(img / 255.0, axis=0)

            # Predict name
            pred = model.predict(img, verbose=0)
            idx = np.argmax(pred)
            confidence = pred[0][idx]

            # Only accept high-confidence predictions (e.g. > 65%)
            if confidence > 0.65:
                name = persons[idx]
            else:
                name = "Unknown"
        except Exception as e:
            print(f"Prediction error: {e}")

        # Draw green bounding box around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.1%})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Get Greeting & Alert based on predicted name
    greeting, alert = get_health_alert_and_greeting(name)

    # Time
    now_time = datetime.now().strftime("%I:%M:%S %p")
    now_date = datetime.now().strftime("%A, %b %d, %Y")

    # Weather
    temp, hum = get_weather()

    # =======================
    #  DISPLAY ON SCREEN (Clean glassmorphic overlay simulator)
    # =======================
    # Create semi-transparent overlay at the top left for readable text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 270), (0, 0, 0), -1)
    # Blend with original frame for translucent glass effect
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Render texts (avoiding unsupported characters like ⚕ in cv2.putText)
    cv2.putText(frame, now_date, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, now_time, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Temp: {temp} C  |  Humidity: {hum}%", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"User: {name}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw greeting & health alert
    cv2.putText(frame, greeting, (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Alert: {alert}", (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Smart Mirror - Face Based Health Alerts", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
