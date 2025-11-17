import cv2
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime
import os
from collections import deque

# =======================
#  LOAD TRAINED MODEL
# =======================
model = tf.keras.models.load_model("model/face_model.h5")
persons = os.listdir("dataset")   # person1, person2 folders

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
def get_health_alert(name):
    if "person1" in name.lower():
        return "⚕ Stay Hydrated Today!"
    elif "person2" in name.lower():
        return "⚕ Take Deep Breaths & Relax!"
    else:
        return "⚕ Stay Healthy & Positive!"


# =======================
#  CAMERA START
# =======================
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Resize the frame for model
    img = cv2.resize(frame, (100, 100))
    img = np.expand_dims(img / 255.0, axis=0)

    # Predict
    pred = model.predict(img)
    idx = np.argmax(pred)
    name = persons[idx]

    # Time
    now_time = datetime.now().strftime("%H:%M:%S")

    # Weather
    temp, hum = get_weather()

    # Health Alert
    alert = get_health_alert(name)

    # =======================
    #  DISPLAY ON SCREEN
    # =======================

    cv2.putText(frame, f"User: {name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Time: {now_time}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Temp: {temp}°C", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Humidity: {hum}%", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, alert, (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Smart Mirror - Face Based Health Alerts", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
