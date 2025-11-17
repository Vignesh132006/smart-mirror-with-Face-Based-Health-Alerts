import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model("model/face_model.h5")
persons = os.listdir("dataset")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    img = cv2.resize(frame, (100,100))
    img = np.expand_dims(img/255.0, axis=0)

    predictions = model.predict(img)
    index = np.argmax(predictions)
    name = persons[index]

    cv2.putText(frame, name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Smart Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
