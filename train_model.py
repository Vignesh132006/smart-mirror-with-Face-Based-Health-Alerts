import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = []
labels = []
persons = os.listdir("dataset")

for i, person in enumerate(persons):
    folder = "dataset/" + person
    for img in os.listdir(folder):
        img_path = folder + "/" + img
        image = cv2.imread(img_path)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        labels.append(i)

data = np.array(data) / 255.0
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(persons), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save("model/face_model.h5")
print("Training completed and model saved!")
