import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# List only directories under dataset/
persons = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]

# If there is only 1 person, generate a dummy 'Unknown' class so classification model works
if len(persons) == 1:
    unknown_dir = os.path.join("dataset", "Unknown")
    os.makedirs(unknown_dir, exist_ok=True)
    if len(os.listdir(unknown_dir)) == 0:
        print("Only one person found. Generating dummy 'Unknown' class to enable face classification...")
        for k in range(100):
            # Generate random noise image
            noise_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(unknown_dir, f"noise_{k}.jpg"), noise_img)
            # Generate plain black/dark image
            dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(unknown_dir, f"dark_{k}.jpg"), dark_img)
        print("Dummy 'Unknown' dataset generated successfully.")
    # Reload persons list
    persons = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]

print(f"Detected classes for training: {persons}")

data = []
labels = []

for i, person in enumerate(persons):
    folder = os.path.join("dataset", person)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        if not os.path.isfile(img_path):
            continue
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        # Resize to 100x100
        image = cv2.resize(image, (100, 100))
        data.append(image)
        labels.append(i)

if len(data) == 0:
    raise ValueError("No training images found! Please capture some images first.")

data = np.array(data) / 255.0
labels = to_categorical(labels, num_classes=len(persons))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

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

print(f"Training model with {len(data)} images across {len(persons)} classes...")
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)
model.save("model/face_model.h5")
print("Training completed and model saved to 'model/face_model.h5'!")
