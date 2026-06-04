import cv2
import os

# Load OpenCV's pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name = input("Enter person's name: ")

path = 'dataset/' + name
os.makedirs(path, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0
print("Starting camera. Look directly at the camera and press 'q' to stop.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert to grayscale for Haar cascade face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangle around face and save it
    for (x, y, w, h) in faces:
        # Crop the face from the BGR (color) frame
        face_crop = frame[y:y+h, x:x+w]
        
        # Save the cropped color face image
        cv2.imwrite(f"{path}/{count}.jpg", face_crop)
        count += 1
        
        # Draw bounding box on the displayed frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break  # Only process the first detected face per frame for dataset collection

    cv2.imshow("Capturing Faces", frame)

    if count >= 200:  # collect 200 images
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print(f"Dataset for {name} created successfully! Collected {count} images.")
