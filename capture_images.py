import cv2
import os

name = input("Enter person's name: ")

path = 'dataset/' + name
os.makedirs(path, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = camera.read()
    if not ret:
        break

    face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{path}/{count}.jpg", face)
    count += 1

    cv2.imshow("Capturing Faces", frame)

    if count >= 200:  # collect 200 images
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Dataset created successfully!")
