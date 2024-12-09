import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the saved model
model = load_model('model.h5')  # Replace with the path to your saved model
class_names = ['Beverages', 'Snacks']  # Replace with your actual class labels

# Connect to the IP Webcam feed
url = 'http://192.168.0.102:8080/video'  # Replace with your IP Webcam URL
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the original frame
    cv2.imshow("Live Feed", frame)

    # Preprocess the frame for model prediction
    img = cv2.resize(frame, (224, 224))  # Resize to the model's input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values

    # Make predictions
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Overlay predictions on the video feed
    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Classification", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
