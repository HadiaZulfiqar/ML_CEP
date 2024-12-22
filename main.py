import os
import cv2
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib
import tensorflow as tf
import re


# Text Extraction from Images
def extract_text_from_image(image_path, reader):
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    extracted_text = ' '.join([text for (_, text, _) in results])
    return extracted_text


# Load Pre-trained Models
def load_hybrid_model(vectorizer_path='vectorizer.joblib', classifier_path='classifier.joblib', label_encoder_path='label_encoder.joblib'):
    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    label_encoder = joblib.load(label_encoder_path)
    return vectorizer, classifier, label_encoder


# Image Feature Extraction using Pre-trained CNN
def extract_image_features(image_path, cnn_model, img_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    features = cnn_model.predict(image)
    return features.flatten()


# Prediction Function
def predict_with_saved_model(image_path, vectorizer, classifier, label_encoder, reader, cnn_model):
    # Extract text from image
    text = extract_text_from_image(image_path, reader)
    clean_text = text.lower()
    clean_text = re.sub('[^a-z0-9\s]', ' ', clean_text)
    text_features = vectorizer.transform([clean_text])
    
    # Extract image features
    img_features = extract_image_features(image_path, cnn_model)
    
    # Combine features
    combined_features = np.hstack((text_features.toarray(), img_features.reshape(1, -1)))
    
    # Predict category
    prediction = classifier.predict(combined_features)[0]
    probabilities = classifier.predict_proba(combined_features)[0]
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    return predicted_category, text, probabilities


# Visualize Detected Text
def visualize_detected_text(image_path, reader):
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Display Prediction Confidence
def display_prediction_confidence(probabilities, label_encoder):
    plt.bar(label_encoder.classes_, probabilities)
    plt.title('Prediction Confidence')
    plt.xlabel('Categories')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.show()


# Main Code
vectorizer, classifier, label_encoder = load_hybrid_model(
    vectorizer_path='C:/ML_CEP/classification_model_3/vectorizer.joblib',
    classifier_path='C:/ML_CEP/classification_model_3/classifier.joblib',
    label_encoder_path='C:/ML_CEP/classification_model_3/label_encoder.joblib'
)

reader = easyocr.Reader(['en'])
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Test Image
test_image = 'C:/ML_CEP/classification_model_3/val/14.jpeg'  # Replace with your test image path
predicted_category, detected_text, probabilities = predict_with_saved_model(
    image_path=test_image,
    vectorizer=vectorizer,
    classifier=classifier,
    label_encoder=label_encoder,
    reader=reader,
    cnn_model=cnn_model
)

# Display Results
print(f"Detected Text: {detected_text}")
print(f"Predicted Category: {predicted_category}")
print("Prediction Confidence:")
for category, prob in zip(label_encoder.classes_, probabilities):
    print(f"  {category}: {prob:.2f}")

# Visualizations
visualize_detected_text(test_image, reader)
display_prediction_confidence(probabilities, label_encoder)

# Save Results to CSV
output_df = pd.DataFrame({
    'Image': [test_image],
    'Detected Text': [detected_text],
    'Predicted Category': [predicted_category],
    'Confidence': [max(probabilities)],
    'Probabilities': [probabilities.tolist()]
})
output_df.to_csv('predictions.csv', index=False)

# Optional: Display Confusion Matrix (requires a labeled test set)
# Replace `test_data` with your actual test dataset
# y_pred = []
# y_true = []
# for test_image_path, true_label in test_data:
#     predicted_category, _, _ = predict_with_saved_model(
#         image_path=test_image_path,
#         vectorizer=vectorizer,
#         classifier=classifier,
#         label_encoder=label_encoder,
#         reader=reader,
#         cnn_model=cnn_model
#     )
#     y_pred.append(predicted_category)
#     y_true.append(true_label)

# cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()

