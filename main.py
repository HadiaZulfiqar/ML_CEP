# DATE: 17th december 2024


import os
import cv2
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
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

def load_hybrid_model(vectorizer_path='C:/ML_CEP/classification_model_3/vectorizer.joblib', classifier_path='C:/ML_CEP/classification_model_3/classifier.joblib', label_encoder_path='C:/ML_CEP/classification_model_3/label_encoder.joblib'):
    """
    Load the saved vectorizer, classifier, and label encoder.
    """
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


def predict_with_saved_model(image_path, vectorizer, classifier, label_encoder, reader, cnn_model):
    """
    Predict the category of an image using the saved vectorizer, classifier, and label encoder.
    """
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
    predicted_category = label_encoder.inverse_transform([prediction])[0]
    return predicted_category, text


vectorizer, classifier, label_encoder = load_hybrid_model(
    vectorizer_path='C:/ML_CEP/classification_model_3/vectorizer.joblib',
    classifier_path='C:/ML_CEP/classification_model_3/classifier.joblib',
    label_encoder_path='C:/ML_CEP/classification_model_3/label_encoder.joblib'
)


reader = easyocr.Reader(['en'])
# Load Pre-trained CNN
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

test_image = 'C:/ML_CEP/classification_model_3/val/14.jpg'  # Replace with your test image path
predicted_category, detected_text = predict_with_saved_model(
    image_path=test_image,
    vectorizer=vectorizer,
    classifier=classifier,
    label_encoder=label_encoder,
    reader=reader,
    cnn_model=cnn_model
)
print(f"\nPredicted Category: {predicted_category}")
print(f"Detected Text: {detected_text}")
