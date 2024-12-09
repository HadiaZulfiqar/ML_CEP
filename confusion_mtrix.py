from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from data_loader import train_dataset, val_dataset, test_dataset
from cnn_model import model  # Explicitly import the model object

# Get predictions
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = np.argmax(model.predict(test_dataset), axis=1)

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_true, y_pred))
