from data_loader import train_dataset, val_dataset, test_dataset
from cnn_model import model  # Explicitly import the model object

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")
