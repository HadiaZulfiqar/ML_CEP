from data_loader import train_dataset, val_dataset, test_dataset
from cnn_model import model  # Explicitly import the model object

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")
model.save('model.h5')  # Save the trained model as 'model.h5'


