import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'C:/ML_CEP/classification_model_3/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'C:/ML_CEP/classification_model_3/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# model

# Load Pretrained Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# Build the transfer learning model
cnn = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])
# Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
cnn.fit(x=training_set, validation_data=test_set, epochs=10)
# Fine-tuning (Optional)
base_model.trainable = True
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=10)
#preprocess new image
from tensorflow.keras.preprocessing import image
import numpy as np
test_image = image.load_img('C:/ML_CEP/classification_model_3/val/12.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices

if result[0][0]>0.7:
    print('Beverages')
elif result[0][1]>0.7:
    print('Biscuits')
elif result[0][2]>0.7:
    print('Books')
elif result[0][3]>0.7:
    print('Dairy Cheese & Cream')
elif result[0][4]>0.7:
    print('Laundry')
elif result[0][5]>0.7:
    print('Snacks')
print(result)










import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pytesseract
from PIL import Image

# Function to extract text from an image
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'C:/ML_CEP/classification_model_3/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'C:/ML_CEP/classification_model_3/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

cnn = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=10)

# Fine-tuning (Optional)
base_model.trainable = True
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=10)

# Prediction on a new image with OCR integration
test_image_path = 'C:/ML_CEP/classification_model_3/val/12.jpg'

# Preprocess the image for CNN
test_image = image.load_img(test_image_path, target_size=(64, 64))
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)

# Predict with the CNN
result = cnn.predict(test_image_array)
print("Image Prediction:", result)

# Extract text using OCR
extracted_text = extract_text_from_image(test_image_path)
print("Extracted Text:", extracted_text)

# Combine predictions from image and OCR
if result[0][0] > 0.7:
    category = 'Beverages'
elif result[0][1] > 0.7:
    category = 'Biscuits'
elif result[0][2] > 0.7:
    category = 'Books'
elif result[0][3] > 0.7:
    category = 'Dairy Cheese & Cream'
elif result[0][4] > 0.7:
    category = 'Laundry'
elif result[0][5] > 0.7:
    category = 'Snacks'
else:
    category = 'Unknown'

print(f"Predicted Category (Image): {category}")

# Decision based on OCR and Image
if 'biscuit' or 'cookies' in extracted_text.lower():
    final_category = 'Biscuits'
elif 'snack' or 'chips' in extracted_text.lower():
    final_category = 'Snacks'
elif 'surf' or 'detergent' in extracted_text.lower():
    final_category = 'Laundry'
else:
    final_category = category  # Default to image prediction if OCR isn't conclusive

print(f"Final Predicted Category: {final_category}")
