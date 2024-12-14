import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# this code taken from keras
# preprocessing for train data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,  # Random rotations
    brightness_range=[0.8, 1.2],  # Adjust brightness
    horizontal_flip=True,
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    fill_mode='nearest'  # Fill empty pixels after shifts/rotations
)

training_set = train_datagen.flow_from_directory(
        'C:/ML_CEP/classification_model_3/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')  #here catagorical because more than 2 classes, else we woild have used binary

# same preprocessing for test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'C:/ML_CEP/classification_model_3/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

######################################################## CNN ########################################################3
# model

cnn = tf.keras.models.Sequential()

#now according to 'https://www.upgrad.com/blog/basic-cnn-architecture/' we have 3 layers in cnn:
#  1) Convolutional
#  2) Pooling
#  3) Fully Connected 


# Convolutional layer & Pooling
cnn.add(tf.keras.layers.Conv2D(filters=64 , kernel_size=3 , activation='relu' , input_shape=[64,64,3]))   # here filter is basically 64*64 matrix

# there can be many activation func available on keras: layers activation
# input shape is defined in preprocessing and 3 represent RGB image
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# repeating so that after filtration of 2 times, the modelcan learn feature properly
#also no image shape given because that is alsready done
cnn.add(tf.keras.layers.Conv2D(filters=128 , kernel_size=3 , activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 , strides=2))

# also added to increase model's efficiency
cnn.add(tf.keras.layers.Dropout(0.5))

# # Flattening here
# cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.GlobalAveragePooling2D())

# Fully Connected 
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))   # units is number of hidden layers

# Output Layer
cnn.add(tf.keras.layers.Dense(units=3 , activation='softmax'))      # 3 categories hence unit=3
# for binary data unit=1 , because the ans will be 1 or else 0

# model is ready, compiling results
cnn.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])   #check differnt optimizer available at keras
# adam optimizer goof for binary classification

# now fit the model
cnn.fit(x = training_set , validation_data = test_set , epochs = 30)

#preprocess new image
from tensorflow.keras.preprocessing import image
import numpy as np
test_image = image.load_img('C:/ML_CEP/classification_model_3/val/1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    print('Beverages')
elif result[0][1]==1:
    print('Biscuits')
elif result[0][2]==1:
    print('Snacks')
print(result)
