import image_MobileNetv2
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
