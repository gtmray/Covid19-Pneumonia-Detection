# Predicting our own images

import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

model_dir = '/content/vgg16_model.h5'
image_dir = '/content/2.png'

model_loaded = load_model(model_dir)
img_width, img_height = 224, 224
img = image.load_img(image_dir, target_size = (img_width, img_height))
img = image.img_to_array(img)
plt.imshow(img/255.)
img = np.expand_dims(img, axis = 0)

result = np.argmax(model_loaded.predict(img), axis=-1)[0]
if(result==1):
  print("COVID19 POSITIVE! TAKE CARE")
else:
  print("COVID19 NEGATIVE! STAY SAFE")