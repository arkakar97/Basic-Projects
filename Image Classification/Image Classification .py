#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it


# In[4]:


data_dir


# In[5]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[6]:


list(data_dir.glob('*/*.jpg'))[:5]


# In[7]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[8]:


roses = list(data_dir.glob('roses/*'))
roses[:5]


# In[9]:


PIL.Image.open(str(roses[1]))


# In[10]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# In[11]:


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[12]:


flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[13]:


flowers_images_dict['roses'][:5]


# In[14]:


str(flowers_images_dict['roses'][0])


# In[15]:


img = cv2.imread(str(flowers_images_dict['roses'][0]))


# In[16]:


cv2.resize(img,(180,180)).shape


# In[17]:


X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[18]:


X = np.array(X)
y = np.array(y)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[20]:


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# In[21]:


num_classes = 5

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30)


# In[22]:


predictions = model.predict(X_test_scaled)
predictions


# In[26]:


score = tf.nn.softmax(predictions[0])


# In[29]:


np.argmax(score)


# In[32]:


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(180, 
                                                              180,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# In[33]:


plt.axis('off')
plt.imshow(X[0])


# In[34]:


plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))


# In[35]:


num_classes = 5

model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30) 


# In[36]:


model.evaluate(X_test_scaled,y_test)


# In[ ]:




