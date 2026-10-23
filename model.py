# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:57:12 2026

@author: hp
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# encoding
from sklearn.preprocessing import LabelEncoder

# For DL Model
import keras
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

#dataset load
print("Loading Dataset.....")

with open("array/images.p","rb") as f:
    images=pickle.load(f)
    
with open("array/labels.p","rb") as f:
    labels = pickle.load(f)
    
print("image_shape",images.shape)
print("label_shape",labels.shape)

#encoding

le = LabelEncoder()
labels = le.fit_transform(labels)

print("\nClasses Found:")
print(set(labels))

p = len(set(labels))
print("\nTotal Number of Person",p)

#plot
plt.imshow(images[69],cmap='gray')
plt.title("sample image")
plt.show()

#ImageProcessing
def preprocess(img):
    img = cv2.equalizeHist(img)
    img = img.reshape(100,100,1)
    img = img/255.0
    return img

images = np.array(list(map(preprocess,images)))
print("\nPreprocessed Image shape:",images.shape)

#to categorical
labels = to_categorical(labels)
print("Output label shape:",labels.shape)

#model create
model=Sequential()

model.add(Conv2D(128,(5,5),activation='relu',input_shape=(100,100,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten layer
model.add(Flatten())

# Fully connected layer

model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))



model.add(Dense(p,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.01), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.fit
model.fit(images,labels,validation_split=0.1,epochs=21,batch_size=32)

#save model

model.save("Final_Model.h5")
print("\nModelSaved")
