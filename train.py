import pandas as pd
import numpy as np
import cv2
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt

# Load dataset

# Create dictionary for alphabets and related numbers
alphabets_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 6: 'G', 7: 'H',
                 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5',
                 32: '6', 33: '7', 34: '8', 35: '9'}

alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
             'B', 'C', 'D', 'E', 'G', 'H']
dataset_classes = []

for cls in alphabets:
    dataset_classes.append([cls])

# Load old dataset
d = open("data.pickle", "rb")
la = open("labels.pickle", "rb")
data = pickle.load(d)
labels = pickle.load(la)

label_list = []
for l in labels:
    label_list.append([l])

# One hot encoding format for output
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(dataset_classes)
labels_ohe = ohe.transform(label_list).toarray()

data = np.array(data)
labels = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_ohe, test_size=0.20, random_state=42)

X_train = X_train.reshape(2241, 28, 28, 1)
X_test = X_test.reshape(561, 28, 28, 1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=15, batch_size=64)

model.save('cnn_classifier.h5')
# Visualization
plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
