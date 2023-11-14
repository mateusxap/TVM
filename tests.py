import keras
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Преобразование меток в one-hot кодирование
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = keras.models.load_model("resnet50_cifar10.h5")
# print("Prediction: ", np.argmax(model.predict(x_train[:50]), axis=1))
# print("Labels:     ", np.argmax(y_train[:50], axis=1))
# print(np.argmax(model.predict(x_test[:50]) == np.argmax(y_test[:50])))
model = keras.models.load_model("resnet50_cifar10.h5")
print("Prediction: ", np.argmax(model.predict(x_train[:50]), axis=1))
print("Labels:     ", np.argmax(y_train[:50], axis=1))