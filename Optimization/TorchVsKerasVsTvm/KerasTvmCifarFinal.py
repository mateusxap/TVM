import os
import time
from keras.utils import to_categorical
import numpy as np
#import matplotlib.pyplot as plt # plot the first image in the dataset
import keras
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.datasets import cifar10
import tvm
from tvm import relay
from tvm.contrib import graph_executor

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #(num_samples, 32, 32, 3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10,
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)
model.save("resnet50_cifar10_final.h5")
model = keras.models.load_model("resnet50_cifar10_final.h5")

countImg = 250
reshaped_data = [data.reshape(1, 32, 32, 3) for data in x_train[:countImg]]
predictions = []
start_time = time.time()
for data in reshaped_data:
    predictions.append(model.predict(data))
end_time = time.time()
inference_time_keras = end_time - start_time
print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
print ("FPS: ", countImg/inference_time_keras)


input_shape = [1, 32, 32, 3] # [batch, height, width, channels]
shape_dict = {"input_1": input_shape}
from tvm import relay
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC") #подгрузка модели

target = tvm.target.Target("llvm -mcpu=core-avx2")

dev = tvm.cpu(0)

tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
results = []

#Проводим инференс над измененными данными
start_time_tvm = time.time()
for data in reshaped_data:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)