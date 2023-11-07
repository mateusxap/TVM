from resnet50 import ResNet50

import tvm
from tvm import relay
from tvm.contrib import graph_executor
import os

from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt # plot the first image in the dataset
import keras
#from tensorflow.keras.applications.resnet import ResNet


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #(num_samples, 32, 32, 3)

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10,
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
model.save("restnet50_cifar10.h5")
model = keras.models.load_model("restnet50_cifar10.h5")

input_shape = [1, 32, 32, 3] # [batch, height, width, channels]
shape_dict = {"input_input": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
print(mod)








# model = keras.models.load_model("my_model.h5")
#
# # predict first 4 images in the test set
# print("Prediction: ", np.argmax(model.predict(X_test[:10]), axis=1))
# print("Labels:     ", np.argmax(y_test[:10], axis=1))
#
# input_shape = [1, 28, 28, 1] # [batch, height, width, channels]
# shape_dict = {"input_input": input_shape}
# mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
#
# print(mod)
#
# target = tvm.target.Target("llvm  -mcpu=core-avx2")
# dev = tvm.cpu(0)
#
# with tvm.transform.PassContext(opt_level=3): #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
#     tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
#
# print("||||||||||||||||||||||||")
# print(tvm_model)
#
# out = []
# for data in X_test[:10]:
#     # HWC -> NHWC
#     out.append(tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]) #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
#
# print("Prediction: ", np.argmax(out, axis=1))
# print("Labels:     ", np.argmax(y_test[:10], axis=1))

