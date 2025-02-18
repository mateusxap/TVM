import os
import time
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from keras_vit.vit import ViT_B32
from keras.datasets import cifar10
import tvm
from tvm import relay
from tvm.contrib import graph_executor

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

model = ViT_B32(
    image_size = 32,
    activation = 'softmax',
    pretrained = False,
    include_top = True,
    pretrained_top = False,
    classes = 10
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.save("vit_cifar10_final.h5")
model = keras.models.load_model("vit_cifar10_final.h5")

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

input_shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3):
    tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

results = []
start_time_tvm = time.time()
for data in reshaped_data:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)
