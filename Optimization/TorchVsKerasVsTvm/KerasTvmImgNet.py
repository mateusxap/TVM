import tvm
from tvm import relay
from tvm.contrib import graph_executor
import os
import time
from keras.utils import to_categorical
import numpy as np
import keras
import tensorflow as tf
from keras.applications.resnet import ResNet50
from tvm import auto_scheduler

from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

import os
#n01632777 акселотли
folder_path = "imagenet_validation/n01440764"
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(image_files)

all_images = []
countImg = 50
for img_name in image_files:
    img_path = f'imagenet_validation/n01440764/{img_name}'
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    #x = np.expand_dims(x,axis=0) #(num_samples, 224, 224, 3) нужно если по одной
    x = preprocess_input(x)
    all_images.append(x)

x_data = np.array(all_images)
print(x_data.shape)


# model = ResNet50(
#     weights="imagenet",
# )
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
# #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)
# model.save("resnet50_imgnet.h5")
#model = keras.models.load_model("/kaggle/input/my-files/resnet50_cifar10ep30.h5")

model = keras.models.load_model("resnet50_imgnet.h5")
model.summary()
#print(x_train[:50].shape)
print("Prediction: ", np.argmax(model.predict(x_data), axis=1))
#print("Labels:     ", np.argmax(y_train[:50], axis=1))
#Замер времени

reshaped_data = [data.reshape(1, 224, 224, 3) for data in x_data]
predictions = []
start_time = time.time()
for data in reshaped_data:
    # делайте что-то с каждым подтензором
    #print(chunk_tensor.size())
    out = model.predict(data)
    predictions.append(out)
end_time = time.time()
inference_time_keras = end_time - start_time
print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
print ("FPS: ", countImg/inference_time_keras)

predictions = np.array(predictions)
predictions = np.array(predictions).reshape((countImg, 1000))

print("Prediction: ", np.argmax(predictions, axis=1))

start_time = time.time()
predictions = model.predict(x_data, batch_size=1) #поправил batch_size
end_time = time.time()
inference_time_keras = end_time - start_time
print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
print ("FPS: ", countImg/inference_time_keras)



input_shape = [1, 224, 224, 3] # [batch, height, width, channels]
#shape_dict = {"input_input": input_shape}
shape_dict = {"input_1": input_shape}
from tvm import relay
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC") #подгрузка модели
#print(mod)


#target = tvm.target.Target("llvm -mcpu=core-avx2")
target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

#with tvm.transform.PassContext(opt_level=3): #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

print(x_data.shape)
reshaped_data = [data.reshape(1, 224, 224, 3) for data in x_data]
results = []

#Проводим инференс над измененными данными
start_time_tvm = time.time()
for data in reshaped_data:
    tvm_model(data).numpy()[0]
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)

# def evaluate_performance(lib, data_shape, dtype="float32"):
#     dev = tvm.cpu()
#     data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
#     module = graph_executor.GraphModule(lib["default"](dev))
#     module.set_input(model.input.name, data_tvm)
#
#     print("Evaluate inference time cost...")
#     print(module.benchmark(dev, number=10, repeat=3))
#
# print("Без оптимизации")
# target = tvm.target.Target("llvm")
# dev = tvm.cpu(0)
# #with tvm.transform.PassContext(opt_level=3):
# tvm_model = relay.build(mod, target=target, params=params)
# 
# evaluate_performance(tvm_model, input_shape)
#
# print("С оптимизацией")
# target = tvm.target.Target("llvm -mcpu=core-avx2")
# with tvm.transform.PassContext(opt_level=3):
#     tvm_model = relay.build(mod, target=target, params=params)
#
# evaluate_performance(tvm_model, input_shape)

