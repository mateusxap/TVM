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
folder_path = "imagenet_validation/n01440764"
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(image_files)

all_images = []
countImg = 50
for img_name in image_files:
    img_path = f'imagenet_validation/n01440764/{img_name}'
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    all_images.append(x)

x_data = np.array(all_images)
print(x_data.shape)


model = keras.models.load_model("resnet50_imgnet.h5")
model.summary()

print("Prediction: ", np.argmax(model.predict(x_data), axis=1))


reshaped_data = [data.reshape(1, 224, 224, 3) for data in x_data]
predictions = []
start_time = time.time()
for data in reshaped_data:
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



input_shape = [1, 224, 224, 3]
shape_dict = {"input_1": input_shape}
from tvm import relay
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC") #подгрузка модели
#print(mod)


target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.cpu(0)

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