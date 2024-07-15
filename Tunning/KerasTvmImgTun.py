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

from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

import multiprocessing
from tvm import meta_schedule as ms


import os
#n01632777 акселотли
# folder_path = "imagenet_validation/n01440764"
# image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
# print(image_files)

# all_images = []
# countImg = 50
# for img_name in image_files:
#     img_path = f'imagenet_validation/n01440764/{img_name}'
#     img = image.load_img(img_path,target_size=(224,224))
#     x = image.img_to_array(img)
#     #x = np.expand_dims(x,axis=0) #(num_samples, 224, 224, 3) нужно если по одной
#     x = preprocess_input(x)
#     all_images.append(x)

# x_data = np.array(all_images)
# print(x_data.shape)

# model = ResNet50(
#     weights="imagenet",
# )

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
# #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)
# model.save("resnet50_imgnet.h5")
model = keras.models.load_model("resnet50_imgnet.h5")

#print(x_train[:50].shape)
# print("Prediction: ", np.argmax(model.predict(x_data), axis=1))
#print("Labels:     ", np.argmax(y_train[:50], axis=1))
#Замер времени

# reshaped_data = [data.reshape(1, 224, 224, 3) for data in x_data]
# predictions = []
# start_time = time.time()
# for data in reshaped_data:
#     # делайте что-то с каждым подтензором
#     #print(chunk_tensor.size())
#     predictions.append(model.predict(data))
# end_time = time.time()
# inference_time_keras = end_time - start_time
# print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
# print ("FPS: ", countImg/inference_time_keras)

# predictions = np.array(predictions)
# predictions = np.array(predictions).reshape((countImg, 1000))

# print("Prediction: ", np.argmax(predictions, axis=1))

# start_time = time.time()
# predictions = model.predict(x_data, batch_size=1) #поправил batch_size
# end_time = time.time()
# inference_time_keras = end_time - start_time
# print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
# print ("FPS: ", countImg/inference_time_keras)



input_shape = [1, 224, 224, 3] # [batch, height, width, channels]
#shape_dict = {"input_input": input_shape}
shape_dict = {"input_1": input_shape}
from tvm import relay
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC") #подгрузка модели
#print(mod)


strategy_name = "evolutionary"
work_dir = "meta-scheduler-keras-img"

target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 6")
dev = tvm.cpu(0)


countImg = 250


def evaluate_performance(lib, data_shape, dtype="float32"):
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input('input_input', data_tvm)

    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))
    
def extract_tasks(mod, target, params, strategy):
    print("Extract tasks...")
    extracted_tasks = ms.relay_integration.extract_tasks(
        mod, target, params
    )
    assert(len(extracted_tasks) > 0)
    
    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
        extracted_tasks, work_dir, strategy=strategy
    )

    for idx, task in enumerate(tasks):
        print("Task: %d, desc: %s" % (idx, task.task_name))

    return tasks, task_weights

def run_tuning(tasks, task_weights, work_dir, n_trials):
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    print("Begin tuning...")    
    evaluator_config = ms.runner.config.EvaluatorConfig(number=1, repeat=10, enable_cpu_cache_flush=True);
    database = ms.tune.tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
        work_dir=work_dir,
        max_trials_global=n_trials,
        num_trials_per_iter=64,
        max_trials_per_task=256,
        builder=ms.builder.LocalBuilder(),
        runner=ms.runner.LocalRunner(evaluator_config=evaluator_config),
    )


tasks, task_weights = extract_tasks(mod, target, params, strategy_name)
n_trials = len(tasks) * 64 *2
run_tuning(tasks, task_weights, work_dir, n_trials)
