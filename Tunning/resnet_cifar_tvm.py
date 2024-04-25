#from resnet50 import ResNet50

import tvm
from tvm import relay
from tvm.contrib import graph_executor
import os

from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt # plot the first image in the dataset
import keras
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tvm import auto_scheduler
from keras.datasets import cifar10
print(len(tf.config.experimental.list_physical_devices('GPU')))

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #(num_samples, 32, 32, 3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = ResNet50(
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     input_shape=(32, 32, 3),
#     pooling=None,
#     classes=10,
# )

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)
# model.save("resnet50_cifar10.h5")
model = keras.models.load_model("resnet50_cifar10.h5")
model.summary()
print("Prediction: ", np.argmax(model.predict(x_train[:50]), axis=1))
print("Labels:     ", np.argmax(y_train[:50], axis=1))

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
input_shape = [1, 32, 32, 3] # [batch, height, width, channels]
#shape_dict = {"input_input": input_shape}
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
print(mod)


target = tvm.target.Target("llvm  -mcpu=core-avx2")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3): #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
    tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

out = []
for data in x_test[:30]:
    # HWC -> NHWC
    out.append(tvm_model(data.reshape(1, 32, 32, 3)).numpy()[0]) #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]

print("Prediction: ", np.argmax(out, axis=1))
print("Labels:     ", np.argmax(y_test[:30], axis=1))

def evaluate_performance(lib, data_shape, dtype="float32"):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))
##


model_name = "restnet50_cifar10"
input_name = "input1"
target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.cpu()

import multiprocessing
# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = multiprocessing.cpu_count()
print("Num threads: ", int(num_threads))
os.environ["TVM_NUM_THREADS"] = str(int(num_threads))


log_file = "%s.auto-scheduler.log" % model_name

# extract workloads from relay program
def extract_tasks(mod, target, params):
    print("Mod:")
    print(mod)
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)
    assert(len(tasks) > 0)


    for idx, task in enumerate(tasks):
        print("Task: %d, desc: %s" % (idx, task.desc))
    return tasks, task_weights

tasks, task_weights = extract_tasks(mod, target, params)
print (tasks)
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
#print (task_weights)

def run_tuning(tasks, task_weights, log_file, n_trials):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

trials = len(tasks) * 2 * 64
print (trials)
run_tuning(tasks, task_weights, log_file,  trials)

def evaluate(module, data_shape, log_file, target="llvm"):
    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    with auto_scheduler.ApplyHistoryBest(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)


evaluate(mod, input_shape, log_file, target)

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
#for data in X_test[:10]:
    # HWC -> NHWC
    #out.append(tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0])

# X1_test = np.reshape(X1_test, input_shape)
# # Преобразование данных списка X1_test в тип float32
# X1_test = np.asarray(X1_test, dtype=np.float32)

# Создание массива data_tvm с использованием данных из X1_test
#dtype="float32"

# for X1 in X1_test:
#     X1_resh = X1.reshape(1,28,28,1)
#     data_tvm =  tvm.nd.array(X1_resh)
#     dev = tvm.cpu()
#     #data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
#     module = graph_executor.GraphModule(lib["default"](dev))
#     module.set_input(input_name, data_tvm)
#
#
# print("Prediction: ", np.argmax(out, axis=1))
# print("Labels:     ", np.argmax(y_test[:10], axis=1))

print("||||||||||||||||||||||||||||||||послойная статистика||||||||||||||||||||||||||||||||||||||||||")
from tvm.contrib.debugger import debug_executor

def collect_per_layer_stat(lib, device, json_graph=None):
    if json_graph is None:
        json_graph = lib.get_graph_json()
    debug_module = debug_executor.GraphModuleDebug(lib["debug_create"]("default", device), [device], json_graph, None)
    debug_module.run(number=10, repeat=3)

print("auto_scheduler")

collect_per_layer_stat(lib,dev)

print("default")
with tvm.transform.PassContext(opt_level=3):
    print("Compile...")
    libDefault = relay.build(mod, target=target, params=params)

collect_per_layer_stat(libDefault,dev)

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")








# model = keras.models.load_model("my_model.h5")
#
# # predict first 4 images in the test set
# print("Prediction: ", np.argmax(model.predict(X_test[:10]), axis=1))
# print("Labels:     ", np.argmax(y_test[:10], axis=1))
#
# input_shape = [1, 28, 28, 1] # [batch, height, width, channels]
# shape_dict = {"input1": input_shape}

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

