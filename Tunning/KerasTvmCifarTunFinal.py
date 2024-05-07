import os
# import pickle
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
import multiprocessing
from tvm import meta_schedule as ms

# with open('cifar10.pkl', 'rb') as file:
#     x_train, y_train, x_test, y_test = pickle.load(file)



(x_train, y_train), (x_test, y_test) = cifar10.load_data() #(num_samples, 32, 32, 3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
# print(x_train.shape)
# print(x_test.shape)
# model = ResNet50(
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     input_shape=(32, 32, 3),
#     pooling=None,
#     classes=10,
# )

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# # #model.summary()
# # #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)
# model.save("resnet50_cifar10_final.h5")
model = keras.models.load_model("resnet50_cifar10_final.h5")


countImg = 250
reshaped_data = [data.reshape(1, 32, 32, 3) for data in x_train[:countImg]]


input_shape = [1, 32, 32, 3] # [batch, height, width, channels]
#shape_dict = {"input_input": input_shape}
shape_dict = {"input_1": input_shape}
input_name = "input_1"
model_name = "resnet50_cifar10_final"
from tvm import relay
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC") #подгрузка модели
#print(mod)

target = tvm.target.Target("llvm -mcpu=core-avx2")


input_shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
model, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")

strategy_name = "evolutionary"
work_dir = "meta-scheduler"

target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 6")
dev = tvm.cpu(0)

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
    evaluator_config = ms.runner.config.EvaluatorConfig(repeat=1, min_repeat_ms=100)
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
    
tasks, task_weights = extract_tasks(model, target, params, strategy_name)

# %%time
n_trials = len(tasks) * 64 *2
run_tuning(tasks, task_weights, work_dir, n_trials)
# database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
#                                     f"{work_dir}/database_tuning_record.json",
#                                     allow_missing=False)

# with tvm.transform.PassContext(opt_level=4):
#     lib = ms.relay_integration.compile_relay(database, mod, target, params)
#     print("Optimized mode:")
#     collect_per_layer_stat(lib, dev)

# tasks, task_weights = extract_tasks(lib, target, params, strategy_name)

# n_trials = len(tasks) * 64 #*2

# run_tuning(tasks, task_weights, work_dir, 513)

# evaluate(lib, input_shape, work_dir, target)

# print("Default mode:")
# collect_per_layer_stat(tvm_lib, dev)