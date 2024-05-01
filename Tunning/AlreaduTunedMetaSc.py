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


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

model = keras.models.load_model("resnet50_cifar10_final.h5")


countImg = 400
reshaped_data = [data.reshape(1, 32, 32, 3) for data in x_train[:countImg]]

input_shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
input_name = "input_1"
model_name = "resnet50_cifar10_final"
strategy_name = "evolutionary"
work_dir = "meta-scheduler"

from tvm import relay
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC") #подгрузка модели
dev = tvm.cpu(0)
target = tvm.target.Target("llvm -mcpu=core-avx2")

database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                    f"{work_dir}/database_tuning_record.json",
                                    allow_missing=False)

with tvm.transform.PassContext(opt_level=3):
    lib = ms.relay_integration.compile_relay(database, mod, target, params)

# results = []
# start_time_tvm = time.time()
# for data in reshaped_data:
#     results.append(lib(data).numpy()[0])
# end_time_tvm = time.time()
# inference_time_tvm = end_time_tvm - start_time_tvm
# print("Время инференса модели TVM c тюннингом: {} секунд".format(inference_time_tvm))
# print ("FPS: ", countImg/inference_time_tvm)

# ????????
# results = []
# start_time_tvm = time.time()
# for data in reshaped_data:
#     results.append(ms_mod(data).numpy()[0])
# end_time_tvm = time.time()
# inference_time_tvm = end_time_tvm - start_time_tvm
# print("Время инференса модели TVM c тюннингом: {} секунд".format(inference_time_tvm))
# print ("FPS: ", countImg/inference_time_tvm)

database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                    f"{work_dir}/database_tuning_record.json",
                                    allow_missing=False)

with tvm.transform.PassContext(opt_level=3):
    lib = ms.relay_integration.compile_relay(database, mod, target, params)
    print("Optimized mode:")


ms_mod = graph_executor.GraphModule(lib["default"](dev))

with tvm.transform.PassContext(opt_level=3):
    tvm_lib = relay.build(mod, target=target, params=params)

tvm_mod = graph_executor.GraphModule(tvm_lib["default"](dev))

# def evaluate_performance(lib, data_shape, dtype="float32"):
#     dev = tvm.cpu()
#     data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
#     module = graph_executor.GraphModule(lib["default"](dev))
#     module.set_input('input_input', data_tvm)

#     print("Evaluate inference time cost...")
#     print(module.benchmark(dev, number=100, repeat=3))


from tvm.contrib.debugger import debug_executor

def collect_per_layer_stat(lib, device, json_graph=None):
    if json_graph is None:
        json_graph = lib.get_graph_json()
    debug_module = debug_executor.GraphModuleDebug(lib["debug_create"]("default", device), [device], json_graph, None)
    debug_module.run(number=10, repeat=3)


print("Default mode:")
collect_per_layer_stat(tvm_lib, dev)

print("1")
results1 = []
start_time_tvm = time.time()
for data in reshaped_data:
    tvm_mod.set_input(input_name, data)
    tvm_mod.run()
    results1.append(tvm_mod.get_output(0))
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)



tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

#print(x_data.shape)
#reshaped_data = [data.reshape(1, 224, 224, 3) for data in x_data]
results = []
print("2")
#Проводим инференс над измененными данными
start_time_tvm = time.time()
for data in reshaped_data:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)




print("Optimized mode:")
collect_per_layer_stat(lib, dev)



results2 = []
start_time_tvm = time.time()
for data in reshaped_data:
    ms_mod.set_input(input_name, data)
    ms_mod.run()
    results2.append(ms_mod.get_output(0))
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM c тюннингом: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)

