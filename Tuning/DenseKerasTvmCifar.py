import os
import time
from keras.utils import to_categorical
import numpy as np
import keras
import tensorflow as tf
from keras.applications.densenet import DenseNet121
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

model = DenseNet121(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10,
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.save("densenet_cifar10_final.h5")
model = keras.models.load_model("densenet_cifar10_final.h5")

input_name = "input_1"
countImg = 250
reshaped_data = [data.reshape(1, 32, 32, 3) for data in x_train[:countImg]]

input_shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
#target = tvm.target.Target("llvm -mcpu=skylake-avx512")
target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3):
    tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

results = []
start_time_tvm = time.time()
for data in reshaped_data:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM с опт: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)

shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

results = []
start_time_tvm = time.time()
for data in reshaped_data:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM без опт: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)




input_shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
model, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")

strategy_name = "evolutionary"
work_dir = "meta-scheduler-dense-keras-cifar"

database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                    f"{work_dir}/database_tuning_record.json",
                                    allow_missing=False)

with tvm.transform.PassContext(opt_level=3):
    lib = ms.relay_integration.compile_relay(database, mod, target, params)
    print("Optimized mode:")


ms_mod = graph_executor.GraphModule(lib["default"](dev))

print("Optimized mode:")


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