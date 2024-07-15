import os
import time
from keras.utils import to_categorical
import numpy as np
import keras
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.datasets import cifar10
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import multiprocessing
from tvm import meta_schedule as ms

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

model = MobileNetV2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10,
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.save("mobilenetv2_cifar10_final.h5")
model = keras.models.load_model("mobilenetv2_cifar10_final.h5")

input_name = "input_1"
countImg = 250
reshaped_data = [data.reshape(1, 32, 32, 3) for data in x_train[:countImg]]

# Inference with Keras
predictions = []
start_time = time.time()
for data in reshaped_data:
    predictions.append(model.predict(data))
end_time = time.time()
inference_time_keras = end_time - start_time
print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
print("FPS: ", countImg / inference_time_keras)

# Convert Keras model to Relay
input_shape = [1, 32, 32, 3]
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")

# Compile and run with TVM
target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

#with tvm.transform.PassContext(opt_level=3):
tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

results = []
start_time_tvm = time.time()
for data in reshaped_data:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели no opt TVM: {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)


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
print("Время инференса модели c opt TVM: {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)


# Meta-scheduler
strategy_name = "evolutionary"
work_dir = "meta-scheduler-mobilenetv2-keras-cifar"
# def evaluate_performance(lib, data_shape, dtype="float32"):
#     dev = tvm.cpu()
#     data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
#     module = graph_executor.GraphModule(lib["default"](dev))
#     module.set_input('input_input', data_tvm)

#     print("Evaluate inference time cost...")
#     print(module.benchmark(dev, number=100, repeat=3))
    
# def extract_tasks(mod, target, params, strategy):
#     print("Extract tasks...")
#     extracted_tasks = ms.relay_integration.extract_tasks(
#         mod, target, params
#     )
#     assert(len(extracted_tasks) > 0)
    
#     tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
#         extracted_tasks, work_dir, strategy=strategy
#     )

#     for idx, task in enumerate(tasks):
#         print("Task: %d, desc: %s" % (idx, task.task_name))

#     return tasks, task_weights

# def run_tuning(tasks, task_weights, work_dir, n_trials):
#     if not os.path.exists(work_dir):
#         os.mkdir(work_dir)
#     print("Begin tuning...")    
#     evaluator_config = ms.runner.config.EvaluatorConfig(number=1, repeat=10, enable_cpu_cache_flush=True);
#     database = ms.tune.tune_tasks(
#         tasks=tasks,
#         task_weights=task_weights,
#         work_dir=work_dir,
#         max_trials_global=n_trials,
#         num_trials_per_iter=64,
#         max_trials_per_task=256,
#         builder=ms.builder.LocalBuilder(),
#         runner=ms.runner.LocalRunner(evaluator_config=evaluator_config),
#     )


# tasks, task_weights = extract_tasks(mod, target, params, strategy_name)
# n_trials = len(tasks) * 64 *2
# run_tuning(tasks, task_weights, work_dir, n_trials)

database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                    f"{work_dir}/database_tuning_record.json",
                                    allow_missing=False)

with tvm.transform.PassContext(opt_level=3):
    lib = ms.relay_integration.compile_relay(database, mod, target, params)
    print("Optimized mode:")

ms_mod = graph_executor.GraphModule(lib["default"](dev))

results2 = []
start_time_tvm = time.time()
for data in reshaped_data:
    ms_mod.set_input(input_name, data)
    ms_mod.run()
    results2.append(ms_mod.get_output(0))
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM c тюннингом: {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)