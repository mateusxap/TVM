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

from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

import os
#n01632777 акселотли
folder_path = "imagenet_validation/n01440764"
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(image_files)

input_name = "input_1"
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

model = DenseNet121(
    weights="imagenet",
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.save("densenet_img_final.h5")
model = keras.models.load_model("densenet_img_final.h5")

reshaped_data = [data.reshape(1, 224, 224, 3) for data in x_data]
predictions = []
start_time = time.time()
# for data in reshaped_data:
#     # делайте что-то с каждым подтензором
#     #print(chunk_tensor.size())
#     predictions.append(model.predict(data))
# end_time = time.time()
# inference_time_keras = end_time - start_time
# print("Время инференса модели Keras: {} секунд".format(inference_time_keras))
# print ("FPS: ", countImg/inference_time_keras)

input_shape = [1, 224, 224, 3]
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
#target = tvm.target.Target("llvm -mcpu=skylake-avx512")
target = tvm.target.Target("llvm -mcpu=core-avx2")
#target = tvm.target.Target("llvm")
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


target = tvm.target.Target("llvm -mcpu=core-avx2")
#target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

#with tvm.transform.PassContext(opt_level=3):
# tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

# results = []
# start_time_tvm = time.time()
# for data in reshaped_data:
#     results.append(tvm_model(data).numpy()[0])
# end_time_tvm = time.time()
# inference_time_tvm = end_time_tvm - start_time_tvm
# print("Время инференса модели TVM без опт: {} секунд".format(inference_time_tvm))
# print ("FPS: ", countImg/inference_time_tvm)


input_shape = [1, 224, 224, 3]
shape_dict = {"input_1": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")

strategy_name = "evolutionary"
work_dir = "meta-scheduler-dense-keras-img"

# target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 6")
# dev = tvm.cpu(0)



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
# n_trials = len(tasks) * 64 * 3 // 2 
# run_tuning(tasks, task_weights, work_dir, n_trials)


database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                    f"{work_dir}/database_tuning_record.json",
                                    allow_missing=False)

with tvm.transform.PassContext(opt_level=3):
    lib = ms.relay_integration.compile_relay(database, mod, target, params)



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

