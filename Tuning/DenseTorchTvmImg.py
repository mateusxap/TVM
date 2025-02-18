import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
#warnings.filterwarnings("ignore")
from tvm.contrib import graph_executor
import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata
# PyTorch imports
import torch
import torchvision
from torchvision import transforms
import multiprocessing
from tvm import meta_schedule as ms

# Загрузка модели DenseNet121 с весами, обученными на ImageNet
model_name = "densenet121"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

all_images = []
from PIL import Image
countImg = 50
folder_path = "imagenet_validation/n01440764"
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(image_files)
for img_name in image_files:
    img_path = f'imagenet_validation/n01440764/{img_name}'
    img = Image.open(img_path).resize((224, 224))
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    imp_np = img.numpy()
    all_images.append(imp_np)

x_data_np = np.array(all_images)
x_data_t = torch.from_numpy(x_data_np)
chunks = x_data_t.chunk(x_data_t.size(0), dim=0)

# итерация по подмассивам
predictions = []
start_time = time.time()
for chunk_tensor in chunks:
    out = model(chunk_tensor)
    predictions.append(out)
end_time = time.time()
inference_time_torch = end_time - start_time
print("Время инференса модели PyTorch: {} секунд".format(inference_time_torch))
print ("FPS: ", countImg/inference_time_torch)



# Конвертация модели PyTorch в Relay
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

target = tvm.target.Target("llvm")

dev = tvm.cpu(0)

tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

reshaped_data = [data.reshape(1, 3, 224, 224) for data in x_data_np]
results = []

start_time_tvm = time.time()
for data in reshaped_data:
    tvm_model(data).numpy()[0]
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели без опт TVM: {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)

target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3):
    tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

start_time_tvm = time.time()
for data in reshaped_data:
    tvm_model(data).numpy()[0]
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: с опт {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)

# Мета-планировщик
work_dir = "meta-scheduler-torch-densenet-img"

strategy_name = "evolutionary"

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