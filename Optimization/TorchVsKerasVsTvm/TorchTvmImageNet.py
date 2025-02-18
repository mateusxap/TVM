import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torchsummary import summary
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


# Загрузка модели с весами, обученными на ImageNet
model_name = "resnet50"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# Вывод списка слоев
summary(model, (3, 224, 224))

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

predictions = [out.detach().numpy() for out in predictions]
predictions = np.array(predictions).reshape((countImg, 1000))

print("Prediction: ", np.argmax(predictions, axis=1))

input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(mod)


target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.cpu(0)

tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

print(x_data_np.shape)
reshaped_data = [data.reshape(1, 3, 224, 224) for data in x_data_np]
results = []

#Проводим инференс над измененными данными
start_time_tvm = time.time()
for data in reshaped_data:
    tvm_model(data).numpy()[0]
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)