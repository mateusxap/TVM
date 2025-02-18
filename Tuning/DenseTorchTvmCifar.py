import os
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import multiprocessing
from tvm import meta_schedule as ms


# Преобразование для изображений
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Загрузка обучающего набора данных
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Загрузка тестового набора данных
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainsetInList = []
countImg = 250
input_name = "input0"
for i in range(countImg):
    data, label = trainset[i]
    data = data.unsqueeze(0)
    trainsetInList.append(data)

trainsetInList_np = [pic.numpy() for pic in trainsetInList]
results = []

# Создание модели DenseNet121
model = torchvision.models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, 10)

# Сохранение модели PyTorch
torch.save(model.state_dict(), "densenet_cifar10_final.pth")

# Загрузка модели PyTorch
model = torchvision.models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load("densenet_cifar10_final.pth"))
model.eval()


start_time = time.time()
predictions = []
for data in trainsetInList:
    predictions.append(model(data))
end_time = time.time()
inference_time_pytorch = end_time - start_time
print("Время инференса модели PyTorch: {} секунд".format(inference_time_pytorch))
print("FPS: ", countImg / inference_time_pytorch)

# Конвертация модели PyTorch в Relay
input_shape = [1, 3, 32, 32]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
mod, params = relay.frontend.from_pytorch(scripted_model, [(input_name, input_shape)])

target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3):
    tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

results = []
start_time_tvm = time.time()
for data in trainsetInList_np:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)

# Мета-планировщик
input_shape = [1, 3, 32, 32]
shape_dict = {input_name: input_shape}
target = tvm.target.Target("llvm -mcpu=core-avx2")
strategy_name = "evolutionary"
work_dir = "meta-scheduler-densenet-pytorch-cifar"


database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                    f"{work_dir}/database_tuning_record.json",
                                    allow_missing=False)

with tvm.transform.PassContext(opt_level=3):
    lib = ms.relay_integration.compile_relay(database, mod, target, params)

ms_mod = graph_executor.GraphModule(lib["default"](dev))

print("Optimized mode:")

results2 = []
start_time_tvm = time.time()
for data in trainsetInList_np:
    ms_mod.set_input(input_name, data)
    ms_mod.run()
    results2.append(ms_mod.get_output(0))
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM c тюннингом: {} секунд".format(inference_time_tvm))
print("FPS: ", countImg / inference_time_tvm)