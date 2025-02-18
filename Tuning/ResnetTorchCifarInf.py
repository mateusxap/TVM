import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import pickle
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


model_name = "resnet50"
model = getattr(torchvision.models, model_name)(pretrained=False)
model = model.eval()


input_shape = [1, 3, 32, 32]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

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
countImg = 500
for i in range(countImg):
    data, label = trainset[i]
    data = data.unsqueeze(0)
    trainsetInList.append(data)



# итерация по подмассивам
predictions = []
start_time = time.time()
for pic in trainsetInList:    
    # делайте что-то с каждым подтензором
    #print(chunk_tensor.size())
    out = model(pic)
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


#target = tvm.target.Target("llvm -mcpu=core-avx2")
target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

#with tvm.transform.PassContext(opt_level=3): #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

trainsetInList_np = [pic.numpy() for pic in trainsetInList]
results = []

#Проводим инференс над измененными данными
start_time_tvm = time.time()
for data in trainsetInList_np:
    results.append(tvm_model(data).numpy()[0])
end_time_tvm = time.time()
inference_time_tvm = end_time_tvm - start_time_tvm
print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
print ("FPS: ", countImg/inference_time_tvm)

target = tvm.target.Target("llvm -mcpu=core-avx2")



model_name = "resnet50_cifar10_final"
strategy_name = "evolutionary"
work_dir = "meta-scheduler-torch-resnet-cifar"

from tvm import relay
dev = tvm.cpu(0)
target = tvm.target.Target("llvm -mcpu=core-avx2")

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
