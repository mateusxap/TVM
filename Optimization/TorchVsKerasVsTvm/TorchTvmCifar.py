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

# Загрузка модели с весами, обученными на ImageNet
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
countImg = 1000
for i in range(countImg):
    data, label = trainset[i]
    data = data.unsqueeze(0)
    trainsetInList.append(data)



# итерация по подмассивам
predictions = []
start_time = time.time()
for pic in trainsetInList[:countImg]:    
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


target = tvm.target.Target("llvm -mcpu=core-avx2")
#target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level=3): #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
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


# def evaluate_performance(lib):
#     dev = tvm.cpu()
#     module = graph_executor.GraphModule(lib["default"](dev))

#     print("Evaluate inference time cost...")
#     print(module.benchmark(dev, number=10, repeat=3))



# target = tvm.target.Target("llvm")
# dev = tvm.cpu(0)
# #with tvm.transform.PassContext(opt_level=3):
# tvm_model = relay.build(mod, target=target, params=params)

# evaluate_performance(tvm_model)
# print("Без оптимизации")

# target = tvm.target.Target("llvm -mcpu=core-avx2")
# with tvm.transform.PassContext(opt_level=3):
#     tvm_model = relay.build(mod, target=target, params=params)

# evaluate_performance(tvm_model)
# print("С оптимизацией")