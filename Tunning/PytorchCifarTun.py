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
countImg = 1000
for i in range(countImg):
    data, label = trainset[i]
    data = data.unsqueeze(0)
    trainsetInList.append(data)



# # итерация по подмассивам
# predictions = []
# start_time = time.time()
# for pic in trainsetInList[:countImg]:    
#     # делайте что-то с каждым подтензором
#     #print(chunk_tensor.size())
#     out = model(pic)
#     predictions.append(out)
# end_time = time.time()
# inference_time_torch = end_time - start_time
# print("Время инференса модели PyTorch: {} секунд".format(inference_time_torch))
# print ("FPS: ", countImg/inference_time_torch)

# predictions = [out.detach().numpy() for out in predictions]
# predictions = np.array(predictions).reshape((countImg, 1000))

# print("Prediction: ", np.argmax(predictions, axis=1))

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
    evaluator_config = ms.runner.config.EvaluatorConfig(number=1, repeat=10, enable_cpu_cache_flush=True);
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


input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

strategy_name = "evolutionary"
work_dir = "meta-scheduler-torch-resnet-cifar"

target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 6")
dev = tvm.cpu(0)



tasks, task_weights = extract_tasks(mod, target, params, strategy_name)
n_trials = len(tasks) * 64 *3#*2
run_tuning(tasks, task_weights, work_dir, n_trials)




# input_name = "input0"
# shape_list = [(input_name, input_shape)]
# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
# print(mod)


# target = tvm.target.Target("llvm -mcpu=core-avx2")
# #target = tvm.target.Target("llvm")
# dev = tvm.cpu(0)

# with tvm.transform.PassContext(opt_level=3): #проводим тесты над нейросетью в tvm //    tvm_model(data.reshape(1, 28, 28, 1)).numpy()[0]
#     tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

# trainsetInList_np = [pic.numpy() for pic in trainsetInList]
# results = []

# #Проводим инференс над измененными данными
# start_time_tvm = time.time()
# for data in trainsetInList_np:
#     results.append(tvm_model(data).numpy()[0])
# end_time_tvm = time.time()
# inference_time_tvm = end_time_tvm - start_time_tvm
# print("Время инференса модели TVM: {} секунд".format(inference_time_tvm))
# print ("FPS: ", countImg/inference_time_tvm)
