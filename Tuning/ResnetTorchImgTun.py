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


# Загрузка модели с весами, обученными на ImageNet
model_name = "resnet50"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# # Вывод списка слоев
# summary(model, (3, 224, 224))

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()


input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(mod)


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
work_dir = "meta-scheduler-torch-resnet-img"

target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores 6")
dev = tvm.cpu(0)



tasks, task_weights = extract_tasks(mod, target, params, strategy_name)
n_trials = len(tasks) * 64 *2
run_tuning(tasks, task_weights, work_dir, n_trials)
