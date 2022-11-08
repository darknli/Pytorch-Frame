"""
该脚本主要用于测试计算资源对模型推理的速度
"""
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.vgg import vgg16
from .progress_bar import ProgressBar
import torch
import time
import pandas as pd


BENCHMARK = {
    (16, 3, 224, 224): {
        "mobilenet_v2": {
            "RTX-3090": 6,
            "V100": 12,
            "A100": 8,
        },
        "resnet50": {
            "RTX-3090": 7,
            "V100": 9,
            "A100": 8
        },
        "vgg16": {
            "RTX-3090": 9,
            "V100": 7,
            "A100": 5
        },
    },
    (64, 3, 224, 224): {
        "mobilenet_v2": {
            "RTX-3090": 16,
            "V100": 25,
            "A100": 24,
        },
        "resnet50": {
            "RTX-3090": 19,
            "V100": 33,
            "A100": 24,
        },
        "vgg16": {
            "RTX-3090": 29,
            "V100": 28,
            "A100": 22,
        },
    }
}


def gpu_cnn_speed(model_type: str = None, input_size: tuple = (16, 3, 224, 224), gpu_id: int = None):
    """
    测试gpu在cnn上的性能

    Parameters
    ----------
    model_type : str, default resnet50. 模型类型
        * vgg16
        * resnet18
        * resnet50
        * resnet101
        * resnet152
        * mobilenet_v2
    input_size : tuple, default (16, 3, 224, 224). 模型输入尺寸
    gpu_id : int, default None. 显卡号，如果是None，且显卡可用，则默认为0
    """
    if model_type is None:
        model_type = "resnet50"
    assert model_type in ("vgg16", "resnet18", "resnet50", "resnet101", "resnet152", "mobilenet_v2")

    if gpu_id is None:
        gpu_id = 0
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model = eval(model_type)().to(device)
    iters = 300

    begin = time.time()
    pbar = ProgressBar(total=iters, desc=model_type)
    for i in range(iters):
        x = torch.randn(input_size).to(device)
        _ = model(x)
        pbar.update(1)
    print(f"全程耗时: {int(time.time() - begin + 0.5)}s")
    print("参考各显卡benchmark")
    for bis, v in BENCHMARK.items():
        print("*" * 50)
        print("benchmark input size:", bis)
        print(pd.DataFrame(v))
