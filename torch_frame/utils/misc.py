import logging
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict

import datetime
import numpy as np
import torch
from tabulate import tabulate

logger = logging.getLogger(__name__)

__all__ = [
    "set_random_seed",
    "collect_env",
    "symlink",
    "create_small_table",
    "get_workspace"
]


def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The variable of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
        - PyTorch compiling details: The output of ``torch.__config__.show()``.
        - TorchVision (optional): TorchVision version.
        - OpenCV (optional): OpenCV version.

    Returns:
        str: A string describing the running environment.
    """
    env_info = []
    env_info.append(("sys.platform", sys.platform))
    env_info.append(("Python", sys.version.replace("\n", "")))
    env_info.append(("Numpy", np.__version__))

    cuda_available = torch.cuda.is_available()
    env_info.append(("CUDA available", cuda_available))

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info.append(("GPU " + ",".join(device_ids), name))

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision

        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass

    try:
        import cv2

        env_info.append(("OpenCV", cv2.__version__))
    except ModuleNotFoundError:
        pass

    torch_config = torch.__config__.show()
    env_str = tabulate(env_info) + "\n" + torch_config
    return env_str


def set_random_seed(seed: int, rank: int = 0) -> None:
    """Set random seed.

    Args:
        seed (int): Nonnegative integer.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert seed >= 0, f"Got invalid seed value {seed}."
    seed += rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False  # 保证每次卷积的算子都是固定的，而非使用最高效的方法
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def create_small_table(small_dict: Dict[str, Any]) -> str:
    """Create a small table using the keys of ``small_dict`` as headers.
    This is only suitable for small dictionaries.

    Args:
        small_dict (dict): A result dictionary of only a few items.

    Returns:
        str: The table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def get_workspace(work_dir, create_new_dir):
    if create_new_dir not in (None, "time_s", "time_m", "time_h", "time_d", "count"):
        logger.warning("create_new_dir参数输入错误, 使用`time_s`为其赋值")
        create_new_dir = "time_s"
    if os.path.exists(work_dir):
        if create_new_dir == "time_s":
            now = datetime.datetime.now()
            now_format = now.strftime("%Y-%m-%d %H_%M_%S")
            work_dir = f"{work_dir}_{now_format}"
        elif create_new_dir == "time_m":
            now = datetime.datetime.now()
            now_format = now.strftime("%Y-%m-%d %H_%M")
            work_dir = f"{work_dir}_{now_format}"
        elif create_new_dir == "time_h":
            now = datetime.datetime.now()
            now_format = now.strftime("%Y-%m-%d %H")
            work_dir = f"{work_dir}_{now_format}"
        elif create_new_dir == "time_d":
            now = datetime.datetime.now()
            now_format = now.strftime("%Y-%m-%d")
            work_dir = f"{work_dir}_{now_format}"
        elif create_new_dir == "count":
            for i in range(10000):
                if not os.path.exists(f"{work_dir}_{i}"):
                    break
            work_dir = f"{work_dir}_{i}"
    return work_dir
