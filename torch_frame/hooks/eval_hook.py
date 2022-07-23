from typing import Callable, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from .checkpoint_hook import CheckpointerHook
import numpy as np
from ..utils.dist_utils import *
from ..utils import ProgressBar
from .._get_logger import logger
import pickle


def collect_result_gpu(result_part):
    """收集多卡的结果放到0号卡上"""
    result_part = [result_part]
    rank, world_size = get_rank(), get_world_size()
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda")
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    dist.all_gather(part_recv_list, part_send)

    if rank > 0:
        return
    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_list.append(
            pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
        )
    final_res = part_list[0][0]
    for res in part_list[1:]:
        res = res[0]
        for k in final_res:
            final_res[k].extend(res[k])
    return final_res


class EvalHook(CheckpointerHook):
    """`CheckpointerHook` 的派生类, 周期性执行的评估器, 在每个epoch的最后阶段执行"""

    def __init__(self,
                 dataloader: DataLoader,
                 eval_func: Callable,
                 period: int = 1,
                 max_to_keep: Optional[int] = None,
                 save_metric: Optional[str] = None,
                 max_first: bool = True,
                 save_last: bool = True,
                 prefix: str = "eval"
                 ):
        """

        Parameters
        ----------
        dataloader : DataLoader.
            测试数据的dataloader
        eval_func : Callable.
            一个函数, 没有输入参数, 返回一个评估结果的Dict[list], k对应指标名称, v是包含每个样本得分的list
        period : int, default 1.
            执行eval_func函数的周期
        max_to_keep : int, 保存checkpoints的数量, 更早期的checkpoints会被删除
        save_metric : int, default None.
            保存模型的指标是哪个, 需要从trainer.metric_storage选择
        max_first : bool, default True.
            用于保存模型的指标是取最大还是最小作为最优模型
        save_last : bool, default True
            是否保存最近一次的epoch的模型, 如果是True, 每轮将更新模型到latest.pth中
        """
        self.prefix = prefix+"_"
        super(EvalHook, self).__init__(period, max_to_keep, self.prefix + save_metric, max_first, save_last)
        self._eval_func = eval_func
        self.dataloader = dataloader

    def _do_eval(self):
        tot_res = {}
        self.trainer.model.eval()
        with torch.no_grad():
            with tqdm(self.dataloader, desc="eval") as pbar:
                for batch in pbar:
                    res = self._eval_func(self.trainer.model, batch)
                    for k, v in res.items():
                        tot_res.setdefault(k, []).extend(v)
        self.trainer.model.train()
        if tot_res:
            rename_res = {self.prefix + k: np.mean(v) for k, v in tot_res.items()}
            self.log(self.trainer.epoch, **rename_res, smooth=False, window_size=1)

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
            self.save_model()


class DDPEvalHook(CheckpointerHook):
    """`EvalHook`类的DDP版本，支持多卡预测数据集，但需要注意每张卡的模型应该保持一致"""
    def __init__(self,
                 dataset: Dataset,
                 dataset_params: dict,
                 eval_func: Callable,
                 period: int = 1,
                 max_to_keep: Optional[int] = None,
                 save_metric: Optional[str] = None,
                 max_first: bool = True,
                 save_last: bool = True,
                 prefix: str = "eval"
                 ):
        """

        Parameters
        ----------
        dataset : Dataset.
            测试数据的dataset
        dataset_params: dict.
            dataset传入dataloader时的参数
        eval_func : Callable.
            一个函数, 没有输入参数, 返回一个评估结果的Dict[list], k对应指标名称, v是包含每个样本得分的list
        period : int, default 1.
            执行eval_func函数的周期
        max_to_keep : int, 保存checkpoints的数量, 更早期的checkpoints会被删除
        save_metric : int, default None.
            保存模型的指标是哪个, 需要从trainer.metric_storage选择
        max_first : bool, default True.
            用于保存模型的指标是取最大还是最小作为最优模型
        save_last : bool, default True
            是否保存最近一次的epoch的模型, 如果是True, 每轮将更新模型到latest.pth中
        """
        self.prefix = prefix+"_"
        super(DDPEvalHook, self).__init__(period, max_to_keep, self.prefix + save_metric, max_first, save_last)
        self._eval_func = eval_func

        self.use_dist = torch.cuda.device_count() > 1
        if self.use_dist:
            num_tasks = get_world_size()
            sampler = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            logger.warning("检测到该环境不支持多卡，退回普通eval_hook")
            sampler = RandomSampler(dataset)
        self.dataloader = DataLoader(dataset, sampler=sampler, **dataset_params)
        self.rank = get_rank()

    @torch.no_grad()
    def _do_eval(self):
        tot_res = {}
        self.trainer.model.eval()
        pbar = ProgressBar(total=len(self.dataloader), desc="eval")
        for batch in self.dataloader:
            res = self._eval_func(self.trainer.model, batch)
            for k, v in res.items():
                tot_res.setdefault(k, []).extend(v)
            pbar.update(1)
        self.trainer.model.train()
        if self.use_dist:
            tot_res = collect_result_gpu(tot_res)
        if self.rank == 0 and tot_res:
            rename_res = {self.prefix + k: np.mean(v) for k, v in tot_res.items()}
            self.log(self.trainer.epoch, **rename_res, smooth=False, window_size=1)

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
            if self.rank == 0:
                self.save_model()
