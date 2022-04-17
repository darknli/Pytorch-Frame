from typing import Callable, Optional
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from .checkpoint_hook import CheckpointerHook
import numpy as np


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
