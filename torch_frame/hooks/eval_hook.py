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
            一个函数, 输入参数是model和batch, 返回一个评估结果的Dict[list], k对应指标名称, v是包含每个样本得分的list
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

    @torch.no_grad()
    def _do_eval(self):
        tot_res = {}
        mode_model = self.trainer.model_evaluate.training
        self.trainer.model_evaluate.eval()
        with tqdm(self.dataloader, desc="eval") as pbar:
            for batch in pbar:
                res = self._eval_func(self.trainer.model_evaluate, batch)
                for k, v in res.items():
                    tot_res.setdefault(k, []).extend(v)
        self.trainer.model_evaluate.train(mode_model)
        if tot_res:
            rename_res = {self.prefix + k: np.mean(v) for k, v in tot_res.items()}
            self.log(self.trainer.epoch, **rename_res, smooth=False, window_size=1)

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
            self.save_model()


class EvalTotalHook(CheckpointerHook):
    """
    `CheckpointerHook` 的派生类, 周期性执行的评估器, 在每个epoch的最后阶段执行.
    与`EvalHook`评估器区别是: 一个是每个batch都去评估, 最后求每次评估结果的均值; 一个是先把batch结果存下来, 最后一起评估
    """

    def __init__(self,
                 dataloader: DataLoader,
                 eval_metric: object,
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
        eval_metric : object.
            一个对象, 需要包含`update`和`evaluate`方法. 其中:
            * `update`方法, 需要输入参数是model和batch, 无返回值, 建议在内部存储当前batch的值
            * `evaluate`方法, 无形参, return的是一个Dict[str, float]类型的评估结果
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
        super(EvalTotalHook, self).__init__(period, max_to_keep, self.prefix + save_metric, max_first, save_last)
        assert hasattr(eval_metric, "update") and isinstance(getattr(eval_metric, "update"), Callable)
        assert hasattr(eval_metric, "evaluate") and isinstance(getattr(eval_metric, "evaluate"), Callable)
        self._eval_metric = eval_metric
        self.dataloader = dataloader

    @torch.no_grad()
    def _do_eval(self):
        mode_model = self.trainer.model_evaluate.training
        self.trainer.model_evaluate.eval()
        with tqdm(self.dataloader, desc="eval") as pbar:
            for batch in pbar:
                self._eval_metric.update(self.trainer.model_evaluate, batch)
        self.trainer.model_evaluate.train(mode_model)
        tot_res = self._eval_metric.evaluate()
        self._eval_metric.reset()
        if tot_res:
            rename_res = {self.prefix + k: np.mean(v) for k, v in tot_res.items()}
            self.log(self.trainer.epoch, **rename_res, smooth=False, window_size=1)

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
            self.save_model()