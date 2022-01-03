from typing import Callable

from .hookbase import HookBase


class EvalHook(HookBase):
    """周期性执行的评估器, 在每个epoch的最后阶段执行"""

    def __init__(self, period: int, eval_func: Callable):
        """

        Parameters
        ----------
        period : int. 执行eval_func函数的周期
        eval_func: Callable. 一个函数, 没有输入参数, 返回一个评估结果的dict
        """
        self._period = period
        self._eval_func = eval_func

    def _do_eval(self):
        res = self._eval_func()

        if res:
            assert isinstance(res, dict), f"评估函数应该返回一个dict. 但得到了{res}"
            for k, v in res.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        f"[EvalHook] eval_func 应该返回一个float类型的dict, 但得到了 '{k}: {v}'"
                    ) from e
            self.log(self.trainer.epoch, **res, smooth=False)

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
