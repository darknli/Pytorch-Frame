import os
import pickle
import os.path as osp
from typing import Any, Dict, List, Optional
from types import LambdaType
import logging
from .hookbase import HookBase

logger = logging.getLogger(__name__)


class CheckpointerHook(HookBase):
    """
    周期性保存参数
    """

    def __init__(self, period: int = 1,
                 max_to_keep: Optional[int] = None,
                 save_metric: Optional[str] = None,
                 max_first: bool = True,
                 save_last: bool = True
                 ) -> None:
        """
        初始化
        Parameters
        ----------
        period : int, 保存checkpoints的周期
        max_to_keep : int, 保存checkpoints的数量, 更早期的checkpoints会被删除
        save_metric : int, default None.
            保存模型的指标是哪个, 需要从trainer.metric_storage选择
        max_first : bool, default True.
            用于保存模型的指标是取最大还是最小作为最优模型
        save_last : bool, default True
            是否保存最近一次的epoch的模型, 如果是True, 每轮将更新模型到latest.pth中
        """
        self._period = period
        assert max_to_keep is None or max_to_keep > 0
        self._max_to_keep = max_to_keep
        if save_metric is None:
            logger.warning("没有指定保存模型的指标，因此每period都将保存模型")
        self.save_metric = save_metric
        if max_first:
            self.cur_best = float("-inf")
            self.is_better = lambda a: a > self.cur_best
        else:
            self.cur_best = float("inf")
            self.is_better = lambda a: a < self.cur_best

        self.save_last = save_last

        self._recent_checkpoints: List[str] = []

    def after_epoch(self) -> None:
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self.save_model()

    def save_model(self):
        if self.save_last:
            self.trainer.save_checkpoint("latest.pth", False)

        # 如果当前epoch指标没有更好, 则不保存模型
        if self.save_metric is not None:
            if not self.is_better(self.trainer.metric_storage[self.save_metric]):
                return
            else:
                self.cur_best = self.trainer.metric_storage[self.save_metric].avg
                logger.info(f"{self.save_metric} update to {round(self.cur_best, 4)}")

        self.trainer.save_checkpoint("best.pth")
        if self._max_to_keep is not None and self._max_to_keep >= 1:
            epoch = self.trainer.epoch  # ranged in [0, max_epochs - 1]
            checkpoint_name = f"epoch_{epoch}.pth"
            self.trainer.save_checkpoint(checkpoint_name)
            self._recent_checkpoints.append(checkpoint_name)
            if len(self._recent_checkpoints) > self._max_to_keep:
                # delete the oldest checkpoint
                file_name = self._recent_checkpoints.pop(0)
                file_path = osp.join(self.trainer.ckpt_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        for key, value in self.__dict__.items():
            if key == "trainer" or isinstance(value, LambdaType):
                continue
            try:
                pickle.dumps(value)
            except BaseException:
                continue
            state[key] = value
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
