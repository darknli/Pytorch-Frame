import os
import os.path as osp
from typing import Any, Dict, List, Optional

from .hookbase import HookBase


class CheckpointerHook(HookBase):
    """
    周期性保存参数
    """

    def __init__(self, period: int, max_to_keep: Optional[int] = None) -> None:
        """
        初始化
        Parameters
        ----------
        period : int, 保存checkpoints的周期
        max_to_keep : int, 保存checkpoints的数量, 更早期的checkpoints会被删除
        """
        self._period = period
        assert max_to_keep is None or max_to_keep > 0
        self._max_to_keep = max_to_keep

        self._recent_checkpoints: List[str] = []

    def after_epoch(self) -> None:
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            epoch = self.trainer.epoch  # ranged in [0, max_epochs - 1]
            checkpoint_name = f"epoch_{epoch}.pth"
            self.trainer.save_checkpoint(checkpoint_name)

            if self._max_to_keep is not None:
                self._recent_checkpoints.append(checkpoint_name)
                if len(self._recent_checkpoints) > self._max_to_keep:
                    # delete the oldest checkpoint
                    file_name = self._recent_checkpoints.pop(0)
                    file_path = osp.join(self.trainer.ckpt_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "trainer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
