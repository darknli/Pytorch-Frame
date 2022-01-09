import numpy as np
from collections import deque


class HistoryBuffer:
    """
    该类保存了一些数, 并且可以得到平滑均值

    Example::

        >>> his_buf = HistoryBuffer()
        >>> his_buf.update(0.1)
        >>> his_buf.update(0.2)
        >>> his_buf.avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        """
        Parameters
        ----------
        window_size : int, default 20. 滑窗大小
        """
        self._history = deque(maxlen=window_size)
        self._count: int = 0
        self._sum: float = 0.0

    def update(self, value: float) -> None:
        """
        在列表新增变量, 如果新增后超出窗口大小, 舍去最早的那个值
        """
        self._history.append(value)
        self._count += 1
        self._sum += value

    @property
    def latest(self) -> float:
        return self._history[-1]

    @property
    def avg(self) -> float:
        return np.mean(self._history)

    @property
    def global_avg(self) -> float:
        return self._sum / self._count

    @property
    def global_sum(self) -> float:
        return self._sum

    def __le__(self, other):
        return self.avg <= other

    def __lt__(self, other):
        return self.avg < other

    def __ge__(self, other):
        return self.avg >= other

    def __gt__(self, other):
        return self.avg > other

    def __str__(self):
        return str(round(self.avg, 4))
