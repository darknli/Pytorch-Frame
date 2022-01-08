# from ..trainer import Trainer, MetricStorage
# import numpy as np


class HookBase:
    """
    hooks的基类

    Hook类在Trainer类中被初始化。每个Hook可以在六个阶段执行, 对应Trainer的六个方法分别为：

    * before_train, 训练前
    * after_train, 训练后
    * before_epoch, 一轮epoch前
    * after_epoch, 一轮epoch后
    * before_iter, 一个iter前
    * after_iter, 一个iter后
    目前在Hook中, 不能得到通过self.trainer得到类似model、optimizer等信息

    Examples:

    >>> hook.before_train()
    >>> for epoch in range(start_epoch, max_epochs):
    >>>     hook.before_epoch()
    >>>     for iter in range(epoch_len):
    >>>         hook.before_iter()
    >>>         train_one_iter()
    >>>         hook.after_iter()
    >>>     hook.after_epoch()
    >>> hook.after_train()
    """

    # A weak reference to the trainer object. Set by the trainer when the hook is registered.
    trainer: "torch_frame.Trainer" = None

    def before_train(self) -> None:
        """整体训练前调用"""
        pass

    def after_train(self) -> None:
        """全部训练结束后调用"""
        pass

    def before_epoch(self) -> None:
        """epoch前调用"""
        pass

    def after_epoch(self) -> None:
        """epoch结束后调用"""
        pass

    def before_iter(self) -> None:
        """iter前调用"""
        pass

    def after_iter(self) -> None:
        """iter结束后调用"""
        pass

    @property
    def checkpointable(self) -> bool:
        """A hook is checkpointable when it has :meth:`state_dict` method.
        Its state will be saved into checkpoint.
        """
        return callable(getattr(self, "state_dict", None))

    @property
    def class_name(self) -> str:
        """返回类名"""
        return self.__class__.__name__

    @property
    def metric_storage(self) -> "torch_frame.MetricStorage":
        return self.trainer.metric_storage

    def log(self, *args, **kwargs) -> None:
        self.trainer.log(*args, **kwargs)

    # belows are some helper functions that are often used in hook
    def every_n_epochs(self, n: int) -> bool:
        return (self.trainer.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, n: int) -> bool:
        return (self.trainer.iter + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, n: int) -> bool:
        return (self.trainer.inner_iter + 1) % n == 0 if n > 0 else False

    def is_last_epoch(self) -> bool:
        return self.trainer.epoch == self.trainer.max_epochs - 1

    def is_last_iter(self) -> bool:
        return self.trainer.iter == self.trainer.max_iters - 1

    def is_last_inner_iter(self) -> bool:
        return self.trainer.inner_iter == self.trainer.epoch_len - 1
