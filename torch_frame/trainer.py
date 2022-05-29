import logging
import os
import os.path as osp
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import datetime
from .hooks import CheckpointerHook, HookBase, LoggerHook
from .utils import setup_logger
from .lr_scheduler import LRWarmupScheduler
from .utils import HistoryBuffer
from ._get_logger import logger


class Trainer:
    """
    一个基于epoch的通用训练框架（目前只支持单gpu运行）, 包含：
    1. 计算从dataloader中计算loss
    2. 计算梯度
    3. 用optimizer更新参数
    4. 调整学习率
    如果想完成更复杂的功能, 也可以继承该类编写子类, 重写里面的`train_one_iter`等方法
    以下是代码示例
    .. code-block:: python
        model = ...          # 初始化你的模型
        optimizer = ...      # 你的优化器
        lr_scheduler = ...   # 初始化你的调节器
        data_loader = ...    # 初始化你的数据生成器
        # 训练100轮
        trainer = Trainer(model, optimizer, lr_scheduler, data_loader, max_epochs=100)
        trainer.train()
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            lr_scheduler: optim.lr_scheduler._LRScheduler,
            data_loader: DataLoader,
            max_epochs: int,
            work_dir: str = "work_dir",
            clip_grad_norm: float = 0.0,
            enable_amp=False,
            warmup_method: Optional[str] = None,
            warmup_iters: int = 1000,
            warmup_factor: float = 0.001,
            hooks: Optional[List[HookBase]] = None,
            create_new_dir: Optional[str] = "time_s"
    ):
        """
        初始化

        Parameters
        ---------
        model : torch.nn.Module, 模型
        optimizer : torch.optim.Optimizer, 优化器
        lr_scheduler : optim.lr_scheduler._LRScheduler, 学习率调节器
        data_loader : torch.utils.data.DataLoader, 数据生成器
        max_epochs : int, 训练的总轮数
        work_dir : str, 保存模型和日志的根目录地址
        clip_grad_norm : float, default 0.0
            梯度裁剪的设置, 如果置为小于等于0, 则不作梯度裁剪
        enable_amp : bool, 使用混合精度
        warmup_method : str, default None
            warmup的类型, 包含以下四种取值
            * constant
            * linear
            * exp
            * None : 不使用warmup
        warmup_iters : int, default 1000, warmup最后的iter数
        warmup_factor : float, default 0.001
            warmup初始学习率 = warmup_factor * initial_lr
        hooks : List[HookBase], default None.
            hooks, 保存模型、输出评估指标、loss等用
        create_new_dir : Optional[str], default time
            存在同名目录时以何种策略创建目录
            * None, 直接使用同名目录
            * `time_s`, 如果已经存在同名目录, 则以时间(精确到秒)为后缀创建新目录
            * `time_m`, 如果已经存在同名目录, 则以时间(精确到分)为后缀创建新目录
            * `time_h`, 如果已经存在同名目录, 则以时间(精确到小时)为后缀创建新目录
            * `time_d`, 如果已经存在同名目录, 则以时间(精确到日)为后缀创建新目录
            * `count`, 如果已经存在同名目录, 则以序号为后缀创建新目录
        """
        logger.setLevel(logging.INFO)

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

        self.model = model
        self.optimizer = optimizer
        # convert epoch-based scheduler to iteration-based scheduler
        self.lr_scheduler = LRWarmupScheduler(
            lr_scheduler, len(data_loader), warmup_method, warmup_iters, warmup_factor
        )
        self.data_loader = data_loader
        self.work_dir = work_dir
        self.metric_storage = MetricStorage()

        # counters
        self.inner_iter: int = -1  # [0, epoch_len - 1]
        self.epoch: int = -1  # [0, max_epochs - 1]
        self.start_epoch = 0  # [0, max_epochs - 1]
        self.max_epochs = max_epochs

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp

        if self._enable_amp:
            logger.info("自动混合精度 (AMP) 训练")
            self._grad_scaler = GradScaler()

        if hooks is None:
            self.register_hooks(self._build_default_hooks())
        else:
            self.register_hooks(hooks)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def epoch_len(self) -> int:
        return len(self.data_loader)

    @property
    def max_iters(self) -> int:
        return self.max_epochs * self.epoch_len

    @property
    def cur_iter(self) -> int:
        """返回当前iter数, 范围在 [0, max_iters - 1]."""
        return self.epoch * self.epoch_len + self.inner_iter

    @property
    def start_iter(self) -> int:
        """从哪一个iter开始. 最小的值是0."""
        return self.start_epoch * self.epoch_len

    @property
    def ckpt_dir(self) -> str:
        return osp.join(self.work_dir, "checkpoints")

    @property
    def tb_log_dir(self) -> str:
        return osp.join(self.work_dir, "tb_logs")

    @property
    def log_file(self) -> str:
        return osp.join(self.work_dir, "log.txt")

    @property
    def model_or_module(self) -> nn.Module:
        if isinstance(self.model, (DistributedDataParallel, DataParallel)):
            return self.model.module
        return self.model

    @property
    def registered_hook_names(self) -> List[str]:
        """注册的所有hook名字"""
        return [h.__class__.__name__ for h in self._hooks]

    def log(self, *args, **kwargs) -> None:
        """更新评估指标"""
        self.metric_storage.update(*args, **kwargs)

    def _prepare_for_training(self,
                              console_log_level: int = 2,
                              file_log_level: int = 2) -> None:
        """
        训练前的配置工作
        Parameters
        ----------
        console_log_level : int, default 2
             输出到屏幕的log等级, 可选范围是0-5, 它们对应的关系分别为：
             * 5: FATAL
             * 4: ERROR
             * 3: WARNING
             * 2: INFO
             * 1: DEBUG
             * 0: NOTSET
        file_log_level : int, default 2
             输出到文件里的log等级, 其他方面同console_log_level参数
        """
        # setup the root logger of the `cpu` library to show
        # the log messages generated from this library
        assert console_log_level in (0, 1, 2, 3, 4, 5), f"console_log_level必须在0~5之间而不是{console_log_level}"
        assert file_log_level in (0, 1, 2, 3, 4, 5), f"file_log_level必须在0~5之间而不是{file_log_level}"
        console_log_level *= 10
        file_log_level *= 10
        setup_logger("torch_frame", output=self.log_file,
                     console_log_level=console_log_level, file_log_level=file_log_level)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.start_epoch == 0:
            logger.info(f"Registered default hooks: {self.registered_hook_names}")
            split_line = "-" * 50
            logger.info(
                f"\n{split_line}\n"
                f"Work directory: {self.work_dir}\n"
                f"{split_line}"
            )

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Trainer运行时调用hook
        hook执行时根据它们注册的顺序来进行

        Parameters
        ---------
        hooks : list[HookBase]
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other. This normally
            # does not matter, but will cause memory leak if the involved objects contain __del__.
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
            # We always keep :class:`LoggerHook` as the last hook to avoid losing any records
            # that should have been logged. The order of other hooks remains the same.
            if self._hooks and isinstance(self._hooks[-1], LoggerHook):
                self._hooks.insert(len(self._hooks) - 1, h)
            else:
                self._hooks.append(h)

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _build_default_hooks(self) -> List[HookBase]:
        return [
            CheckpointerHook(),
            LoggerHook(tb_log_dir=self.tb_log_dir),
        ]

    def _update_iter_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float,
                          iter_time: float, lr: float) -> None:
        """
        每个iter评估的log
        Parameters
        ----------
        loss_dict : dict, losses的标量字典
        data_time : float, dataloader的一个iter耗时
        iter_time : float, 一个iter全部耗时
        lr : float, 该iter的学习率
        """
        self.log(self.cur_iter, data_time=data_time, iter_time=iter_time)
        self.log(self.cur_iter, lr=lr, smooth=False)

        loss_dict = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        loss_value = sum(loss_dict.values())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"Loss became infinite or NaN at epoch={self.epoch}! loss_dict = {loss_dict}."
            )

        self.log(self.cur_iter, total_loss=loss_value)
        if len(loss_dict) > 1:
            self.log(self.cur_iter, **loss_dict)

    def train_one_iter(self) -> None:
        """
        包含了训练的一个iter的全部操作

        .. Note::
            标准的学习率调节器是基于epoch的, 但torch_frame框架是基于iter的, 所以它在每次iter之后都会调用
        """
        iter_start_time = time.perf_counter()
        lr_this_iter = self.lr

        ######################
        # 1. 加载一个batch的数据 #
        ######################
        # 这里读取生成器的数据而非data_loader这是为了计算加载数据的耗时
        start = time.perf_counter()
        batch = next(self._data_iter)
        data_time = time.perf_counter() - start

        #####################
        # 2. 计算loss #
        #####################
        if self._enable_amp:
            with autocast():
                loss_dict = self.model(batch)
        else:
            loss_dict = self.model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        ##########################
        # 3. 计算梯度 #
        ##########################
        self.optimizer.zero_grad()
        if self._enable_amp:
            self._grad_scaler.scale(losses).backward()
        else:
            losses.backward()
        if self._clip_grad_norm > 0:
            if self._enable_amp:
                self._grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

        ##############################
        # 4. 更新模型参数 #
        ##############################
        if self._enable_amp:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()

        ###########################
        # 5. 调整学习率 #
        ###########################
        self.lr_scheduler.step()

        show_info = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        self.pbar.set_postfix(show_info)
        self._update_iter_metrics(loss_dict, data_time, time.perf_counter() - iter_start_time, lr_this_iter)

    def _train_one_epoch(self) -> None:
        """执行模型一个epoch的全部操作"""
        self.model.train()
        self.pbar = tqdm(total=self.epoch_len, desc=f"epoch={self.epoch}", ascii=True)
        for self.inner_iter in range(self.epoch_len):
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
            self.pbar.update(1)
        self.pbar.close()
        del self.pbar
        self._data_iter = iter(self.data_loader)

    def train(self,
              console_log_level: int = 2,
              file_log_level: int = 2) -> None:
        """
        训练入口

        Parameters
        ----------
        console_log_level : int, default 2.
            输出到屏幕的log等级, 可选范围是0-5, 它们对应的关系分别为：
            * 5: FATAL
            * 4: ERROR
            * 3: WARNING
            * 2: INFO
            * 1: DEBUG
            * 0: NOTSET
        file_log_level : int, default 2.
            输出到文件里的log等级, 其他方面同console_log_level参数
        """
        logger.info(f"开始从第{self.start_epoch}epoch训练")
        self._prepare_for_training(console_log_level, file_log_level)
        self._call_hooks("before_train")
        for self.epoch in range(self.start_epoch, self.max_epochs):
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
        self._call_hooks("after_train")

    def save_checkpoint(self, file_name: str, save_single_model: bool = True,
                        print_info: bool = False) -> None:
        """
        保存参数, 包含:

        * epoch : 当前轮数
        * model : 当前模型参数
        * optimizer : 当前优化器
        * lr_scheduler : 当前调节器
        * metric_storage : 评估指标
        * hooks(非必须) : 一些中间量
        * grad_scaler(非必须) : 混合精度的参数

        Parameters
        ----------
        file_name : str, 保存文件名
        save_single_model : bool, default True. 如果是True, 还会额外保存一个只有模型参数的文件到best_model.pth
        print_info : bool, default True. 如果是True, 则输出保存模型的提示信息
        """

        data = {
            "epoch": self.epoch,
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage,
            "work_dir": self.work_dir,
        }
        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states
        if self._enable_amp:
            data["grad_scaler"] = self._grad_scaler.state_dict()

        file_path = osp.join(self.ckpt_dir, file_name)
        if print_info:
            logger.info(f"Saving checkpoint to {file_path}")
        if save_single_model:
            torch.save(self.model_or_module.state_dict(), file_path)
        else:
            torch.save(data, file_path)

    def load_checkpoint(self, path: str = None, checkpoint: Dict[str, Any] = None):
        """
        加载参数

        Parameters
        ----------
        path : str, default None. checkpoint的地址
        checkpoint : dict, default None.
            如果path非空, 优先使用path的数据, 否则直接加载checkpoint的数据。
            直接加载的时候，将只加载模型，而不带各种状态
        """
        assert checkpoint is None or path is None
        if path is None:
            incompatible = self.model_or_module.load_state_dict(checkpoint, strict=False)
            if incompatible.missing_keys:
                logger.warning("Encounter missing keys when loading model weights:\n"
                               f"{incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning("Encounter unexpected keys when loading model weights:\n"
                               f"{incompatible.unexpected_keys}")
            logger.info("只加载模型本身...")
            return

        checkpoint = torch.load(path, map_location="cpu")

        # 1. 加载 epoch
        self.start_epoch = checkpoint["epoch"] + 1

        # 2. 加载 metric_storage
        self.metric_storage = checkpoint["metric_storage"]

        # 3. 加载 optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 4. 加载 lr_scheduler
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 5. 加载 grad scaler
        consistent_amp = not (self._enable_amp ^ ("grad_scaler" in checkpoint))
        assert consistent_amp, "Found inconsistent AMP training setting when loading checkpoint."
        if self._enable_amp:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        # 6. 加载 模型
        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning("Encounter missing keys when loading model weights:\n"
                           f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning("Encounter unexpected keys when loading model weights:\n"
                           f"{incompatible.unexpected_keys}")

        # 7. 加载 hooks
        hook_states = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}")

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break

        # 8. 加载保存目录
        self.work_dir = checkpoint["work_dir"]

        if path:
            logger.info(f"加载模型{path}成功")


class MetricStorage(dict):
    """The class stores the values of multiple metrics (some of them may be noisy, e.g., loss,
    batch time) in training process, and provides access to the smoothed values for better logging.

    The class is designed for automatic tensorboard logging. User should specify the ``smooth``
    when calling :meth:`update`, in order to we can determine which metrics should be
    smoothed when performing tensorboard logging.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss will be smoothed, but lr will not
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}
        >>> # like dict, can be indexed by string
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, default_win_size: int = 20) -> None:
        self._default_win_size = default_win_size
        self._history: Dict[str, HistoryBuffer] = self
        self._smooth: Dict[str, bool] = {}
        self._latest_iter: Dict[str, int] = {}

    def update(self, iter: Optional[int] = None, smooth: bool = True, window_size: int = None, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (int): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (bool): If True, return the smoothed values of these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same ``smooth`` in different calls to :meth:`update`.
            window_size : int
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
                self._history[key] = HistoryBuffer(window_size=window_size if window_size else self._default_win_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key], "检查total_loss是不是存在于model给出的loss_dict中"
                self._latest_iter[key] = iter
            else:
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """Return the smoothed values or the latest values of multiple metrics.
        The specific behavior depends on the ``smooth`` when updating metrics.

        Returns:
            dict[str -> (int, float)]: Mapping from metric name to its
                (the latest iteration, the avg/latest value) pair.
        """
        return {
            key: (self._latest_iter[key], his_buf.avg if self._smooth[key] else his_buf.latest)
            for key, his_buf in self._history.items()
        }
