from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import ProjectConfiguration, set_seed
import warnings
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Union, List, Dict, Any
from .trainer import Trainer, MetricStorage, EMA
from .hooks import CheckpointerHook, HookBase, LoggerHook
from .utils.misc import get_workspace
from .lr_scheduler import LRWarmupScheduler
from ._get_logger import logger


class AccelerateTrainer(Trainer):
    """
    accelerate版的Trainer,底层以来accelerate库
    其他参数都和Trainer一致, 除了以下参数
    Parameters
    ---------
    lr_scheduler: Union[str, optim.lr_scheduler._LRScheduler]. 会有以下差别:
        增加str类型，这时会直接调用diffusers内部的lr_scheduler, 这里建议使用str类型
        支持["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    mixed_precision: str, default None
         * "no", 不使用amp技术
         * "fp16"
         * "bf16", 30系及以后的卡型才能使用
    gradient_accumulation_steps: int, default 1. 梯度累计数，显存不够用又需要大batch的时候可以加大数值
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            lr_scheduler: Union[str, optim.lr_scheduler._LRScheduler],
            data_loader: DataLoader,
            max_epochs: int,
            work_dir: str = "work_dir",
            clip_grad_norm: float = 0.0,
            mixed_precision: str = None,
            warmup_method: Optional[str] = None,
            warmup_iters: int = 0,
            warmup_factor: float = 0.001,
            hooks: Optional[List[HookBase]] = None,
            use_ema: bool = False,
            ema_decay: float = 0.9999,
            gradient_accumulation_steps: int = 1,
            create_new_dir: Optional[str] = "time_s"
    ):
        self.work_dir = get_workspace(work_dir, create_new_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.work_dir, logging_dir=self.work_dir)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            project_config=accelerator_project_config,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.model = model
        if use_ema:
            warnings.warn("accelerate版暂不支持ema, 敬请期待")
            self.model_ema = EMA(self.model_or_module, ema_decay)
        else:
            self.model_ema = None
        self.optimizer = optimizer
        # convert epoch-based scheduler to iteration-based scheduler
        if isinstance(lr_scheduler, str):
            from diffusers.optimization import get_scheduler
            if warmup_method is not None:
                self.logger_print("当`lr_scheduler`输入str类型时, `warmup_method`参数失效", warnings.warn)
            max_train_steps = max_epochs * len(data_loader)
            self.lr_scheduler = get_scheduler(
                lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=warmup_iters * self.accelerator.num_processes,
                num_training_steps=max_train_steps * self.accelerator.num_processes,
            )
        else:
            self.lr_scheduler = LRWarmupScheduler(
                lr_scheduler, len(data_loader), warmup_method, warmup_iters, warmup_factor
            )
        self.data_loader = data_loader
        self.metric_storage = MetricStorage()

        # counters
        self.inner_iter: int = -1  # [0, epoch_len - 1]
        self.epoch: int = -1  # [0, max_epochs - 1]
        self.start_epoch = 0  # [0, max_epochs - 1]
        self.max_epochs = max_epochs

        self._hooks: List[HookBase] = []
        self._clip_grad_norm = clip_grad_norm

        if self.accelerator.is_main_process:
            if hooks is None:
                hooks = self._build_default_hooks()
            self.register_hooks(hooks)

        self.info_params = []

    def prepare_model(self):
        """如果有多个模型的话可以在这里重写方法"""
        self.model, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.data_loader, self.lr_scheduler
        )

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
        self.prepare_model()  # 此处和原版trainer不同!!!
        super()._prepare_for_training(console_log_level, file_log_level)

    def train_one_iter(self, batch) -> dict:
        """
        包含了accelerate版训练的一个iter的全部操作

        .. Note::
            标准的学习率调节器是基于epoch的, 但torch_frame框架是基于iter的, 所以它在每次iter之后都会调用
        """
        # 这里貌似只支持一个模型, 如果类似SD的训练任务可能需要把text_encoder和unet都包在一个模型里
        with self.accelerator.accumulate(self.model):
            #####################
            # 1. 计算loss #
            #####################
            loss_info = self.model(batch)
            if isinstance(loss_info, torch.Tensor):
                losses = loss_info
                loss_info = {"total_loss": loss_info}
            elif isinstance(loss_info, tuple):
                assert len(loss_info) == 2, "loss_info需要是一个二元组，第一个是需要反向传播的，第二个是其他参考指标"
                backward_params, metric_params = loss_info
                assert isinstance(metric_params, dict), "loss_info的第二个值需要是dict类型"
                if isinstance(backward_params, torch.Tensor):
                    losses = backward_params
                    metric_params["total_loss"] = backward_params
                    loss_info = metric_params
                elif isinstance(backward_params, dict):
                    losses = sum(backward_params.values())
                    backward_params["total_loss"] = losses
                    backward_params.update(metric_params)
                    loss_info = backward_params
            else:
                assert "total_loss" not in loss_info, "当model返回是一个dict的时候不可以传出包含"
                losses = sum(loss_info.values())
                loss_info["total_loss"] = losses

            ##########################
            # 2. 计算梯度 #
            ##########################
            self.optimizer.zero_grad()
            self.accelerator.backward(losses)
            if self._clip_grad_norm > 0 and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

            ##############################
            # 3. 更新模型参数 #
            ##############################
            self.optimizer.step()

            ###########################
            # 4. 调整学习率 #
            ###########################
            self.lr_scheduler.step()

        if self.model_ema:
            self.model_ema.update(self.model)

        show_info = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in loss_info.items()}
        show_info = dict(sorted(show_info.items(), key=lambda x: x[0] != "total_loss"))  # 保证total_loss在最后一位
        self.pbar.set_postfix(show_info)
        return show_info

    @property
    def model_or_module(self) -> nn.Module:
        if self.model.__class__.__name__ == "DistributedDataParallel":
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def check_main(self):
        """判断是否为主进程, 对于单卡来说永远是True, 对于多卡来说只有一个主进程"""
        return self.accelerator.is_main_process
