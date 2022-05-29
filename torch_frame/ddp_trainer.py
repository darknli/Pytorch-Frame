from .trainer import Trainer, nn, optim, Optional, List, HookBase, torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class DDPTrainer(Trainer):
    """
    Trainer的DDP版本。需要注意的是，在调用该类前依然要执行init_process_group，为了避免重复打印或保存数据，应该在外部选择性
    的根据gpu_id调用hooker
    """

    def __init__(self,
                 use_dist: bool,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 lr_scheduler: optim.lr_scheduler._LRScheduler,
                 dataset: Dataset,
                 dataset_params: dict,
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
        use_dist : bool, 是否使用ddp
        model : torch.nn.Module, 模型
        optimizer : torch.optim.Optimizer, 优化器
        lr_scheduler : optim.lr_scheduler._LRScheduler, 学习率调节器
        dataset : torch.utils.data.Dataset, 训练集数据生成器, 不需要创建dataloader, 由DDPTrainer内部创建
        dataset_params : dict, dataset的参数, key是诸如batch_size的参数
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
        assert torch.cuda.is_available(), "DDPtrainer必须要是可以运行cuda的"

        if use_dist:
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler_trainer = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank)
        else:
            sampler_trainer = RandomSampler(dataset)
        self.data_loader = DataLoader(dataset, sampler=sampler_trainer, **dataset_params)
        super(DDPTrainer, self).__init__(model, optimizer, lr_scheduler, self.data_loader, max_epochs, work_dir,
                                         clip_grad_norm, enable_amp,  warmup_method, warmup_iters, warmup_factor,
                                         hooks, create_new_dir)
        self.use_dist = use_dist

    def _train_one_epoch(self) -> None:
        """执行模型一个epoch的全部操作"""
        if self.use_dist:
            dist.barrier()
            self.data_loader.sampler.set_epoch(self.epoch)
        super(DDPTrainer, self)._train_one_epoch()


        