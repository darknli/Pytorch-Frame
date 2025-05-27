# Pytorch Frame
原代码来自https://github.com/machineko/coreml_torch_utils
，此为改版。在原版基础之上加入了大量功能

# 安装
❌ pip方式不再维护：
~~pip install torch-frame~~

✅ 推荐使用pip install git+https://github.com/darknli/Pytorch-Frame.git
# 单卡训练
使用Trainer训练，下面是代码示例（Trainer支持混合精度训练，可自行去Trainer类中翻阅）
```commandline
# 创建dataset和dataloader
from torch.util.dataset import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch_frame import Trainer

train_dataset = Dataset(...)
train_dataloader = DataLoader(...)

# 创建网络相关对象
model = get_model(conf)
optimizer = Adam(model.parameters(), lr)
lr_scheduler = MultiStepLR(optimizer, ...)

# 创建hooker，承载验证集部分和评估保存模型的任务
# 这里也可以不做定制化创建，走Trainer默认的hooks，这个时候只支持log和checkpoint latest保存
hooks = [EvalHook(...), LoggerHook(...)]

# 创建Trainer对象并开始训练
trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, num_epochs, "保存路径", hooks=hooks)
traine.train()  # 开始正式训练
```
也可以加载之前训练到一半的模型以及的训练状态，接着训练
```commandline
trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, num_epochs, "保存路径", hooks=hooks)
trainer.load_checkpoint("latest.pth")
trainer.train(1, 1)
```
也可以只加载模型参数
```commandline
trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader, num_epochs, "保存路径", hooks=hooks)
weights = torch.load("best.pth")
trainer.load_checkpoint(checkpoint=weights)
trainer.train(1, 1)
```

# 多卡训练
使用DDPTrainer训练，在单卡的基础上扩展了多卡训练的能力，以multiprocessing的方式举例
```commandline
from torch.util.dataset import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch_frame import DDPTrainer
import torch.multiprocessing as mp
import torch.distributed as dist

def main(cur_gpu, args):
    if args.use_dist:
        rank = args.nprocs * args.machine_rank + cur_gpu
        dist.init_process_group(backend="nccl", init_method=args.url, world_size=args.world_size, rank=rank)
        args.train_batch_size =  args.train_batch_size // args.world_size  # 这里需要把一个batch平分到每个gpu的数量
    else:
        args.train_batch_size = args.batch_size
        
   # 创建网络相关对象
    model = get_model(conf)
    
    if args.use_dist: 
        model = torcg,nn.SyncBatchNorn.convert_batchnorm(model)
        torch.cuda.set_device(cur_gpu)
        model.cuda(cur_gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cur_gpu], find_unused_parameter=True)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    optimizer = Adam(model.parameters(), lr)
    lr_scheduler = MultiStepLR(optimizer, ...)
    
    train_dataset = Dataset(...)
    train_params = dict(batch_size=32, ...)
    
    # 这里的hooks和单卡同理
    if cur_gpu == 0:
        hooks = [EvalHook(...), LoggerHook(...)] 
    else:
        hooks = []
    
    trainer = DDPtrainer(model, optimizer, lr_scheduler, train_dataset, train_params, num_epochs, hooks=hooks,
                         use_dist=args.use_dist)
                         

if __name__ == "__name__":
    nprocs = torch.cuda.device_count()
    if args.use_dist:
        mp.spawn(main, nprocs=nprocs, args=(args, ))
    else:
        main(0, args)
    
```

# Accelerate加速训练
基于accelerate库的trainer做训练，支持多卡，相对于上述两种训练，推荐下面这种训练方式。相对于普通的Trainer只需要做少量修改即可运行
```commandline
# 创建dataset和dataloader
from torch.utils.data import Dataset, DataLoader
from torch_frame import AccelerateTrainer
from torch.optim import Adam

train_dataset = Dataset(...)
train_dataloader = DataLoader(...)

# 创建网络相关对象
model = get_model(config)
optimizer = Adam(model.parameters(), lr)
lr_scheduler = "constant"  # 这里建议用字符串而不是类似MultiStepLR的scheduler类，具体和diffusers一致

# 创建hooker，承载验证集部分和评估保存模型的任务
hooks = [EvalHook(...), LoggerHook(...)]

# 创建Trainer对象并开始训练，支持混合精度fp16/bp16
trainer = AccelerateTrainer(model, optimizer, lr_scheduler, train_dataloader, num_epochs, "保存路径", mixed_precision="fp16", hooks=hooks)
trainer.train()  # 开始正式训练
```