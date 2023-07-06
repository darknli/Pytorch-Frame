# Pytorch Frame
原代码来自https://github.com/machineko/coreml_torch_utils
，此为改版
# 版本更新
## v1.6 相关更新
* v1.6.3 
  * 添加了EvalTotalHook，在所有数据跑完之后才开始计算指标。
  * 添加了metric模块，并增加了目标检测评估指标MAP的metric。
  * 调整了代码层级结构，使其扩展性更强。
* v1.6.2
  * 完善了training时每个step模型返回的类型，支持：
    * torch.Tensor，返回的是需要backward的tensor，即为total loss
    * dict，返回的是各路loss头，需要注意的是，这种不应加入total loss，因为torch-frame会自动合并其结果给出total loss
    * tuple， 二元组，分别为
      * backward_params：跟梯度回传、模型参数更新有关，可选torch.Tensor或dict，参考前两种
      * metric_params：dict，纯指标参数，不参与梯度回传
* v1.6.3
  * 调整代码结构，增强通用性
  * 加入多卡计算验证集指标的hook
* v1.6.4
  * 加入目标检测map的metric和相应的hook
* v1.6.5
  * 修复EvalTotalHook bug
* v1.6.6
  * 加入gpu型号性能评估的函数gpu_cnn_speed
* v1.6.7
  * 优化打印日志体验
* v1.6.8
  * 修复个别环境colors模块调用崩溃的问题
## v1.7 相关更新
* v1.7.0
  * 取消iter机制，减少每个epoch开始之前的卡顿
  * 加入高效获取数据的模块，如InfiniteDataLoader
* v1.7.1
  * 修复checkpoint_hook在win系统下保存参数崩溃的问题
* v1.7.2
  * 优化体验
* v1.7.3
  * 加入ema模块

# 安装
pip install torch-frame

# 单卡训练
使用Trainer训练，例如下面伪代码
```commandline
# 创建dataset和dataloader
from torch.util.dataset import Dataset, DataLoader
from torch_frame import Trainer

train_dataset = Dataset(...)
train_dataloader = DataLoader(...)

# 创建网络相关对象
model = get_model(conf)
optimizer = Adam(model.parameters(), lr)
lr_scheduler = MultiStepLR(optimizer, ...)

# 创建hooker，承载验证集部分和评估保存模型的任务
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
使用DDPTrainer训练，以multiprocessing的方式举例
```commandline
from torch.util.dataset import Dataset, DataLoader
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


