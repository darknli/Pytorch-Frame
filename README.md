# Pytorch Frame
原代码来自https://github.com/machineko/coreml_torch_utils
，此为改版

# 单卡训练
使用Trainer训练，例如下面伪代码
```commandline
# 创建dataset和dataloader
train_dataset = Dataset(...)
train_dataloader = DataLoader(...)

# 创建网络相关对象
model = get_model(conf)
optimizer = Adam(model, lr)
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


