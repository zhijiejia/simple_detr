

## 仓库的目的
- 看懂detr的代码, 但是我只想看的是detr怎么实现的目标检测, 并不需要看detr如果实现全景分割和关键点检测的, 所以原始的代码中很多地方对于我很冗余, 这些冗余会影响我看目标检测代码的进度，不太需要，所以删除，使得代码结构更加清晰明了

## 仓库的主要修改
- 无关目标检测代码的删除
- 验证阶段，由DDP模式改为单进程模型
- 修改指标计算部分代码, 因为验证阶段变为单进程模型, 因此借助pycocotools实现更简单
- 官方的ReadMe在: [Official ReadMe](Official_README.md)

## 如何运行

- 启动
```python
    python -m torch.distributed.launch --use_env --nproc_per_node=2 main.py
    # 上面命令中, 2的意思是 我这次训练在DDP模型下, 同时使用2张卡
```

- eval
```python
    python -m torch.distributed.launch --use_env --nproc_per_node=2 main.py --eval
    # 只是测试验证阶段, 不会加载训练好的模型权重
```
