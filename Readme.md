## Multi-Agent PPO Algorithm

本代码基于python 3.10的pytorch环境。训练的入口函数为train.py，命令行启动时需要附带命令行参数。

```
python train.py -n [1,2,3]
```

其中1代表cppo算法，2代表mappo算法，3代表ippo算法；

或者，也可以使用run.bat来批量运行。

在train.py中，torch和vmas环境的种子都被固定。因此，重复训练会得到相同的结果。可以在train.py的main入口将固定种子的if分支改为false，以进行随机测试。

config.py中定义了训练过程中的各种超参数，如有需要可以在此处进行调整。

