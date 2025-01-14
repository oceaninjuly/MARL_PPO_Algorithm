import numpy as np
import torch

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self,capacity=200):
        self.init_flag = False
        self.capacity = capacity
        self.buffer = []
    
    def add(self,*args):
        if not self.init_flag:
            self.init_flag = True
            for arg in args:
                self.buffer.append([arg])
        else:
            for i,arg in enumerate(args):
                self.buffer[i].append(arg)

    def __len__(self):
        return 0 if not self.init_flag else len(self.buffer[0])

    def get(self):
        return tuple([self.buffer[i] for i in range(len(self.buffer))])
    
    def clear(self):
        self.buffer = []
        self.init_flag = False


# 帧堆叠管理器
class Stack_Manager:
    def __init__(self,init_obs,num_stack=4):
        self.stack = [torch.cat([init_obs[i] for _ in range(num_stack)]) for i in range(len(init_obs))]
        self.obs_size = init_obs[0].shape[1]
        self.capacity = self.obs_size*num_stack

    def add(self,obs):
        for i in range(len(obs)):
            self.stack[i] = torch.cat((self.stack[i],obs[i]),dim=-1)
            self.stack[i] = self.stack[i][:,self.obs_size:]

    def get(self): return self.stack 