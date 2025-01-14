import numpy as np
import torch
from config import device

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self,capacity=200):
        self.init_flag = False
        self.capacity = capacity
        self.buffer = []
        self.counter=0
    
    def add(self,*args):
        if not self.init_flag:
            self.init_flag = True
            for arg in args:
                self.buffer.append(torch.zeros((self.capacity,)+arg.shape[1:],dtype=arg.dtype).to(device))
                self.buffer[-1][0]=arg[0]
        else:
            for i,arg in enumerate(args):
                self.buffer[i][self.counter]=arg[0]
        self.counter+=1

    def __len__(self):
        return 0 if not self.init_flag else self.counter

    def get(self):
        return tuple([self.buffer[i] for i in range(len(self.buffer))])
    
    def clear(self):
        self.buffer = []
        self.init_flag = False
        self.counter = 0


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