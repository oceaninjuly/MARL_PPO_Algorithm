import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def init_weights(m,gain):
    if isinstance(m,nn.Linear):
        init.orthogonal_(m.weight,gain=gain)
        if m.bias is not None:
            init.constant_(m.bias,0)
 

class Net_Policy(nn.Module):
    def __init__(self,obs_size,action_dim,n_agents,frame_stack=4):
        hidden_size = 256
        self.action_size = action_dim*n_agents
        input_dim = n_agents*obs_size*frame_stack
        self.gain=init.calculate_gain('relu')
        
        super().__init__()
        self.l1 = nn.Sequential(
            self.init_(nn.Linear(input_dim,hidden_size//2),self.gain),
            nn.LayerNorm(hidden_size//2),
            nn.ReLU(),
            self.init_(nn.Linear(hidden_size//2,2*hidden_size),self.gain),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.init_(nn.Linear(hidden_size,2*hidden_size),self.gain),
            nn.MaxPool1d(2,2,0),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # self.l3=nn.Sequential(
        #     self.init_(nn.Linear(hidden_size,2*hidden_size),self.gain),
        #     nn.MaxPool1d(2,2,0),
        #     nn.LayerNorm(hidden_size),
        #     nn.ReLU()
        # )

        # 策略网络输出均值和标准差
        self.mu_head = self.init_(nn.Linear(hidden_size, self.action_size),gain=0.01)
        
        self.log_std = nn.Parameter(torch.zeros(self.action_size)) # 对数标准差
        
    def init_(self,layer,gain):
        init_weights(layer,gain)
        return layer

    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        # x = self.l3(x)

        mu = self.mu_head(x)  # 均值
        std = torch.exp(self.log_std)  # 标准差
        
        return mu, std
    
class Net_Critic(nn.Module):
    def __init__(self,obs_size,n_agents,frame_stack=4):
        hidden_size = 256
        input_dim = n_agents*obs_size*frame_stack
        self.gain=init.calculate_gain('relu')
        
        super().__init__()
        self.l1 = nn.Sequential(
            self.init_(nn.Linear(input_dim,hidden_size//2+1),self.gain),
            nn.LayerNorm(hidden_size//2+1),
            nn.ReLU(),
            self.init_(nn.Linear(hidden_size//2+1,hidden_size),self.gain),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.init_(nn.Linear(hidden_size,2*hidden_size),self.gain),
            nn.MaxPool1d(2,2,0),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        # self.l3 = nn.Sequential(
        #     self.init_(nn.Linear(hidden_size,2*hidden_size),self.gain),
        #     nn.MaxPool1d(2,2,0),
        #     nn.LayerNorm(hidden_size),
        #     nn.ReLU()
        # )

        # 价值网络
        self.value_head = self.init_(nn.Linear(hidden_size, 1),gain=1)

    def init_(self,layer,gain):
        init_weights(layer,gain)
        return layer
        
    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        # x = self.l3(x)
        value = self.value_head(x)  # 价值估计
        
        return value
    

class Net_Policy_MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = 256,frame_stack=4):
        super().__init__()
        
        self.output_dim = output_dim
        input_dim = input_dim*frame_stack
        self.gain=init.calculate_gain('relu')
        self.l0 = nn.LayerNorm(input_dim)
        self.l1 = nn.Sequential(
            self.init_(nn.Linear(input_dim,hidden_size),self.gain),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.l2 = nn.Sequential(
            self.init_(nn.Linear(hidden_size,hidden_size),self.gain),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.mu_head = self.init_(nn.Linear(hidden_size, self.output_dim),gain=0.01)
        
        self.log_std = nn.Parameter(torch.zeros(self.output_dim)) # 对数标准差
    

    def init_(self,layer,gain):
        init_weights(layer,gain)
        return layer

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)

        mu = self.mu_head(x)  # 均值
        std = torch.exp(self.log_std)  # 标准差
        
        return mu, std

class Net_Critic_MLP(nn.Module):
    def __init__(self,input_dim,hidden_size = 256,frame_stack=4):
        super().__init__()
        
        input_dim = input_dim*frame_stack
        self.gain=init.calculate_gain('relu')
        self.l0 = nn.LayerNorm(input_dim)
        self.layer = nn.Sequential(
            self.init_(nn.Linear(input_dim,hidden_size),self.gain),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            self.init_(nn.Linear(hidden_size,hidden_size),self.gain),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.value_head = self.init_(nn.Linear(hidden_size, 1),1)


    def init_(self,layer,gain):
        init_weights(layer,gain)
        return layer
        
    def forward(self, x):
        x = self.l0(x)
        x = self.layer(x)
        value = self.value_head(x)  # 价值估计
        
        return value


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean
