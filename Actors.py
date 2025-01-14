import torch
import torch.optim as optim
# from torch.distributions import Normal
from nn_utils import FixedNormal as Normal

from Buffers import ReplayBuffer
from Normalize import ValueNorm
from random import *
from config import *
from ppo_algorithms import compute_adv, ppo_update
import numpy as np
from nn_utils import *

class Centerize_Actor_Critic:
    def __init__(self,lr,n_agents,env):
        self.env=env
        hidden_size = 64
        self.model = Net_Policy_MLP(obs_size*n_agents,action_size*n_agents,hidden_size,frame_stack=frame_stack).to(device)
        self.critic = Net_Critic_MLP(obs_size*n_agents,hidden_size,frame_stack=frame_stack).to(device)
        self.optim_policy = optim.Adam(self.model.parameters(), lr=lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.n_agents=n_agents
        self.action_size=action_size

        self.Rep = ReplayBuffer(rep_capacity)
        self.Norm_V = ValueNorm(input_shape=1,device=device)
    
    def get_rep_len(self):
        return len(self.Rep)

    def get_optims(self):
        return self.optim_policy, self.optim_critic

    def convert_state(self,state):
        return torch.cat(state,dim=-1).float().to(device)

    def select_action(self,state,device,epsilon=None):
        mu, std = self.model(state)
        dist = Normal(mu, std)
        if epsilon is not None and random() < epsilon:
            action = torch.tensor([randint(-1,1) for _ in range(self.action_size)]).float().unsqueeze(0).to(device)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action.squeeze(0))  # 对数概率
        action = [action[:,i:i+self.action_size] for i in range(0,action.shape[1],self.action_size)]
        return action, log_prob
    
    def evaluate(self, state, action):
        mu, std = self.model(state)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return log_prob, entropy, mu, std
    

    def add(self,timestep,state,action,reward,log_prob,value,done):
        # 整理
        reward = reward[0]
        action = torch.cat(action,dim=-1)
        self.Rep.add(state,action,reward,log_prob,value.reshape(-1),done)


    def critic_forward(self,state):
        return self.critic(state)


    def construct(self,state):
        with torch.no_grad():
            next_value = self.critic(state)
        self.states,self.actions,rewards,self.log_probs,self.values,dones = self.Rep.get()
        gae_values = torch.cat((self.values,next_value.squeeze(0)),dim=0)
        if use_valuenoprm:
            gae_values = self.Norm_V.denormalize(gae_values.unsqueeze(-1)).squeeze(-1)

        # 计算优势函数
        dones = dones.float()
        returns = torch.tensor(compute_adv(rewards, gae_values, dones, gamma, lambda_)).to(device)
        advantages = returns - gae_values[:-1]
        mean_advantages = torch.mean(advantages)
        std_advantages = torch.std(advantages)

        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        self.returns, self.advantages = returns, advantages
        self.dones = dones


    def shuffle(self):
        idx = torch.randperm(len(self.states))
        self.states = self.states[idx]
        self.actions = self.actions[idx]
        self.returns = self.returns[idx]
        self.log_probs = self.log_probs[idx]
        self.values = self.values[idx]
        self.advantages = self.advantages[idx]
        self.dones = self.dones[idx]


    def get_sample(self,idx):
        return self.states[idx], self.actions[idx], self.returns[idx], \
            self.log_probs[idx], self.values[idx], self.advantages[idx], \
            self.dones, None


    def epoch_train(self,state,epoch,LR):
    # PPO训练循环
        self.construct(state)
        ploss_list = []
        vloss_list = []
        for _ in range(ppo_epoches*n_agents):
            self.shuffle()
            for i in range(0,len(self.Rep),batch_size):
                sample = self.get_sample(slice(i,i+batch_size))
                p,v=ppo_update(self.evaluate,self.model,self.critic,self.optim_policy,self.optim_critic,sample,self.Norm_V)
            ploss_list.append(p)
            vloss_list.append(v)
        LR.step(epoch)
        # update_clip_epsilon()
        self.clear()
        print(f'avg_policy_loss:{sum(ploss_list)/len(ploss_list)},avg_value_loss:{sum(vloss_list)/len(vloss_list)}')

    def clear(self):
        self.Rep.clear()
        self.returns = None
        self.advantages = None
        self.states = None
        self.actions = None
        self.log_probs = None
        self.values = None

    def prep_eval(self):
        self.model.eval()
        self.critic.eval()

    def prep_train(self):
        self.model.train()
        self.critic.train()



class Decenterilzed_Actor_Critic:
    def __init__(self,lr,n_agents,env,use_central_critic=True):
        self.env=env
        hidden_size = 64
        multi_times = n_agents if use_central_critic else 1
        self.model = [Net_Policy_MLP(obs_size,action_size,hidden_size,frame_stack).to(device) for _ in range(n_agents)]
        self.critic = [Net_Critic_MLP(obs_size*multi_times,hidden_size,frame_stack).to(device) for _ in range(n_agents)]
        self.optim_policy = [optim.Adam(self.model[i].parameters(), lr=lr,eps=1e-5,weight_decay=0) for i in range(n_agents)]
        self.optim_critic = [optim.Adam(self.critic[i].parameters(), lr=lr,eps=1e-5,weight_decay=0) for i in range(n_agents)]

        self.n_agents=n_agents
        self.action_size=action_size

        self.Rep = [ReplayBuffer(rep_capacity) for _ in range(n_agents)]
        self.Norm_V = [ValueNorm(input_shape=1,device=device) for _ in range(n_agents)]
        
        self.use_central_critic = use_central_critic
        if use_central_critic:
            self.shared_obs_buffer = ReplayBuffer(rep_capacity)
            self.shared_obs = None
    
    def get_rep_len(self):
        return len(self.Rep[0])

    def get_optims(self):
        return tuple(self.optim_policy+self.optim_critic)

    def convert_state(self,state):
        if self.use_central_critic:
            self.shared_obs = torch.cat(state,dim=-1)
        return state


    def select_action(self,state,device,epsilon=None):
        action,log_prob = [],[]
        for i in range(self.n_agents):
            obs = state[i]
            mu, std = self.model[i](obs)
            dist = Normal(mu, std)
            if epsilon is not None and random() < epsilon:
                action_ = torch.tensor([randint(-1,1) for _ in range(self.action_size)]).float().unsqueeze(0).to(device)
            else:
                action_ = dist.sample()
            log_prob_ = dist.log_prob(action_.squeeze(0))
            action.append(action_)
            log_prob.append(log_prob_)
        return action, log_prob
    

    def evaluate(self, state, action,agent_idx):
        mu, std = self.model[agent_idx](state)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return log_prob, entropy, mu, std
    

    def add(self,timestep,state,action,reward,log_prob,value,done):
        if self.use_central_critic:
            self.shared_obs_buffer.add(self.shared_obs)

        for i in range(self.n_agents):
            self.Rep[i].add(state[i],action[i],reward[i],log_prob[i],value[i].reshape(-1),done)


    def critic_forward(self,state):
        if not self.use_central_critic:
            return [self.critic[i](state[i]) for i in range(self.n_agents)]
        else:
            return [self.critic[i](self.shared_obs) for i in range(self.n_agents)]


    def construct(self,state):
        self.returns = []
        self.advantages = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        for i in range(self.n_agents):
            with torch.no_grad():
                next_value = self.critic[i](self.shared_obs if self.use_central_critic else state[i])
            states,actions,rewards,log_probs,values,dones = self.Rep[i].get()
            gae_values = torch.cat((values,next_value.squeeze(0)),dim=0)
            if use_valuenoprm:
                gae_values = self.Norm_V[i].denormalize(gae_values.unsqueeze(-1)).squeeze(-1)
            # 计算优势函数
            dones = dones.float()
            returns = torch.tensor(compute_adv(rewards, gae_values, dones, gamma, lambda_)).to(device)
            advantages = returns - gae_values[:-1]
            mean_advantages = torch.mean(advantages)
            std_advantages = torch.std(advantages)

            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            self.returns.append(returns)
            self.advantages.append(advantages)
            self.states.append(states)
            self.actions.append(actions)
            self.log_probs.append(log_probs)
            self.values.append(values)
        self.dones = dones

        if self.use_central_critic:
            self.shared_obs_list = self.shared_obs_buffer.get()[0]
    
    def shuffle(self):
        idx = torch.randperm(len(self.states_))
        self.states_ = self.states_[idx]
        self.actions_ = self.actions_[idx]
        self.returns_ = self.returns_[idx]
        self.log_probs_ = self.log_probs_[idx]
        self.values_ = self.values_[idx]
        self.advantages_ = self.advantages_[idx]
        if self.use_central_critic:
            self.shared_obs_list_cpy = self.shared_obs_list_cpy[idx]

    def load(self,a_id):
        self.states_ = self.states[a_id].clone()
        self.actions_ = self.actions[a_id].clone()
        self.returns_ = self.returns[a_id].clone()
        self.log_probs_ = self.log_probs[a_id].clone()
        self.values_ = self.values[a_id].clone()
        self.advantages_ = self.advantages[a_id].clone()
        self.dones_ = self.dones.clone().float()
        if self.use_central_critic:
            self.shared_obs_list_cpy = (self.shared_obs_list).clone()

    def get_sample(self,idx):
        return self.states_[idx], \
            self.actions_[idx], self.returns_[idx], \
            self.log_probs_[idx], self.values_[idx], self.advantages_[idx], \
            self.dones_[idx], \
            self.shared_obs_list_cpy[idx] if self.use_central_critic else None

    def epoch_train(self,state,epoch,LR):
        # PPO训练循环
        self.construct(state)
        for a_id in range(self.n_agents):
            ploss_list = []
            vloss_list = []
            self.load(a_id)
            for _ in range(ppo_epoches):
                self.prep_train(a_id)
                self.shuffle()
                for i in range(0,len(self.Rep),batch_size):
                    sample = self.get_sample(slice(i,i+batch_size))
                    p,v=ppo_update(lambda x,y: self.evaluate(x,y,a_id), self.model[a_id], \
                                   self.critic[a_id],self.optim_policy[a_id],self.optim_critic[a_id],sample,self.Norm_V[a_id])
                ploss_list.append(p)
                vloss_list.append(v)
            print(f'agent{a_id}, avg_policy_loss:{sum(ploss_list)/len(ploss_list)},avg_value_loss:{sum(vloss_list)/len(vloss_list)}')
        LR.step(epoch)
        # update_clip_epsilon()
        self.clear()
        self.prep_eval()

        
    def clear(self):
        for i in range(self.n_agents): 
            self.Rep[i].clear()
        self.returns = None
        self.advantages = None
        self.states = None
        self.actions = None
        self.log_probs = None
        self.values = None
        self.dones = None
        if self.use_central_critic: 
            self.shared_obs_buffer.clear()
            self.shared_obs_list = None


    def prep_train(self,a_id):
        self.model[a_id].train()
        self.critic[a_id].train()

    def prep_eval(self):
        for net in self.model:
            net.eval()
        for net in self.critic:
            net.eval()
        

    
        