import torch
import vmas
import torch.nn.functional as F
from nn_utils import *
from Actors import *
from Buffers import Stack_Manager
import numpy as np
from timeit import default_timer as T

from matplotlib import pyplot as plt
import datetime

from config import *

import sys,getopt

def get_distance(state):
    return ((state[0][0,8]**2+state[0][0,9]**2)**0.5).item()

class Lr_Scheduler:
    def __init__(self,max_epoches,*args):
        self.optimizer = [arg for arg in args]
        self.max_epoches = max_epoches
        self.init_lr = [param_group['lr'] for optimizer in self.optimizer for param_group in optimizer.param_groups]
        self.group_num = len(self.optimizer[0].param_groups)
        
    def step(self,epoch):
        for i,optimizer in enumerate(self.optimizer):
            for j,param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = self.init_lr[i*self.group_num+j]*(1-epoch/self.max_epoches)
                # print(param_group['lr'])

epoch = 0

def main_train(actor, env):
    global random_epsilon,epoch
    
    LR = Lr_Scheduler(max_epoches,*actor.get_optims())

    game_round = 0
    round_reward_list = []
    distance_list = []
    total_timesteps = 0
    while epoch < max_epoches:
        state = env.reset()
        #帧堆叠管理器
        Obs_M = Stack_Manager(state,frame_stack)
        state = Obs_M.get()
        init_distance = min_distance = get_distance(state)
        state = actor.convert_state(state)
        done = False
        rewards = []
        # 收集数据
        ti = T()
        # env.render()
        current_step = 0
        while not done:
            with torch.no_grad():
                action, log_prob = actor.select_action(state,device,random_epsilon)
                value = actor.critic_forward(state)

            action_real = [act.clip(-1.0,1.0) for act in action]
            next_state, reward, done, _ = env.step(action_real)


            current_step += 1

            rewards.append((sum(reward)/len(reward)).item())
            actor.add(current_step,state,action,reward,log_prob,value,done.item())

            Obs_M.add(next_state)
            next_state=Obs_M.get()
            next_dis = get_distance(next_state)
            min_distance = min(min_distance,next_dis)
            next_state = actor.convert_state(next_state)

            # print(f'{reward.item():.2f}',end=',')
            # print(f'{value.item()}')
            state = next_state
            total_timesteps += 1

            if actor.get_rep_len() == rep_capacity:
                print(f'epoch:{epoch} ',end=',')
                actor.epoch_train(state,epoch,LR)
                epoch+=1
                
            # if T()-ti > 1/4:
                # ti = T()
                # env.render()
        
        distance_list.append(min_distance/init_distance)
        game_round+=1
        if random_epsilon > 0.01:
            random_epsilon *= 0.9

        round_reward_list.append(sum(rewards))
        # for i in range(len(rewards)):/
            # print(f'{rewards[i]}')
        print(f'{game_round}th round, reward:{round_reward_list[-1]}')
        
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.set_xlabel('rounds')
    ax.set_ylabel('rewards')
    ax.plot(round_reward_list)

    bx = fig.add_subplot(122)
    bx.set_xlabel('rounds')
    bx.set_ylabel('min distance')
    bx.plot(distance_list)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"./output/plot_{current_time}.png")
    # plt.show()


# 主函数运行
if __name__ == '__main__':
    seed = None
    if True:
        # seed
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True

    env = vmas.make_env(
        scenario="balance", # can be scenario name or BaseScenario class
        num_envs=1,
        seed=seed,
        device=device, # Or "cuda" for GPU
        max_steps=max_timesteps, # Maximum number of steps per episode
        continuous_actions=True,
        n_agents = n_agents,
        random_package_pos_on_line = False
    )


    # 批处理接口
    type = None
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args,"n:",["type=",])
    except getopt.GetoptError:
        print('train.py -n [1,2,3]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-n", "--ttpe"):
            type = arg
    assert(type!=None)

    if type=='1':
        #1
        actor = Centerize_Actor_Critic(lr,n_agents,env)
        torch.cuda.empty_cache()
        main_train(actor, env)
        print(1)
    elif type=='2':
        #2
        actor = Decenterilzed_Actor_Critic(lr,n_agents,env,use_central_critic=True)
        torch.cuda.empty_cache()
        main_train(actor, env)
        print(2)
    elif type=='3':
        #3
        actor = Decenterilzed_Actor_Critic(lr,n_agents,env,use_central_critic=False)
        torch.cuda.empty_cache()
        main_train(actor, env)
        print(3)