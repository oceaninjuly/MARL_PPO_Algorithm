import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_agents = 5
obs_size=16
action_size=2
frame_stack=1

max_epoches=400
max_timesteps = 2000
lr=5e-4
gamma = 0.99
lambda_ = 0.95
clip_epsilon = 0.2
random_epsilon = 0.0
rep_capacity = 200
ppo_epoches= 10
batch_size = 200
std_penalty_coef = -0.05
mean_penalty_coef = 0.05

max_grad_norm = 10.0

def update_clip_epsilon():
    global clip_epsilon
    if clip_epsilon > 0.2:
        clip_epsilon *= 0.9