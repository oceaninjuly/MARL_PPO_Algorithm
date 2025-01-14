import torch
import torch.nn as nn
from config import clip_epsilon, std_penalty_coef, mean_penalty_coef,rep_capacity, max_grad_norm


def huber_loss(e, d=10.0):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mean_penalty_loss(mu):
    return torch.std(mu,dim=-1).mean()


# 计算优势函数 (GAE)
def compute_gae(rewards, values, dones, gamma, lambda_):
    gae = 0
    returns = []
    for step in reversed(range(rep_capacity)):
        delta = rewards[step] + gamma * values[step+1]*(1 - dones[step]) - values[step]
        gae = delta + gamma * lambda_ * (1 - dones[step]) * gae
        returns.insert(0, gae)
        # print(f'{gae:.2f}',end=',')
    # print()
    return returns


def ppo_update(evaluate,policy,critic,optim_policy,optim_critic,sample,Norm_V):
    batch_state, batch_actions, batch_returns, batch_log_probs, batch_values, batch_advantages, \
        batch_masks,batch_shared_obs = sample
    # 计算Actor损失
    new_log_probs, entropy, mu_pred , std_pred = evaluate(batch_state, batch_actions)
    if batch_shared_obs is None:
        state_values = critic(batch_state)
    else:
        state_values = critic(batch_shared_obs)
        
    ratio = torch.exp(new_log_probs - batch_log_probs)
    batch_advantages = batch_advantages.unsqueeze(-1)
    surrogate_loss = ratio * batch_advantages
    clipped_loss = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
    # print(ratio)

    policy_loss = -torch.sum(torch.min(surrogate_loss, clipped_loss),
                             dim=-1,keepdim=True).mean() - 0.01 * entropy
    #    std_penalty_coef * std_pred.norm() + mean_penalty_coef * mean_penalty_loss(mu_pred)
    
    # print('loss:',loss.item())
    
    optim_policy.zero_grad()
    # with torch.autograd.detect_anomaly():
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),max_grad_norm)
    optim_policy.step()

    # 计算Critic损失
    Norm_V.update(batch_returns)
    batch_returns = Norm_V.normalize(batch_returns)
    value_pred_clipped = batch_values + (state_values.squeeze(-1) - batch_values).clamp(-clip_epsilon, clip_epsilon)
    value_pred_original = state_values.squeeze(-1)
    value_loss_clipped = huber_loss(batch_returns - value_pred_clipped)
    value_loss_original = huber_loss(batch_returns - value_pred_original)
    value_loss = torch.max(value_loss_clipped, value_loss_original).mean()
    
    optim_critic.zero_grad()
    # with torch.autograd.detect_anomaly():
    value_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    optim_critic.step()

    return policy_loss.item(), value_loss.item()