import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as distributions
import matplotlib.pyplot as plt

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def plot_image(test_rewards, train_rewards, threshold, figure_file):
    plt.figure(figsize=(12,8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(figure_file)
    


def calculate_returns(rewards, gamma, normalize=True):
    returns = []
    R       = 0

    for r in reversed(rewards):
        R = r + R * gamma 
        returns.insert(0, R)

    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std() 

    return returns


def calculate_advantages(rewards, values, normalize=True):

    advantages = rewards - values

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def update_loss(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    states              = states.detach()
    actions             = actions.detach()
    log_prob_actions    = log_prob_actions.detach()
    advantages          = advantages.detach()
    returns             = returns.detach()

    for _ in range(ppo_steps):

        actions_pred, state_values  = policy(states)

        state_values        = state_values.squeeze(-1)

        action_probs        = F.softmax(actions_pred, dim=-1)
        action_distribution = distributions.Categorical(action_probs)

        log_prob_actions_   = action_distribution.log_prob(actions)  

        policy_ratio        = (log_prob_actions_ - log_prob_actions).exp()

        actor_loss_1        = policy_ratio * advantages
        actor_loss_2        = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages

        actor_loss          = - torch.min(actor_loss_1, actor_loss_2).mean() 
        critic_loss         = F.smooth_l1_loss(returns, state_values).mean()

        optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        optimizer.step()


def train(env, policy, optimizer, gamma, ppo_steps, ppo_clip):

    policy.train()

    states              = []
    actions             = []
    log_prob_actions    = []
    values              = []
    rewards             = []
    done                = False
    episode_reward      = 0

    state, _ = env.reset()

    for _ in range(1000):

        state = torch.tensor(state).unsqueeze(0).float()

        states.append(state)

        action_pred, state_value = policy(state)

        action_probs        = F.softmax(action_pred, dim=-1)
        action_distribution = distributions.Categorical(action_probs)

        action = action_distribution.sample()
        log_prob_action = action_distribution.log_prob(action)

        state, reward, done, _, _ = env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(state_value)
        rewards.append(reward)

        episode_reward += reward

        if done:
            break 

    states              = torch.cat(states)
    actions             = torch.cat(actions)
    values              = torch.cat(values).squeeze(-1) 
    log_prob_actions    = torch.cat(log_prob_actions)

    returns     = calculate_returns(rewards, gamma)
    advantages  = calculate_advantages(returns, values)

    update_loss(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return episode_reward


def test(env, policy):

    policy.eval()

    rewards         = []
    done            = False 
    episode_reward  = 0

    state, _ = env.reset()

    for _ in range(1000):

        state = torch.tensor(state).unsqueeze(0).float()

        with torch.no_grad():
            action_pred, _ = policy(state)

            action_probs = F.softmax(action_pred, dim=-1)

        action = torch.argmax(action_probs, dim=-1)

        state, reward, done, _, _ = env.step(action.item())

        episode_reward += reward

        if done:
            break 

    return episode_reward