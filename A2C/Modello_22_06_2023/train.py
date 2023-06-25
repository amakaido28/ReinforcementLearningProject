import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from network import Actor, Critic
import numpy as np
from collections import deque
from utils import plot_learning_curve,t

def train():
    lr=2e-3
    gamma=0.99
    mean_threshold=190
    values_for_threashold=100

    figure_file='./learning_curve.png'
    
    view_step=10
    best_score=-400
    mean=0
    ep_rewards = np.array([])

    actor_loss=0
    critic_loss=0
    
    env = gym.make("LunarLander-v2")

    state_num = env.observation_space.shape[0]
    action_num = env.action_space.n

    actor=Actor(state_num,action_num)
    critic=Critic(state_num)

    load_weights=False
    
    if(load_weights):
        #actor load model
        actor = Actor(state_num,action_num)
        adam_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        checkpoint = torch.load('C:\\Users\\kaidoama\\Desktop\\mio\\Reinforcement Learning\\LunarLander\\actor_model.pth')
        actor.load_state_dict(checkpoint['actor_state_dict'])
        adam_actor.load_state_dict(checkpoint['adam_actor_state_dict'])
        j = checkpoint['step']
        actor_loss = checkpoint['loss_actor']
        best_score=checkpoint['best_score']

        #critic load model
        critic = Critic(state_num)
        adam_critic = torch.optim.Adam(critic.parameters(), lr=lr)
        checkpoint = torch.load('C:\\Users\\kaidoama\\Desktop\\mio\\Reinforcement Learning\\LunarLander\\critic_model.pth')
        critic.load_state_dict(checkpoint['critic_state_dict'])
        adam_critic.load_state_dict(checkpoint['adam_critic_state_dict'])
        j = checkpoint['step']
        critic_loss = checkpoint['loss_critic']

        #make other x steps
        total_episodes = j+10000
    else:
        total_episodes = 10000
        
        adam_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        adam_critic = torch.optim.Adam(critic.parameters(), lr=lr)
    
    for i in range(total_episodes):
        done=False
        total_reward=0

        if i%view_step==0:
            env = gym.make("LunarLander-v2",render_mode="human")
        else:
            env = gym.make("LunarLander-v2")

        state,_=env.reset()
        
        for j in range(2000):
            probs=actor(t(state))
            dist=torch.distributions.Categorical(probs=probs)
            action=dist.sample()
            next_state, reward, done, truncated, info = env.step(action.detach().data.numpy())
            #print(reward)

            #calculate advantage
            td_target = reward + gamma * critic(t(next_state)) * (1-done)
            advantage = td_target - critic(t(state))
                
            total_reward += reward

            state = next_state

            #compute loss
            critic_loss = advantage.pow(2).mean()
            adam_critic.zero_grad()
            critic_loss.backward()
            adam_critic.step()

            actor_loss = -dist.log_prob(action)*advantage.detach()
            adam_actor.zero_grad()
            actor_loss.backward()
            adam_actor.step()
    
    
            if done:
                ep_rewards=np.append(ep_rewards,total_reward)
                x=round(np.mean(ep_rewards[-values_for_threashold:]), 3)
                if(x>best_score):
                    best_score=round(np.mean(ep_rewards[-values_for_threashold:]), 3)
                    torch.save(
                        {'step': i,
                        'actor_state_dict': actor.state_dict(),
                        'adam_actor_state_dict': adam_actor.state_dict(),
                        'best_score':best_score,
                        'loss_actor': actor_loss},
	                    'C:\\Users\\kaidoama\\Desktop\\mio\\Reinforcement Learning\\LunarLander\\actor_model.pth'
                    )
                    torch.save(
                        {'step': i,
                        'critic_state_dict': critic.state_dict(),
                        'adam_critic_state_dict': adam_critic.state_dict(),
                        'loss_critic': critic_loss},
	                    'C:\\Users\\kaidoama\\Desktop\\mio\\Reinforcement Learning\\LunarLander\\critic_model.pth'
                    )
                
                if i % 10 == 0:
                    print("episode: {}\treward: {}".format(i, total_reward))
                break

        mean=round(np.mean(ep_rewards[-values_for_threashold:]), 3)
        if mean>=mean_threshold:
            print("!!!PROBLEM RESOLVED!!!")
            break

    x = [(i+1)*20 for i in range(len(ep_rewards))]
    plot_learning_curve(x, ep_rewards_=ep_rewards, figure_file_=figure_file)

    env.close()

if __name__ == '__main__':
    train()