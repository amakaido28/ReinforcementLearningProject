from utils import calculateLoss, choose_action, plot_learning_curve
from network import PolicyNetwork, StateValueNetwork
import torch
import torch.optim as optim
import gym
import numpy as np 
from collections import deque


def train():
    gamma               = 0.99
    lr                  = 0.001
    
    env                 = gym.make('LunarLander-v2', render_mode='human')
    
    actor               = PolicyNetwork()
    critic              = StateValueNetwork()
    policy_optimizer    = optim.Adam(actor.parameters(), lr=lr)
    stateval_optimizer  = optim.Adam(critic.parameters(), lr=lr)
    
    running_reward      = 0
    score_history       = []
    recent_scores       = deque(maxlen = 100)
    figure_file         = './learning_curve_2net_2.png'

    for i_episode in range(0, 10000):
        state, _    = env.reset()
        score       = 0
        
        for t in range(10000):
            action, log_prob = choose_action(actor, state)

            state_, reward, done, _, _  = env.step(action)
            running_reward  += reward
            score           += reward 

            actor_loss, critic_loss = calculateLoss(critic, state, reward, state_, done, log_prob, gamma)
            
            policy_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            policy_optimizer.step()

            stateval_optimizer.zero_grad()
            critic_loss.backward()
            stateval_optimizer.step() 

            if done:
                break

            state = state_

        recent_scores.append(score)

        if i_episode % 20 == 0:
            running_reward = running_reward/20
            score_history.append(running_reward)
            print('Episode {}\tavg_scores: {}'.format(i_episode, np.array(recent_scores).mean()))
            running_reward = 0

        if np.array(recent_scores).mean() >= 195:
            torch.save(actor.state_dict(), './preTrained/2_net/LunarLander_{}_actor_2.pth'.format(lr))
            torch.save(critic.state_dict(), './preTrained/2_net/LunarLander_{}_critic_2.pth'.format(lr))
            print("########## Solved! ##########")
            break
    
    x = [(i+1)*20 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    env.close()
            
if __name__ == '__main__':
    train()