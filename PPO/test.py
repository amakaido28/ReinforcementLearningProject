import gym
import torch 
import numpy as np
from network import Network, ActorCritic
from utils import test 

N_TRIALS    = 25
PRINT_EVERY = 10

def evaluate():
    env             = gym.make('LunarLander-v2', render_mode='human')
    test_rewards    = []

    actor           = Network(env.observation_space.shape[0], env.action_space.n)
    critic          = Network(env.observation_space.shape[0], 1)
    policy          = ActorCritic(actor, critic)

    policy.load_state_dict(torch.load('./weights/LunarLander_0.001.pth'))

    for episode in range(1, 11):    
        test_reward = test(env, policy)
    
        test_rewards.append(test_reward)
    
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    
        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:7.1f}')


if __name__ == '__main__':
    evaluate()
