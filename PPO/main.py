import numpy as np
import torch
import torch.optim as optim
import gym 
from utils import train, test, init_weights, plot_image
from network import Network, ActorCritic

MAX_EPISODES        = 1_500
DISCOUNT_FACTOR     = 0.99
N_TRIALS            = 25
REWARD_THRESHOLD    = 200
PRINT_EVERY         = 10
PPO_STEPS           = 5
PPO_CLIP            = 0.2

def main():
    train_rewards   = []
    test_rewards    = []

    train_env       = gym.make('LunarLander-v2')
    test_env        = gym.make('LunarLander-v2')

    actor           = Network(train_env.observation_space.shape[0], train_env.action_space.n)
    critic          = Network(train_env.observation_space.shape[0], 1)
    policy          = ActorCritic(actor, critic)
    policy.apply(init_weights)
    optimizer       = optim.Adam(policy.parameters(), lr=0.001)

    for episode in range(1, MAX_EPISODES+1):
    
        train_reward    = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
        test_reward     = test(test_env, policy)
    
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
    
        mean_train_rewards  = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards   = np.mean(test_rewards[-N_TRIALS:])
    
        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
    
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            torch.save(policy.state_dict(), './weights/LunarLander_{}_2.pth'.format(0.001))
            break

    plot_image(test_rewards, train_rewards, REWARD_THRESHOLD, './learning_curve.png')



if __name__ == '__main__':
    main()