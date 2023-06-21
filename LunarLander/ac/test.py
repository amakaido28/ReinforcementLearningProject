from utils import choose_action
from network import PolicyNetwork, StateValueNetwork
import torch
import gym


def test(n_episodes=5, name_actor='LunarLander_0.001_actor.pth', name_critic='LunarLander_0.001_critic.pth'):
    env     = gym.make('LunarLander-v2', render_mode='human')
    actor   = PolicyNetwork()
    critic  = StateValueNetwork()
    
    actor.load_state_dict(torch.load('./preTrained/2_net/{}'.format(name_actor)))
    critic.load_state_dict(torch.load('./preTrained/2_net/{}'.format(name_critic)))

    for i_episode in range(1, n_episodes+1):
        state, _        = env.reset()
        running_reward  = 0

        for _ in range(10000):
            action, _ = choose_action(actor, state)
            state, reward, done, _, _ = env.step(action)
            running_reward += reward
            
            if done:
                break

        print('Episode: {}\tReward: {}'.format(i_episode, running_reward))

    env.close()

            
if __name__ == '__main__':
    test()