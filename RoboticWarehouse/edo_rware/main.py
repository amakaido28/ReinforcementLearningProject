import gym 
import numpy as np
from actor_critic import Agent
import matplotlib.pyplot as plt
from PIL import Image


def plot_learning_curve(x, score_history, figure_file):
    title = 'Learning Curve'
    xlabel = 'Number of games'
    ylabel = 'Score'
    
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, score_history, label="Training score")
    plt.legend(loc="best")
    plt.savefig(figure_file)
    plt.show()



if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800

    filename = 'cartpole.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    frames = []
    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward 
            
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            
            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        if load_checkpoint:
            if i % (n_games // 50) == 0:
                frame = env.render()
                frames.append(Image.fromarray(frame))

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    if load_checkpoint:
        frames[0].save('plots/cartpole.gif', save_all=True, append_images=frames[1:], loop=0, duration=1)