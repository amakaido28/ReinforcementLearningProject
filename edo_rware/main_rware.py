import rware 
import numpy as np
from actor_critic import Agent
import matplotlib.pyplot as plt
from PIL import Image

def plot_learning_curve(x, score_history, figure_file):
    title   = 'Learning Curve'
    xlabel  = 'Number of games'
    ylabel  = 'Score'
    
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, score_history, label="Training score")
    plt.legend(loc="best")
    plt.savefig(figure_file)
    plt.show()



if __name__ == '__main__':

    layout = """
    x.....x
    ...x...
    ...x...
    x..x..x
    ...x...
    ...x...
    .g...g.
    """
    n_agents        = 4
    n_games         = 150
    filename        = 'cartpole_rware.png'
    figure_file     = 'plots/' + filename
    score_history   = []
    frames          = []
    load_checkpoint = False

    env             = rware.Warehouse(9, 1, 1, n_agents, 1, 0, 6, 800, 1000, rware.RewardType.GLOBAL, layout=layout)
    agents          = [Agent(alpha=1e-4, n_actions=5) for _ in range(n_agents)]
    best_score      = env.reward_range[0]

    # if load_checkpoint:
    #     for agent in agents:
    #         agent.load_models()

    for i in range(n_games):
        observations    = env.reset()
        dones           = [False for _ in range(n_agents)]
        scores          = [0.0 for _ in range(n_agents)]
        score           = 0
        actions         = []
        
        while not any(dones):
            env.render("human")
            for agent, observation in zip(agents, observations):
                action = agent.choose_action(observation)
                actions.append(action)
            
            observations_, rewards, dones, info = env.step(actions)
            scores  = np.add(scores, rewards)
            score   = sum(scores)
            
            if not load_checkpoint:
                for agent, observation, reward, observation_, done in zip(agents, observations, rewards, observations_, dones):
                    agent.learn(observation, reward, observation_, done)
            
            observations = observations_
            actions.clear()
        
            score_history.append(score)
            avg_score = np.mean(score_history[-600:])

            if avg_score > best_score:
                best_score = avg_score
                # if not load_checkpoint:
                #     agent.save_models()

        # if load_checkpoint:
        #     if i % (n_games // 50) == 0:
        #         frame = env.render("rgb_array")
        #         frames.append(Image.fromarray(frame))

        print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    # if load_checkpoint:
    #     frames[0].save('plots/cartpole_rware.gif', save_all=True, append_images=frames[1:], loop=0, duration=1)

    env.close()