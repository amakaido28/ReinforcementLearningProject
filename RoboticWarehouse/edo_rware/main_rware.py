import rware 
import numpy as np
from actor_critic import Agent
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


N_AGENTS        = 2
N_GAMES         = 8
MAX_STEPS       = 100000


def plot_learning_curve(x, score_history, figure_file):
    title   = 'Learning Curve'
    xlabel  = 'Number of Games'
    ylabel  = 'Score'
    
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, score_history, label="Training Score")
    plt.legend(loc="best")
    plt.savefig(figure_file)
    plt.show()



if __name__ == '__main__':

    layout = """
    ....
    .xx.
    .xx.
    g..g
    """
    
    filename        = 'cartpole_rware.png'
    figure_file     = 'plots/' + filename
    score_history   = []
    frames          = []
    load_checkpoint = False

    env             = rware.Warehouse(9, 1, 1, N_AGENTS, 1, 0, 2, 5, 7, rware.RewardType.INDIVIDUAL, layout=layout)
    agents          = [Agent(alpha=1e-5, n_actions=5) for _ in range(N_AGENTS)]
    best_score      = env.reward_range[0]

    if load_checkpoint:
        for index, agent in enumerate(agents):
            agent.load_models(index)

    for i in range(N_GAMES):
        dones           = [False for _ in range(N_AGENTS)]
        scores          = [0.0 for _ in range(N_AGENTS)]
        actions         = []
        score           = 0
        num_step        = 0
        pbar = tqdm(total=MAX_STEPS, desc=f'Episode {i}')

        observations    = env.reset()
        
        while num_step <= MAX_STEPS:
            num_step += 1
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
        
            if num_step % 10000 == 0:
                #print('Episode', i, 'Step', num_step)
                pbar.update(10000)
                
        score_history.append(score)
        avg_score = np.mean(score_history[-2:])

        if avg_score > best_score:
            best_score = avg_score

            if not load_checkpoint:
                for index, agent in enumerate(agents):
                    agent.save_models(index)    

        print('\n')
        print('Episode', i, 'Score %.1f' % score, 'Avg_score %.1f' % avg_score)
        print('\n')
        pbar.close()

    if not load_checkpoint:
        x = [i+1 for i in range(N_GAMES)]
        plot_learning_curve(x, score_history, figure_file)

    env.close()