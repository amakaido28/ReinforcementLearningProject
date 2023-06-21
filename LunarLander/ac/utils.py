import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt


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


def choose_action(network, state):
    state = torch.from_numpy(state).float()

    probs = network(state)
    state = state.detach()

    action_distribution = Categorical(probs)
    action = action_distribution.sample()

    return action.item(), action_distribution.log_prob(action) 
    
    
def calculateLoss(network, state, reward, state_, done, log_prob, gamma=0.99):
    state               = torch.from_numpy(state).float()
    state_              = torch.from_numpy(state_).float()
        
    state_value         = network(state)
    state_value_        = network(state_)

    delta               = reward + gamma * state_value_ * (1-int(done)) - state_value 
    actor_loss          = -log_prob * delta
    critic_loss         = F.mse_loss(reward + gamma * state_value_ * (1-int(done)), state_value)   
        
    return actor_loss, critic_loss
