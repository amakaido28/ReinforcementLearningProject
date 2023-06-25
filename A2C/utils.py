import matplotlib.pyplot as plt
import torch

def plot_learning_curve(x, ep_rewards_, figure_file_):
    title   = 'Learning Curve'
    xlabel  = 'Number of games'
    ylabel  = 'Score'
    
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, ep_rewards_, label="Training score")
    plt.legend(loc="best")
    plt.savefig(figure_file_)

def t(x): return torch.from_numpy(x).float()
    
