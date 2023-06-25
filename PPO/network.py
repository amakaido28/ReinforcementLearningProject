import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, dropout = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x 
    

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor  = actor 
        self.critic = critic 

    def forward(self, x):
        action_pred     = self.actor(x)
        state_value     = self.critic(x)

        return action_pred, state_value