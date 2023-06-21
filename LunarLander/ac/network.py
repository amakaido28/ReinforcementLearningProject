import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space=8, action_space=4):
        super(PolicyNetwork, self).__init__()
        self.input_layer    = nn.Linear(observation_space, 128)
        self.hidden_layer   = nn.Linear(128, 64)
        self.output_layer   = nn.Linear(64, action_space)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
    
        actions         = self.output_layer(x)
        action_probs    = F.softmax(actions, dim=0)
        
        return action_probs
    

class StateValueNetwork(nn.Module):
    def __init__(self, observation_space=8):
        super(StateValueNetwork, self).__init__()
        self.input_layer    = nn.Linear(observation_space, 128)
        self.hidden_layer   = nn.Linear(128, 64)
        self.output_layer   = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        
        state_value = self.output_layer(x)
        
        return state_value