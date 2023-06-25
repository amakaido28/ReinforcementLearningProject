import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)
    
class Critic(nn.Module):
    def __init__(self,input):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.model(x)