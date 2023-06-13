import torch.nn as nn

class MLP(nn.Module):
    '''This is a class to build a multi-layer perceptron used for the actor & critic networks'''

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        '''The constructor for the MLP
            
        Args:
            input_dim: input size to the network
            hidden_dim: number of neurons to have in the intermediate hidden layers
            output_dim: output size of the network
            dropout: rate of dropout to use during training for regularization
        '''

        super().__init__()
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
            
    def forward(self, x):
        '''Overrides the forward pass of the MLP'''
        return self.net(x)
        
class ActorCritic(nn.Module):
    '''This class is the combines the actor & critic into one agent'''

    def __init__(self, actor, critic):
        '''Constructor for the agent

        Args: 
            actor, critic: MLP objects to be used as the actor and critic'''
        super().__init__()
            
        self.actor = actor
        self.critic = critic
            
    def forward(self, state):
        '''Performs the forward pass over the two models and returns their outputs'''
        action_pred = self.actor(state)
        value_pred = self.critic(state)
            
        return action_pred, value_pred