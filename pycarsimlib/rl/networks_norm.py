import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class PolicyNet(nn.Module):
    """
    Actor 网络
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        
        self.out = nn.Linear(64, action_dim)    
        self.action_bound = action_bound
        
        self.apply(weight_init)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        # 动作输出
        return torch.tanh(self.out(x)) * self.action_bound

class QValueNet(nn.Module):
    """
    Critic 网络
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        # 输入维度 = 6 + 4 = 10
        self.input_dim = state_dim + action_dim
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        
        self.out = nn.Linear(64, 1)
        
        self.apply(weight_init)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        return self.out(x)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q_network = QValueNet(state_dim, action_dim, hidden_dim)
    
    def forward(self, state, action):
        return self.q_network(state, action)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.policy_network = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
    
    def forward(self, state):
        return self.policy_network(state)