import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        
        # Shared feature layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Dueling streams
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_size)
        
    def forward(self, x):
        # Shared features
        x = torch.relu(self.fc1(x))
        features = torch.relu(self.fc2(x))
        
        # Dueling streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.0003, target_update=500, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_min = 0.02  # Lower minimum for more greedy behavior
        self.epsilon_decay = 0.995  # Faster decay - reach min in ~1000 episodes
        self.learning_rate = lr
        self.device = torch.device("cpu")
        
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # soft update parameter
        self.tau = tau
        
        self.update_target_network()
        
    def update_target_network(self):
        # Soft update: θ_target = τ * θ_local + (1 - τ) * θ_target
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None, None
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        # Double DQN: select next actions via online net, evaluate via target net
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (0.99 * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network every step
        self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Return loss and Q-value stats for logging
        with torch.no_grad():
            q_variance = torch.var(current_q_values).item()
            
        return loss.item(), q_variance
            
    def save(self, filename):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'epsilon': self.epsilon,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        
        # Handle both old and new save formats
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            # New format
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Old format - just the state dict
            self.q_network.load_state_dict(checkpoint)
            
        self.update_target_network()