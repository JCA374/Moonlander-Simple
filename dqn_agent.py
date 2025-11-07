"""
Deep Q-Network (DQN) Agent implementation.

This module implements a DQN agent with:
- Dueling network architecture (value and advantage streams)
- Double DQN for action selection
- Soft target network updates
- Experience replay buffer
- Epsilon-greedy exploration
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple
from pathlib import Path

from config import DQNConfig


class DQN(nn.Module):
    """
    Dueling Deep Q-Network architecture.

    Separates state value V(s) and action advantages A(s,a) to improve
    learning stability and performance.

    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        hidden_size: Number of neurons in hidden layers
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()

        # Shared feature layers with LayerNorm for stability
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Dueling streams
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: State tensor of shape (batch_size, state_size)

        Returns:
            Q-values tensor of shape (batch_size, action_size)
        """
        # Shared features with normalization
        x = torch.relu(self.ln1(self.fc1(x)))
        features = torch.relu(self.ln2(self.fc2(x)))

        # Dueling heads
        value = self.value_stream(features)  # shape: [B, 1]
        advantage = self.advantage_stream(features)  # shape: [B, A]

        # Combine into Q(s,a) using dueling architecture formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """
    DQN Agent with experience replay and target network.

    Features:
    - Dueling DQN architecture
    - Double DQN for reduced overestimation
    - Soft target network updates
    - Epsilon-greedy exploration with decay
    - Experience replay buffer

    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        config: DQN configuration object (optional)
        lr: Learning rate (overrides config if provided)
        target_update: Deprecated, kept for backwards compatibility
        tau: Soft update parameter (overrides config if provided)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[DQNConfig] = None,
        lr: Optional[float] = None,
        target_update: Optional[int] = None,  # Deprecated
        tau: Optional[float] = None
    ):
        # Use provided config or create default
        if config is None:
            config = DQNConfig(state_size=state_size, action_size=action_size)

        self.config = config
        self.state_size = state_size
        self.action_size = action_size

        # Memory buffer
        self.memory = deque(maxlen=config.replay_buffer_size)

        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay

        # Learning parameters
        self.learning_rate = lr if lr is not None else config.learning_rate
        self.tau = tau if tau is not None else config.tau
        self.gamma = config.gamma

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DQN(state_size, action_size, config.hidden_size).to(self.device)
        self.target_network = DQN(state_size, action_size, config.hidden_size).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Initialize target network
        self.update_target_network()

    def update_target_network(self) -> None:
        """
        Soft update of target network parameters.

        Performs: θ_target = τ * θ_local + (1 - τ) * θ_target

        This provides more stable learning compared to hard updates.
        """
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action index
        """
        eps = epsilon if epsilon is not None else self.epsilon

        # Exploration: random action
        if np.random.random() <= eps:
            return random.randrange(self.action_size)

        # Exploitation: best action according to Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(np.argmax(q_values.cpu().data.numpy()))

    def replay(self, batch_size: Optional[int] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Train on a batch of experiences from replay buffer.

        Uses Double DQN: online network selects actions, target network evaluates them.

        Args:
            batch_size: Size of training batch (uses config default if None)

        Returns:
            Tuple of (loss, q_variance) or (None, None) if insufficient samples
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        # Need enough samples to train
        if len(self.memory) < batch_size:
            return None, None

        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)

        # Unpack batch
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: select next actions via online net, evaluate via target net
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()

        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Optional: Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Soft update target network every step
        self.update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return loss and Q-value statistics for logging
        with torch.no_grad():
            q_variance = float(torch.var(current_q_values).item())

        return float(loss.item()), q_variance

    def save(self, filename: str | Path) -> None:
        """
        Save agent state to file.

        Args:
            filename: Path to save file

        Raises:
            IOError: If save fails
        """
        try:
            filename = Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else None
            }, filename)

        except Exception as e:
            raise IOError(f"Failed to save model to {filename}: {e}")

    def load(self, filename: str | Path) -> None:
        """
        Load agent state from file.

        Args:
            filename: Path to load file

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If load fails
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"Model file not found: {filename}")

        try:
            checkpoint = torch.load(filename, map_location=self.device)

            # Handle both old and new save formats
            if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
                # New format with full state
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])

                if 'target_network_state_dict' in checkpoint:
                    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                else:
                    # If target network not saved, copy from q_network
                    self.update_target_network()

                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']

                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                # Old format - just the state dict
                self.q_network.load_state_dict(checkpoint)
                self.update_target_network()

        except Exception as e:
            raise IOError(f"Failed to load model from {filename}: {e}")

    def get_stats(self) -> dict:
        """
        Get current agent statistics.

        Returns:
            Dictionary with agent state information
        """
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'learning_rate': self.learning_rate,
            'tau': self.tau,
            'device': str(self.device)
        }
