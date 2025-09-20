"""
Neural network implementation for Alpha Zero Nonaga.
Uses a ResNet architecture with separate policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block for the neural network."""
    
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class NonagaNetwork(nn.Module):
    """Neural network for Nonaga Alpha Zero."""
    
    def __init__(self, num_channels: int = 256, num_residual_blocks: int = 10):
        super().__init__()
        
        # Input: 9x9x3 (flattened 3x3x3x3x3 board representation)
        # We'll reshape the 3x3x3x3x3 input to 9x9x3 for conv layers
        self.input_channels = 3
        self.board_size = 9  # 9x9 representation of the board
        
        # Initial convolution
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.conv_policy = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * self.board_size * self.board_size, 81)  # 81 possible actions
        
        # Value head
        self.conv_value = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * self.board_size * self.board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 3, 3, 3, 3)
            
        Returns:
            policy: Policy probabilities of shape (batch_size, 81)
            value: Value prediction of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Reshape input from (batch, 3, 3, 3, 3, 3) to (batch, 3, 9, 9)
        x = self._reshape_input(x)
        
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
            
        # Policy head
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(batch_size, -1)
        policy = self.fc_policy(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(batch_size, -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        
        return policy, value
        
    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input from (batch, 3, 3, 3, 3, 3) to (batch, 3, 9, 9).
        Maps the 3x3 grid of 3x3 subboards to a single 9x9 board.
        """
        batch_size = x.shape[0]
        
        # x shape: (batch, 3, 3, 3, 3, 3)
        # We want: (batch, 3, 9, 9)
        
        # Rearrange to group subboards properly
        x = x.permute(0, 5, 1, 3, 2, 4)  # (batch, 3, 3, 3, 3, 3)
        x = x.contiguous().view(batch_size, 3, 9, 9)
        
        return x
        
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single board state.
        
        Args:
            board_state: Board state of shape (3, 3, 3, 3, 3)
            
        Returns:
            policy: Policy probabilities of shape (81,)
            value: Value prediction (scalar)
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            x = torch.FloatTensor(board_state).unsqueeze(0)
            
            # Forward pass
            log_policy, value = self.forward(x)
            policy = torch.exp(log_policy).squeeze(0).numpy()
            value = value.squeeze(0).item()
            
        return policy, value


class AlphaZeroLoss(nn.Module):
    """Combined loss function for Alpha Zero training."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, log_policy: torch.Tensor, value: torch.Tensor, 
                target_policy: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Alpha Zero loss.
        
        Args:
            log_policy: Predicted log policy (batch_size, 81)
            value: Predicted value (batch_size, 1)
            target_policy: Target policy (batch_size, 81)
            target_value: Target value (batch_size, 1)
            
        Returns:
            Combined loss
        """
        # Policy loss (cross-entropy)
        policy_loss = -torch.sum(target_policy * log_policy, dim=1)
        policy_loss = torch.mean(policy_loss)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value.squeeze(), target_value.squeeze())
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        return total_loss


def create_network(config) -> NonagaNetwork:
    """Create a neural network with the given configuration."""
    return NonagaNetwork(
        num_channels=config.network.num_channels,
        num_residual_blocks=config.network.num_residual_blocks
    )


def save_checkpoint(network: NonagaNetwork, optimizer: torch.optim.Optimizer, 
                   iteration: int, filepath: str):
    """Save model checkpoint."""
    torch.save({
        'iteration': iteration,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(filepath: str, network: NonagaNetwork, 
                   optimizer: torch.optim.Optimizer = None) -> int:
    """Load model checkpoint. Returns the iteration number."""
    checkpoint = torch.load(filepath, map_location='cpu')
    network.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint['iteration']