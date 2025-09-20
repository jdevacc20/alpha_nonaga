"""Configuration settings for Alpha Zero Nonaga implementation."""

import dataclasses
from typing import Tuple


@dataclasses.dataclass
class GameConfig:
    """Configuration for Nonaga game."""
    board_size: int = 3  # 3x3 grid of 3x3 subboards
    players: int = 2
    

@dataclasses.dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""
    num_simulations: int = 800
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


@dataclasses.dataclass
class NetworkConfig:
    """Configuration for neural network."""
    num_channels: int = 256
    num_residual_blocks: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    

@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    num_iterations: int = 1000
    num_self_play_games: int = 100
    num_epochs: int = 10
    batch_size: int = 32
    checkpoint_interval: int = 10
    arena_games: int = 40
    arena_threshold: float = 0.55
    

# Default configuration instance
@dataclasses.dataclass
class Config:
    game: GameConfig = dataclasses.field(default_factory=GameConfig)
    mcts: MCTSConfig = dataclasses.field(default_factory=MCTSConfig)
    network: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

default_config = Config()