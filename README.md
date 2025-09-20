# Alpha Zero for Nonaga

A complete implementation of DeepMind's Alpha Zero algorithm for Nonaga (Ultimate Tic-Tac-Toe).

## What is Nonaga?

Nonaga, also known as Ultimate Tic-Tac-Toe, is a strategic board game played on a 3×3 grid of 3×3 tic-tac-toe boards. Players must play in the subboard corresponding to the last move's position within its subboard, adding a layer of strategic depth to the classic game.

## What is Alpha Zero?

Alpha Zero is a reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks. It learns to play games through self-play without any human knowledge beyond the game rules.

## Features

- **Complete Nonaga Implementation**: Full game logic with move validation and win detection
- **Neural Network**: ResNet-based architecture with policy and value heads
- **Monte Carlo Tree Search**: UCB-based tree search with neural network guidance
- **Self-Play Training**: Automated data generation through AI vs AI games
- **Human vs AI**: Play against the trained AI
- **Configurable**: Easily adjustable training parameters

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Demo
```bash
python demo.py
```

### Train a Model
```bash
python train.py --iterations 100 --self-play-games 50
```

### Play Against AI
```bash
python train.py --load-model models/final_model.pth --play-human
```

## Project Structure

- `nonaga.py` - Game logic and board representation
- `neural_network.py` - ResNet-based neural network architecture
- `mcts.py` - Monte Carlo Tree Search implementation
- `alpha_zero.py` - Main Alpha Zero algorithm and training loop
- `train.py` - Training script with command-line options
- `demo.py` - Interactive demo and examples
- `config.py` - Configuration settings
- `test_*.py` - Unit tests for components

## Training Parameters

Key training parameters can be adjusted in `config.py`:

- **MCTS simulations**: Number of tree search simulations per move
- **Self-play games**: Games generated per training iteration
- **Network architecture**: Channels and residual blocks
- **Training epochs**: Neural network training epochs per iteration

## Algorithm Overview

1. **Self-Play**: AI plays games against itself using MCTS + neural network
2. **Data Collection**: Store (board_state, move_probabilities, game_outcome) tuples
3. **Neural Network Training**: Train on collected data to predict move probabilities and game values
4. **Iteration**: Repeat with improved neural network

## Game Rules

- 3×3 grid of 3×3 subboards (81 total positions)
- Players alternate placing X and O
- Must play in the subboard corresponding to opponent's last move position
- If that subboard is won/full, can play anywhere
- Win by getting 3 subboards in a row (like tic-tac-toe with subboards)

## Testing

Run all tests:
```bash
python test_nonaga.py      # Test game logic
python test_alpha_zero.py  # Test AI components
```

## Implementation Details

- **Board Representation**: 5D tensor (3×3×3×3×3) for efficient neural network processing
- **Action Space**: 81 possible moves (flattened board positions)
- **Neural Network Input**: Canonical board from current player's perspective
- **MCTS Integration**: Neural network provides prior probabilities and leaf evaluation
- **Training Data**: Experience replay buffer with game outcomes as supervision

This implementation follows the Alpha Zero paper methodology while being adapted for the Nonaga game domain.