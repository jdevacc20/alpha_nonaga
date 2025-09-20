"""
Alpha Zero implementation for Nonaga.
"""

import os
import random
import numpy as np
import torch
import torch.optim as optim
from typing import List, Tuple, Dict
from collections import deque
from tqdm import tqdm

from nonaga import NonagaGame, Player, GameResult
from neural_network import NonagaNetwork, AlphaZeroLoss, create_network, save_checkpoint, load_checkpoint
from mcts import MCTS
from config import default_config


class ExperienceBuffer:
    """Buffer to store self-play experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        
    def add_experience(self, board_state: np.ndarray, policy: np.ndarray, value: float):
        """Add an experience to the buffer."""
        self.buffer.append((board_state, policy, value))
        
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        experiences = random.sample(list(self.buffer), batch_size)
        
        boards = np.array([exp[0] for exp in experiences])
        policies = np.array([exp[1] for exp in experiences])
        values = np.array([exp[2] for exp in experiences])
        
        return boards, policies, values
        
    def size(self) -> int:
        """Get the size of the buffer."""
        return len(self.buffer)
        
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class AlphaZero:
    """Alpha Zero implementation for Nonaga."""
    
    def __init__(self, config=None):
        self.config = config if config is not None else default_config
        
        # Initialize network
        self.network = create_network(self.config)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.network.learning_rate,
            weight_decay=self.config.network.weight_decay
        )
        
        # Initialize MCTS
        self.mcts = MCTS(self.network, self.config)
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Loss function
        self.loss_fn = AlphaZeroLoss()
        
        # Training statistics
        self.iteration = 0
        self.training_history = []
        
    def self_play_game(self, temperature_threshold: int = 30) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play a single self-play game and return training examples.
        
        Args:
            temperature_threshold: Move number after which to use temperature 0
            
        Returns:
            List of (board_state, policy, value) tuples
        """
        game = NonagaGame()
        examples = []
        move_count = 0
        
        while game.result == GameResult.ONGOING:
            # Get current board state from current player's perspective
            canonical_board = game.get_canonical_form(game.current_player)
            
            # Run MCTS to get action probabilities
            temperature = 1.0 if move_count < temperature_threshold else 0.0
            action_probs = self.mcts.get_action_probs(game, temperature)
            
            # Store example (we'll set the value after the game ends)
            examples.append((canonical_board.copy(), action_probs.copy(), game.current_player))
            
            # Select action
            valid_actions = game.get_valid_actions()
            action_probs_valid = action_probs * valid_actions
            if np.sum(action_probs_valid) > 0:
                action_probs_valid = action_probs_valid / np.sum(action_probs_valid)
                action = np.random.choice(len(action_probs_valid), p=action_probs_valid)
            else:
                # Fallback to random valid action
                valid_indices = np.where(valid_actions > 0)[0]
                action = np.random.choice(valid_indices)
                
            # Make move
            move = game.action_to_move(action)
            game.make_move(*move)
            move_count += 1
            
        # Assign values based on game result
        final_examples = []
        for board_state, policy, player in examples:
            if game.result == GameResult.DRAW:
                value = 0.0
            elif game.result == GameResult.X_WINS:
                value = 1.0 if player == Player.X else -1.0
            else:  # O_WINS
                value = 1.0 if player == Player.O else -1.0
                
            final_examples.append((board_state, policy, value))
            
        return final_examples
        
    def generate_self_play_data(self, num_games: int) -> None:
        """Generate self-play data and add to experience buffer."""
        print(f"Generating {num_games} self-play games...")
        
        for i in tqdm(range(num_games)):
            examples = self.self_play_game()
            
            # Add examples to buffer
            for board_state, policy, value in examples:
                self.experience_buffer.add_experience(board_state, policy, value)
                
    def train_network(self, num_epochs: int = None, batch_size: int = None) -> Dict[str, float]:
        """Train the neural network on collected experiences."""
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        if batch_size is None:
            batch_size = self.config.training.batch_size
            
        if self.experience_buffer.size() < batch_size:
            print(f"Not enough experiences ({self.experience_buffer.size()}) for training")
            return {}
            
        print(f"Training network for {num_epochs} epochs...")
        
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Sample batch
            boards, target_policies, target_values = self.experience_buffer.sample_batch(batch_size)
            
            # Convert to tensors
            boards = torch.FloatTensor(boards)
            target_policies = torch.FloatTensor(target_policies)
            target_values = torch.FloatTensor(target_values)
            
            # Forward pass
            log_policies, values = self.network(boards)
            
            # Calculate loss
            loss = self.loss_fn(log_policies, values, target_policies, target_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        training_stats = {
            'loss': avg_loss,
            'buffer_size': self.experience_buffer.size(),
            'num_batches': num_batches
        }
        
        self.training_history.append(training_stats)
        
        return training_stats
        
    def evaluate_networks(self, challenger_network: NonagaNetwork, num_games: int = 40) -> float:
        """
        Evaluate current network against challenger network.
        
        Args:
            challenger_network: Network to evaluate against
            num_games: Number of games to play
            
        Returns:
            Win rate of current network
        """
        print(f"Evaluating networks over {num_games} games...")
        
        current_mcts = MCTS(self.network, self.config)
        challenger_mcts = MCTS(challenger_network, self.config)
        
        wins = 0
        draws = 0
        
        for i in tqdm(range(num_games)):
            game = NonagaGame()
            
            # Alternate who goes first
            if i % 2 == 0:
                # Current network is X, challenger is O
                x_mcts, o_mcts = current_mcts, challenger_mcts
                current_is_x = True
            else:
                # Challenger is X, current network is O
                x_mcts, o_mcts = challenger_mcts, current_mcts
                current_is_x = False
                
            # Play game
            while game.result == GameResult.ONGOING:
                if game.current_player == Player.X:
                    action = x_mcts.select_action(game, temperature=0.0)
                else:
                    action = o_mcts.select_action(game, temperature=0.0)
                    
                move = game.action_to_move(action)
                game.make_move(*move)
                
            # Count result
            if game.result == GameResult.DRAW:
                draws += 1
            elif (game.result == GameResult.X_WINS and current_is_x) or \
                 (game.result == GameResult.O_WINS and not current_is_x):
                wins += 1
                
        win_rate = (wins + 0.5 * draws) / num_games
        print(f"Win rate: {win_rate:.3f} ({wins} wins, {draws} draws, {num_games - wins - draws} losses)")
        
        return win_rate
        
    def train_iteration(self) -> Dict[str, float]:
        """Run a single training iteration."""
        print(f"\n=== Training Iteration {self.iteration + 1} ===")
        
        # Generate self-play data
        self.generate_self_play_data(self.config.training.num_self_play_games)
        
        # Train network
        training_stats = self.train_network()
        
        # Save checkpoint
        if (self.iteration + 1) % self.config.training.checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_iter_{self.iteration + 1}.pth"
            save_checkpoint(self.network, self.optimizer, self.iteration + 1, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
        self.iteration += 1
        
        return training_stats
        
    def train(self, num_iterations: int = None) -> None:
        """Train the Alpha Zero system."""
        if num_iterations is None:
            num_iterations = self.config.training.num_iterations
            
        print(f"Starting Alpha Zero training for {num_iterations} iterations...")
        
        for i in range(num_iterations):
            stats = self.train_iteration()
            
            if stats:
                print(f"Iteration {self.iteration}: Loss = {stats['loss']:.4f}, "
                      f"Buffer size = {stats['buffer_size']}")
                      
        print("Training completed!")
        
    def save_model(self, filepath: str):
        """Save the trained model."""
        save_checkpoint(self.network, self.optimizer, self.iteration, filepath)
        
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.iteration = load_checkpoint(filepath, self.network, self.optimizer)
        
    def play_human_game(self, human_is_x: bool = True) -> None:
        """Play a game against a human player."""
        game = NonagaGame()
        
        print("Starting game against human!")
        print("Positions are numbered 0-80, or enter coordinates as 'big_row,big_col,small_row,small_col'")
        print(f"Human is {'X' if human_is_x else 'O'}")
        print("\nInitial board:")
        print(game)
        
        while game.result == GameResult.ONGOING:
            print(f"\nCurrent player: {game.current_player.name}")
            
            if (game.current_player == Player.X and human_is_x) or \
               (game.current_player == Player.O and not human_is_x):
                # Human turn
                valid_moves = game.get_valid_moves()
                print(f"Valid actions: {[game.move_to_action(*move) for move in valid_moves]}")
                
                while True:
                    try:
                        user_input = input("Enter your move: ").strip()
                        
                        if ',' in user_input:
                            # Coordinate format
                            coords = [int(x) for x in user_input.split(',')]
                            if len(coords) == 4:
                                big_row, big_col, small_row, small_col = coords
                                action = game.move_to_action(big_row, big_col, small_row, small_col)
                            else:
                                print("Invalid format. Use: big_row,big_col,small_row,small_col")
                                continue
                        else:
                            # Action number format
                            action = int(user_input)
                            
                        move = game.action_to_move(action)
                        if game.make_move(*move):
                            break
                        else:
                            print("Invalid move! Try again.")
                    except (ValueError, IndexError):
                        print("Invalid input! Enter a number 0-80 or coordinates.")
            else:
                # AI turn
                print("AI is thinking...")
                action = self.mcts.select_action(game, temperature=0.0)
                move = game.action_to_move(action)
                game.make_move(*move)
                print(f"AI played action {action} at position {move}")
                
            print("\nBoard:")
            print(game)
            
        # Game over
        print(f"\nGame over! Result: {game.result.name}")
        if game.result == GameResult.X_WINS:
            winner = "Human" if human_is_x else "AI"
        elif game.result == GameResult.O_WINS:
            winner = "Human" if not human_is_x else "AI"
        else:
            winner = "Draw"
        print(f"Winner: {winner}")


if __name__ == "__main__":
    # Example usage
    alpha_zero = AlphaZero()
    alpha_zero.train(num_iterations=5)  # Small number for testing