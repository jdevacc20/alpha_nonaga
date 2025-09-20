"""Test basic Alpha Zero functionality."""

import torch
import numpy as np
from alpha_zero import AlphaZero
from nonaga import NonagaGame, Player
from config import default_config

def test_network():
    """Test neural network functionality."""
    print("Testing neural network...")
    
    # Create Alpha Zero instance
    alpha_zero = AlphaZero(default_config)
    
    # Test network with random input
    game = NonagaGame()
    canonical_board = game.get_canonical_form(Player.X)
    
    policy, value = alpha_zero.network.predict(canonical_board)
    
    assert policy.shape == (81,), f"Expected policy shape (81,), got {policy.shape}"
    assert np.abs(np.sum(policy) - 1.0) < 1e-5, f"Policy should sum to 1, got {np.sum(policy)}"
    assert -1 <= value <= 1, f"Value should be in [-1, 1], got {value}"
    
    print("✓ Network test passed!")

def test_mcts():
    """Test MCTS functionality."""
    print("Testing MCTS...")
    
    # Create Alpha Zero instance with smaller config for testing
    config = default_config
    config.mcts.num_simulations = 10  # Small number for testing
    alpha_zero = AlphaZero(config)
    
    # Test MCTS on initial position
    game = NonagaGame()
    action_probs = alpha_zero.mcts.get_action_probs(game, temperature=1.0)
    
    assert action_probs.shape == (81,), f"Expected action_probs shape (81,), got {action_probs.shape}"
    assert np.abs(np.sum(action_probs) - 1.0) < 1e-5, f"Action probs should sum to 1, got {np.sum(action_probs)}"
    
    # Check that invalid actions have 0 probability
    valid_actions = game.get_valid_actions()
    for i in range(81):
        if valid_actions[i] == 0:
            assert action_probs[i] == 0, f"Invalid action {i} should have 0 probability"
    
    print("✓ MCTS test passed!")

def test_self_play():
    """Test self-play functionality."""
    print("Testing self-play...")
    
    # Create Alpha Zero instance with minimal config
    config = default_config
    config.mcts.num_simulations = 5  # Very small for testing
    alpha_zero = AlphaZero(config)
    
    # Play one self-play game
    examples = alpha_zero.self_play_game()
    
    assert len(examples) > 0, "Self-play should generate examples"
    
    for board_state, policy, value in examples:
        assert board_state.shape == (3, 3, 3, 3, 3), f"Expected board shape (3,3,3,3,3), got {board_state.shape}"
        assert policy.shape == (81,), f"Expected policy shape (81,), got {policy.shape}"
        assert -1 <= value <= 1, f"Value should be in [-1, 1], got {value}"
    
    print(f"✓ Self-play test passed! Generated {len(examples)} examples")

def test_training():
    """Test training functionality."""
    print("Testing training...")
    
    # Create Alpha Zero instance with minimal config
    config = default_config
    config.mcts.num_simulations = 5
    config.training.num_self_play_games = 2
    config.training.num_epochs = 2
    config.training.batch_size = 4
    alpha_zero = AlphaZero(config)
    
    # Run one training iteration
    stats = alpha_zero.train_iteration()
    
    assert 'loss' in stats, "Training should return loss"
    assert 'buffer_size' in stats, "Training should return buffer size"
    assert stats['buffer_size'] > 0, "Buffer should contain examples after training"
    
    print(f"✓ Training test passed! Loss: {stats['loss']:.4f}, Buffer size: {stats['buffer_size']}")

if __name__ == "__main__":
    print("Running Alpha Zero tests...\n")
    
    test_network()
    test_mcts()
    test_self_play()
    test_training()
    
    print("\n✓ All tests passed! Alpha Zero implementation is working correctly.")