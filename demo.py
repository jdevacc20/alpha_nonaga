"""
Demo script for Alpha Zero Nonaga.
"""

from alpha_zero import AlphaZero
from nonaga import NonagaGame, Player
from config import default_config


def demo_self_play():
    """Demonstrate self-play functionality."""
    print("=== Alpha Zero Nonaga Demo ===\n")
    
    # Create Alpha Zero with smaller config for demo
    config = default_config
    config.mcts.num_simulations = 50  # Reasonable for demo
    alpha_zero = AlphaZero(config)
    
    print("1. Watching AI vs AI game (self-play)...")
    game = NonagaGame()
    move_count = 0
    
    print("\nInitial board:")
    print(game)
    
    while game.result.value == 0 and move_count < 50:  # Limit moves for demo
        print(f"\nMove {move_count + 1}: {game.current_player.name}'s turn")
        
        # Get AI move
        action = alpha_zero.mcts.select_action(game, temperature=0.1)
        move = game.action_to_move(action)
        
        print(f"AI plays action {action} at position {move}")
        
        # Make move
        success = game.make_move(*move)
        if not success:
            print("Invalid move!")
            break
            
        print("Board after move:")
        print(game)
        
        move_count += 1
        
        if game.result.value != 0:
            break
    
    print(f"\nGame ended: {game.result.name}")
    
    if move_count >= 50:
        print("Demo ended early (move limit reached)")


def demo_training():
    """Demonstrate training functionality."""
    print("\n2. Training demo (1 iteration with minimal settings)...")
    
    # Create Alpha Zero with very small config for demo
    config = default_config
    config.mcts.num_simulations = 10
    config.training.num_self_play_games = 2
    config.training.num_epochs = 2
    config.training.batch_size = 4
    
    alpha_zero = AlphaZero(config)
    
    # Run one training iteration
    stats = alpha_zero.train_iteration()
    
    print(f"Training completed!")
    print(f"  - Loss: {stats['loss']:.4f}")
    print(f"  - Buffer size: {stats['buffer_size']}")


def interactive_demo():
    """Interactive demo for user to play against AI."""
    print("\n3. Interactive game demo...")
    print("Would you like to play against the AI? (y/n): ", end="")
    
    try:
        response = input().strip().lower()
        if response == 'y' or response == 'yes':
            config = default_config
            config.mcts.num_simulations = 100  # Good for interactive play
            alpha_zero = AlphaZero(config)
            
            print("\nStarting interactive game!")
            alpha_zero.play_human_game(human_is_x=True)
        else:
            print("Skipping interactive demo.")
    except KeyboardInterrupt:
        print("\nInteractive demo cancelled.")


if __name__ == "__main__":
    try:
        demo_self_play()
        demo_training()
        interactive_demo()
        
        print("\n=== Demo Complete ===")
        print("The Alpha Zero implementation for Nonaga is working!")
        print("\nTo train a full model, run:")
        print("  python train.py --iterations 100 --self-play-games 50")
        print("\nTo play against a trained model:")
        print("  python train.py --load-model models/final_model.pth --play-human")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()