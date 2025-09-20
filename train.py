"""
Training script for Alpha Zero Nonaga.
"""

import argparse
import os
from alpha_zero import AlphaZero
from config import default_config


def main():
    parser = argparse.ArgumentParser(description="Train Alpha Zero for Nonaga")
    parser.add_argument("--iterations", type=int, default=100, 
                       help="Number of training iterations")
    parser.add_argument("--self-play-games", type=int, default=25,
                       help="Number of self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs per iteration")
    parser.add_argument("--load-model", type=str, default=None,
                       help="Path to load existing model")
    parser.add_argument("--save-dir", type=str, default="./models",
                       help="Directory to save models")
    parser.add_argument("--play-human", action="store_true",
                       help="Play against human after training")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Update config with command line arguments
    config = default_config
    config.training.num_iterations = args.iterations
    config.training.num_self_play_games = args.self_play_games
    config.training.num_epochs = args.epochs
    
    # Initialize Alpha Zero
    alpha_zero = AlphaZero(config)
    
    # Load existing model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        alpha_zero.load_model(args.load_model)
    
    # Train the model
    print("Starting training...")
    alpha_zero.train()
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    alpha_zero.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Play against human if requested
    if args.play_human:
        alpha_zero.play_human_game()


if __name__ == "__main__":
    main()