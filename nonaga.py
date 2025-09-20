"""
Nonaga (Ultimate Tic-Tac-Toe) game implementation.

Nonaga is a 3x3 grid of tic-tac-toe boards. Players must play in the subboard
corresponding to the last move's position within its subboard.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from enum import Enum


class Player(Enum):
    """Player enumeration."""
    NONE = 0
    X = 1
    O = 2


class GameResult(Enum):
    """Game result enumeration."""
    ONGOING = 0
    X_WINS = 1
    O_WINS = 2
    DRAW = 3


class NonagaGame:
    """Nonaga game implementation."""
    
    def __init__(self):
        # 3x3 grid of 3x3 subboards: board[big_row][big_col][small_row][small_col]
        self.board = np.zeros((3, 3, 3, 3), dtype=int)
        # Track which subboards are won: 0=ongoing, 1=X wins, 2=O wins, 3=draw
        self.subboard_winners = np.zeros((3, 3), dtype=int)
        # Current player (1=X, 2=O)
        self.current_player = Player.X
        # Which subboard the next move must be in (None = any available subboard)
        self.active_subboard = None
        # Game result
        self.result = GameResult.ONGOING
        
    def copy(self) -> 'NonagaGame':
        """Create a deep copy of the game state."""
        new_game = NonagaGame()
        new_game.board = self.board.copy()
        new_game.subboard_winners = self.subboard_winners.copy()
        new_game.current_player = self.current_player
        new_game.active_subboard = self.active_subboard
        new_game.result = self.result
        return new_game
        
    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """Get list of valid moves as (big_row, big_col, small_row, small_col)."""
        if self.result != GameResult.ONGOING:
            return []
            
        valid_moves = []
        
        if self.active_subboard is None:
            # Can play in any non-won subboard
            for big_row in range(3):
                for big_col in range(3):
                    if self.subboard_winners[big_row, big_col] == 0:
                        valid_moves.extend(self._get_subboard_moves(big_row, big_col))
        else:
            # Must play in specific subboard
            big_row, big_col = self.active_subboard
            if self.subboard_winners[big_row, big_col] == 0:
                valid_moves.extend(self._get_subboard_moves(big_row, big_col))
            else:
                # Active subboard is won, can play anywhere
                for big_row in range(3):
                    for big_col in range(3):
                        if self.subboard_winners[big_row, big_col] == 0:
                            valid_moves.extend(self._get_subboard_moves(big_row, big_col))
                            
        return valid_moves
        
    def _get_subboard_moves(self, big_row: int, big_col: int) -> List[Tuple[int, int, int, int]]:
        """Get valid moves within a specific subboard."""
        moves = []
        for small_row in range(3):
            for small_col in range(3):
                if self.board[big_row, big_col, small_row, small_col] == 0:
                    moves.append((big_row, big_col, small_row, small_col))
        return moves
        
    def make_move(self, big_row: int, big_col: int, small_row: int, small_col: int) -> bool:
        """Make a move. Returns True if successful."""
        if (big_row, big_col, small_row, small_col) not in self.get_valid_moves():
            return False
            
        # Make the move
        self.board[big_row, big_col, small_row, small_col] = self.current_player.value
        
        # Check if this subboard is now won
        self._check_subboard_winner(big_row, big_col)
        
        # Set next active subboard
        next_big_row, next_big_col = small_row, small_col
        if self.subboard_winners[next_big_row, next_big_col] != 0:
            # Next subboard is already won, can play anywhere
            self.active_subboard = None
        else:
            self.active_subboard = (next_big_row, next_big_col)
            
        # Check overall game winner
        self._check_game_winner()
        
        # Switch players
        self.current_player = Player.O if self.current_player == Player.X else Player.X
        
        return True
        
    def _check_subboard_winner(self, big_row: int, big_col: int):
        """Check if a subboard has a winner."""
        subboard = self.board[big_row, big_col]
        winner = self._check_winner_3x3(subboard)
        self.subboard_winners[big_row, big_col] = winner
        
    def _check_game_winner(self):
        """Check if the overall game has a winner."""
        winner = self._check_winner_3x3(self.subboard_winners)
        if winner == 1:
            self.result = GameResult.X_WINS
        elif winner == 2:
            self.result = GameResult.O_WINS
        elif winner == 3:
            self.result = GameResult.DRAW
        elif np.all(self.subboard_winners != 0):
            # All subboards are decided, determine winner by count
            x_wins = np.sum(self.subboard_winners == 1)
            o_wins = np.sum(self.subboard_winners == 2)
            if x_wins > o_wins:
                self.result = GameResult.X_WINS
            elif o_wins > x_wins:
                self.result = GameResult.O_WINS
            else:
                self.result = GameResult.DRAW
                
    def _check_winner_3x3(self, board: np.ndarray) -> int:
        """Check winner of a 3x3 board. Returns 0=ongoing, 1=X, 2=O, 3=draw."""
        # Check rows
        for row in range(3):
            if board[row, 0] == board[row, 1] == board[row, 2] != 0:
                return board[row, 0]
                
        # Check columns  
        for col in range(3):
            if board[0, col] == board[1, col] == board[2, col] != 0:
                return board[0, col]
                
        # Check diagonals
        if board[0, 0] == board[1, 1] == board[2, 2] != 0:
            return board[0, 0]
        if board[0, 2] == board[1, 1] == board[2, 0] != 0:
            return board[0, 2]
            
        # Check if board is full (draw)
        if np.all(board != 0):
            return 3
            
        return 0
        
    def get_canonical_form(self, player: Player) -> np.ndarray:
        """Get canonical form of the board from the perspective of the given player."""
        # Shape: (3, 3, 3, 3, 3) - last dimension is [current_player, opponent, empty]
        canonical = np.zeros((3, 3, 3, 3, 3))
        
        current_val = player.value
        opponent_val = Player.O.value if player == Player.X else Player.X.value
        
        for big_row in range(3):
            for big_col in range(3):
                for small_row in range(3):
                    for small_col in range(3):
                        cell = self.board[big_row, big_col, small_row, small_col]
                        if cell == current_val:
                            canonical[big_row, big_col, small_row, small_col, 0] = 1
                        elif cell == opponent_val:
                            canonical[big_row, big_col, small_row, small_col, 1] = 1
                        else:
                            canonical[big_row, big_col, small_row, small_col, 2] = 1
                            
        return canonical
        
    def get_action_size(self) -> int:
        """Get the number of possible actions."""
        return 3 * 3 * 3 * 3  # 81 possible positions
        
    def move_to_action(self, big_row: int, big_col: int, small_row: int, small_col: int) -> int:
        """Convert move coordinates to action index."""
        return big_row * 27 + big_col * 9 + small_row * 3 + small_col
        
    def action_to_move(self, action: int) -> Tuple[int, int, int, int]:
        """Convert action index to move coordinates."""
        big_row = action // 27
        big_col = (action % 27) // 9
        small_row = (action % 9) // 3
        small_col = action % 3
        return (big_row, big_col, small_row, small_col)
        
    def get_valid_actions(self) -> np.ndarray:
        """Get binary array of valid actions."""
        valid = np.zeros(81)
        for move in self.get_valid_moves():
            action = self.move_to_action(*move)
            valid[action] = 1
        return valid
        
    def __str__(self) -> str:
        """String representation of the game."""
        result = []
        for big_row in range(3):
            for small_row in range(3):
                line = ""
                for big_col in range(3):
                    for small_col in range(3):
                        cell = self.board[big_row, big_col, small_row, small_col]
                        if cell == 0:
                            line += "."
                        elif cell == 1:
                            line += "X"
                        else:
                            line += "O"
                    if big_col < 2:
                        line += "|"
                result.append(line)
            if big_row < 2:
                result.append("-" * 11)
        return "\n".join(result)