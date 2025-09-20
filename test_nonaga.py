"""Simple test to verify Nonaga game logic."""

from nonaga import NonagaGame, Player, GameResult

def test_basic_game():
    """Test basic game functionality."""
    game = NonagaGame()
    
    # Test initial state
    assert game.current_player == Player.X
    assert game.result == GameResult.ONGOING
    assert game.active_subboard is None
    
    # Test making a move
    valid_moves = game.get_valid_moves()
    assert len(valid_moves) == 81  # All positions should be valid initially
    
    # Make a move in center subboard, center position
    success = game.make_move(1, 1, 1, 1)
    assert success
    assert game.current_player == Player.O  # Should switch players
    assert game.active_subboard == (1, 1)  # Next move must be in center subboard
    
    # Verify board state
    assert game.board[1, 1, 1, 1] == Player.X.value
    
    # Test valid moves after first move
    valid_moves = game.get_valid_moves()
    # Should only be moves in the (1,1) subboard, minus the occupied center
    expected_moves = 8  # 9 positions - 1 occupied
    assert len(valid_moves) == expected_moves
    
    print("Basic game test passed!")

def test_canonical_form():
    """Test canonical form conversion."""
    game = NonagaGame()
    game.make_move(0, 0, 0, 0)  # X move
    game.make_move(0, 0, 0, 1)  # O move
    
    # Test canonical form from X perspective
    canonical_x = game.get_canonical_form(Player.X)
    assert canonical_x[0, 0, 0, 0, 0] == 1  # X position
    assert canonical_x[0, 0, 0, 1, 1] == 1  # O position (opponent from X's view)
    
    # Test canonical form from O perspective  
    canonical_o = game.get_canonical_form(Player.O)
    assert canonical_o[0, 0, 0, 0, 1] == 1  # X position (opponent from O's view)
    assert canonical_o[0, 0, 0, 1, 0] == 1  # O position
    
    print("Canonical form test passed!")

def test_action_conversion():
    """Test action/move conversion."""
    game = NonagaGame()
    
    # Test move to action conversion
    action = game.move_to_action(1, 2, 0, 1)
    big_row, big_col, small_row, small_col = game.action_to_move(action)
    assert (big_row, big_col, small_row, small_col) == (1, 2, 0, 1)
    
    # Test all conversions are consistent
    for action in range(81):
        move = game.action_to_move(action)
        converted_action = game.move_to_action(*move)
        assert action == converted_action
        
    print("Action conversion test passed!")

if __name__ == "__main__":
    test_basic_game()
    test_canonical_form() 
    test_action_conversion()
    print("All tests passed!")