"""
Monte Carlo Tree Search implementation for Alpha Zero Nonaga.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from nonaga import NonagaGame, Player, GameResult
from neural_network import NonagaNetwork


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, game_state: NonagaGame, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None, prior: float = 0.0):
        self.game_state = game_state.copy()
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior  # Prior probability from neural network
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        
        # Cache for neural network evaluation
        self._policy: Optional[np.ndarray] = None
        self._value: Optional[float] = None
        self._expanded = False
        
    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return self._expanded
        
    def get_value(self) -> float:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def get_ucb_score(self, c_puct: float) -> float:
        """Calculate UCB score for node selection."""
        if self.visit_count == 0:
            return float('inf')
            
        # UCB1 formula adapted for AlphaZero
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.get_value() + exploration
        
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select the best child using UCB."""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            score = child.get_ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
        
    def expand(self, policy: np.ndarray, network: NonagaNetwork):
        """Expand the node by adding children for all valid actions."""
        if self._expanded:
            return
            
        valid_actions = self.game_state.get_valid_actions()
        
        # Add Dirichlet noise to root node for exploration
        if self.parent is None:
            noise = np.random.dirichlet([0.3] * len(policy))
            policy = 0.75 * policy + 0.25 * noise
            
        # Normalize policy over valid actions
        valid_policy = policy * valid_actions
        if np.sum(valid_policy) > 0:
            valid_policy = valid_policy / np.sum(valid_policy)
        else:
            # Uniform distribution if no valid policy
            valid_policy = valid_actions / np.sum(valid_actions)
            
        # Create child nodes for valid actions
        for action in range(len(valid_actions)):
            if valid_actions[action] > 0:
                # Create new game state
                new_game = self.game_state.copy()
                move = new_game.action_to_move(action)
                success = new_game.make_move(*move)
                
                if success:
                    child = MCTSNode(new_game, parent=self, action=action, prior=valid_policy[action])
                    self.children[action] = child
                    
        self._expanded = True
        
    def backup(self, value: float):
        """Backup the value up the tree."""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # Flip value for opponent
            self.parent.backup(-value)
            
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities based on visit counts."""
        probs = np.zeros(81)
        
        if temperature == 0:
            # Deterministic: choose most visited action
            best_action = max(self.children.keys(), key=lambda a: self.children[a].visit_count)
            probs[best_action] = 1.0
        else:
            # Stochastic: probability proportional to visit counts
            counts = np.array([self.children.get(a, MCTSNode(self.game_state)).visit_count 
                              for a in range(81)])
            
            if temperature == 1.0:
                probs = counts / np.sum(counts) if np.sum(counts) > 0 else probs
            else:
                # Apply temperature
                counts = counts ** (1.0 / temperature)
                probs = counts / np.sum(counts) if np.sum(counts) > 0 else probs
                
        return probs


class MCTS:
    """Monte Carlo Tree Search for Alpha Zero."""
    
    def __init__(self, network: NonagaNetwork, config):
        self.network = network
        self.config = config
        
    def search(self, game_state: NonagaGame, num_simulations: int = None) -> np.ndarray:
        """
        Run MCTS simulations and return action probabilities.
        
        Args:
            game_state: Current game state
            num_simulations: Number of simulations to run
            
        Returns:
            Action probabilities based on visit counts
        """
        if num_simulations is None:
            num_simulations = self.config.mcts.num_simulations
            
        # Create root node
        root = MCTSNode(game_state)
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)
            
        # Return action probabilities
        return root.get_action_probs(temperature=1.0)
        
    def _simulate(self, root: MCTSNode):
        """Run a single MCTS simulation."""
        node = root
        path = [node]
        
        # Selection: traverse tree using UCB until leaf
        while node.is_expanded() and len(node.children) > 0:
            node = node.select_child(self.config.mcts.c_puct)
            path.append(node)
            
        # Check if game is terminal
        if node.game_state.result != GameResult.ONGOING:
            # Terminal node - backup actual game result
            if node.game_state.result == GameResult.DRAW:
                value = 0.0
            elif node.game_state.result == GameResult.X_WINS:
                # Value from X's perspective
                value = 1.0 if node.game_state.current_player == Player.O else -1.0
            else:  # O_WINS
                # Value from X's perspective  
                value = -1.0 if node.game_state.current_player == Player.O else 1.0
        else:
            # Expansion and Evaluation
            if not node.is_expanded():
                # Get neural network prediction
                canonical_board = node.game_state.get_canonical_form(node.game_state.current_player)
                policy, value = self.network.predict(canonical_board)
                
                # Expand node
                node.expand(policy, self.network)
                
            else:
                # Leaf node that's already expanded - use network evaluation
                canonical_board = node.game_state.get_canonical_form(node.game_state.current_player)
                _, value = self.network.predict(canonical_board)
                
        # Backup: update all nodes in path
        for node in reversed(path):
            node.backup(value)
            value = -value  # Flip for opponent
            
    def get_action_probs(self, game_state: NonagaGame, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities for the current game state."""
        root = MCTSNode(game_state)
        
        # Run simulations
        for _ in range(self.config.mcts.num_simulations):
            self._simulate(root)
            
        return root.get_action_probs(temperature)
        
    def select_action(self, game_state: NonagaGame, temperature: float = 1.0) -> int:
        """Select an action using MCTS."""
        probs = self.get_action_probs(game_state, temperature)
        
        # Filter valid actions
        valid_actions = game_state.get_valid_actions()
        probs = probs * valid_actions
        
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
            action = np.random.choice(len(probs), p=probs)
        else:
            # Fallback to random valid action
            valid_indices = np.where(valid_actions > 0)[0]
            action = np.random.choice(valid_indices)
            
        return action