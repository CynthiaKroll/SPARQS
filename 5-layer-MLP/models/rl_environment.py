import numpy as np
import torch
import pickle
import os

class QLearningAgent:
    """
    Q-Learning agent for dynamic early exit decisions in branchy neural networks.
    """
    
    def __init__(self, n_exits=2, n_confidence_bins=10, learning_rate=0.1, 
                 discount_factor=0.95, exploration_rate=0.5, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_exits: Number of early exit points
            n_confidence_bins: Number of bins to discretize confidence values
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial epsilon for epsilon-greedy exploration
            exploration_decay: Decay rate for exploration
            min_exploration_rate: Minimum exploration rate
        """
        self.n_exits = n_exits
        self.n_confidence_bins = n_confidence_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Q-table: [exit_idx, confidence_bin, action]
        # Actions: 0 = continue, 1 = exit
        self.q_table = np.zeros((n_exits, n_confidence_bins, 2))

        # How often each (exit, confidence-bin) state was actually visited.
        # Thresholds must only be derived from VISITED bins: unvisited bins hold
        # initialization/noise Q-values, and deriving a threshold from one (e.g.
        # bin 1 -> threshold 0.1, below the 1/n_classes softmax floor) makes
        # every sample exit unconditionally at the first timestep.
        self.visit_counts = np.zeros((n_exits, n_confidence_bins), dtype=np.int64)

        # Initialize confidence thresholds
        self.confidence_thresholds = [0.9, 0.8]  # Default values

        # Keep track of episodes
        self.episode = 0
    
    def get_state(self, exit_idx, confidence):
        """
        Convert exit index and confidence to state.
        
        Args:
            exit_idx: Exit point index (0, 1, ...)
            confidence: Confidence value [0, 1]
        
        Returns:
            state: Discretized state representation
        """
        # Discretize confidence into bins
        confidence = np.clip(confidence, 0.0, 1.0)
        confidence_bin = min(int(confidence * self.n_confidence_bins), 
                           self.n_confidence_bins - 1)
        
        return (exit_idx, confidence_bin)
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (exit_idx, confidence_bin)
        
        Returns:
            action: 0 = continue, 1 = exit
        """
        exit_idx, confidence_bin = state
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Random action
            action = np.random.randint(0, 2)
        else:
            # Greedy action
            action = np.argmax(self.q_table[exit_idx, confidence_bin, :])
        
        return action
    
    def update(self, state, action, reward, next_state=None, done=False):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if episode ended)
            done: Whether episode is complete
        """
        exit_idx, confidence_bin = state
        current_q = self.q_table[exit_idx, confidence_bin, action]
        self.visit_counts[exit_idx, confidence_bin] += 1

        if done or next_state is None:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state
            next_exit_idx, next_confidence_bin = next_state
            max_next_q = np.max(self.q_table[next_exit_idx, next_confidence_bin, :])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        self.q_table[exit_idx, confidence_bin, action] += \
            self.learning_rate * (target_q - current_q)
    
    def end_episode(self):
        """
        End current episode and update exploration rate.
        """
        self.episode += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        # Update confidence thresholds based on learned Q-values
        self._update_confidence_thresholds()
    
    def _update_confidence_thresholds(self):
        """
        Update confidence thresholds based on learned Q-values.
        """
        thresholds = []

        for exit_idx in range(self.n_exits):
            # Find the lowest VISITED confidence bin where exiting is preferred.
            # If no visited bin prefers exiting, stay conservative (never exit).
            best_threshold = 0.9

            for conf_bin in range(self.n_confidence_bins):
                if self.visit_counts[exit_idx, conf_bin] == 0:
                    continue  # unvisited: Q-values are noise, not evidence
                q_continue = self.q_table[exit_idx, conf_bin, 0]
                q_exit = self.q_table[exit_idx, conf_bin, 1]

                if q_exit > q_continue:
                    # Exit is preferred at this confidence level
                    best_threshold = conf_bin / self.n_confidence_bins
                    break

            thresholds.append(best_threshold)

        self.confidence_thresholds = thresholds
    
    def get_confidence_thresholds(self):
        """
        Get current confidence thresholds.
        
        Returns:
            List of confidence thresholds for each exit
        """
        return self.confidence_thresholds
    
    def save(self, path):
        """
        Save agent state to file.
        
        Args:
            path: File path to save agent
        """
        agent_data = {
            'q_table': self.q_table,
            'visit_counts': self.visit_counts,
            'confidence_thresholds': self.confidence_thresholds,
            'exploration_rate': self.exploration_rate,
            'episode': self.episode,
            'n_exits': self.n_exits,
            'n_confidence_bins': self.n_confidence_bins,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_decay': self.exploration_decay,
            'min_exploration_rate': self.min_exploration_rate
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(agent_data, f)
    
    def load(self, path):
        """
        Load agent state from file.
        
        Args:
            path: File path to load agent from
        """
        if not os.path.exists(path):
            print(f"Agent file {path} not found. Using default initialization.")
            return
        
        with open(path, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_table = agent_data['q_table']
        # Older pickles predate visit tracking; treat every bin as visited then.
        self.visit_counts = agent_data.get(
            'visit_counts', np.ones(self.q_table.shape[:2], dtype=np.int64))
        self.confidence_thresholds = agent_data['confidence_thresholds']
        self.exploration_rate = agent_data['exploration_rate']
        self.episode = agent_data['episode']
        self.n_exits = agent_data['n_exits']
        self.n_confidence_bins = agent_data['n_confidence_bins']
        self.learning_rate = agent_data['learning_rate']
        self.discount_factor = agent_data['discount_factor']
        self.exploration_decay = agent_data['exploration_decay']
        self.min_exploration_rate = agent_data['min_exploration_rate'] 