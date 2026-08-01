import numpy as np
import os
import pickle


class QLearningAgent:
    """
    Q-Learning agent for dynamic early-exit decisions in branchy networks.

    The agent learns, per (exit_idx, confidence_bin) state, whether it is better
    to exit now or keep computing. Inference thresholds are read off the learned
    Q-table: for each exit, the lowest confidence bin at which exiting is
    preferred becomes that exit's confidence threshold.

    Action encoding (must match train.py): 0 = continue, 1 = exit.

    The previous LeNet agent stored its Q-table as nested Python lists, decayed
    exploration on every single update (so exploration collapsed within one
    epoch), and -- most importantly -- was never actually driven by train.py
    (select_action/update were never called), so it was pickled with a
    random-init table. This version matches the AlexNet/MLP agent: a numpy
    Q-table, exploration decayed once per epoch via end_episode(), and thresholds
    derived from the learned values.
    """

    def __init__(self, n_exits=2, n_confidence_bins=10, learning_rate=0.1,
                 discount_factor=0.95, exploration_rate=0.5,
                 exploration_decay=0.995, min_exploration_rate=0.01):
        self.n_exits = n_exits
        self.n_confidence_bins = n_confidence_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Q-table: [exit_idx, confidence_bin, action], action 0=continue 1=exit
        self.q_table = np.zeros((n_exits, n_confidence_bins, 2))

        # How often each (exit, confidence-bin) state was actually visited.
        # Thresholds must only be derived from VISITED bins: unvisited bins hold
        # initialization/noise Q-values, and deriving a threshold from one (e.g.
        # bin 1 -> threshold 0.1, below the 1/n_classes softmax floor) makes
        # every sample exit unconditionally at the first timestep.
        self.visit_counts = np.zeros((n_exits, n_confidence_bins), dtype=np.int64)

        # Conservative defaults until the agent has learned something.
        self.confidence_thresholds = [0.9] * n_exits

        self.episode = 0

    def get_state(self, exit_idx, confidence):
        """Map (exit index, confidence in [0,1]) to a discrete state."""
        confidence = float(np.clip(confidence, 0.0, 1.0))
        confidence_bin = min(int(confidence * self.n_confidence_bins),
                             self.n_confidence_bins - 1)
        return (exit_idx, confidence_bin)

    def select_action(self, state):
        """Epsilon-greedy action selection. Returns 0 = continue, 1 = exit."""
        exit_idx, confidence_bin = state
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, 2)
        return int(np.argmax(self.q_table[exit_idx, confidence_bin, :]))

    def update(self, state, action, reward, next_state=None, done=False):
        """Q-learning update for a single (state, action, reward) transition."""
        exit_idx, confidence_bin = state
        current_q = self.q_table[exit_idx, confidence_bin, action]
        self.visit_counts[exit_idx, confidence_bin] += 1

        if done or next_state is None:
            target_q = reward
        else:
            next_exit_idx, next_confidence_bin = next_state
            max_next_q = np.max(self.q_table[next_exit_idx, next_confidence_bin, :])
            target_q = reward + self.discount_factor * max_next_q

        self.q_table[exit_idx, confidence_bin, action] += \
            self.learning_rate * (target_q - current_q)

    def end_episode(self):
        """Call once per EPISODE -- i.e. once per minibatch, which is what the
        trainer does: decays exploration and refreshes thresholds."""
        self.episode += 1
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        self._update_confidence_thresholds()

    def _update_confidence_thresholds(self):
        """Derive per-exit confidence thresholds from the learned Q-table."""
        thresholds = []
        for exit_idx in range(self.n_exits):
            best_threshold = 0.9  # if no visited bin prefers exiting, effectively never exit early
            for conf_bin in range(self.n_confidence_bins):
                if self.visit_counts[exit_idx, conf_bin] == 0:
                    continue  # unvisited: Q-values are noise, not evidence
                q_continue = self.q_table[exit_idx, conf_bin, 0]
                q_exit = self.q_table[exit_idx, conf_bin, 1]
                if q_exit > q_continue:
                    best_threshold = conf_bin / self.n_confidence_bins
                    break
            thresholds.append(best_threshold)
        self.confidence_thresholds = thresholds

    def get_confidence_thresholds(self):
        return self.confidence_thresholds

    def save(self, path):
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
            'min_exploration_rate': self.min_exploration_rate,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(agent_data, f)

    def load(self, path):
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
