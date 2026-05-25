"""RL optimization agent for solar microgrid power routing.

Supports tabular Q-learning and Deep Q-Network (DQN) backends.
The agent observes normalized state vectors and selects discrete power
routing actions. Rewards are computed externally by simulator.py.
"""

from __future__ import annotations

import collections
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Module-level constants
STATE_DIM: int = 5
# Dimensions:
# [0] hour_of_day   normalized to [0, 1] by dividing by 23
# [1] battery_soc   already in [0, 1]
# [2] pv_power_kw   normalized by max expected PV output (e.g. 50 kW)
# [3] load_kw       normalized by max expected load (e.g. 20 kW)
# [4] grid_price    normalized by max expected price (e.g. 0.30 $/kWh)

NUM_ACTIONS: int = 4

ACTION_DESCRIPTIONS: dict[int, str] = {
    0: "PV -> Load; excess PV -> Battery",
    1: "PV -> Load; excess PV -> Grid (sell)",
    2: "Battery -> Load; Grid supplements if needed",
    3: "Grid -> Load; Grid -> Battery (arbitrage)",
}

CHECKPOINT_DIR: str = "./checkpoints"

log = logging.getLogger(__name__)

# Type alias for a single replay buffer experience
_Experience = tuple[np.ndarray, int, float, np.ndarray, bool]


class StateNormalizer:
    """Converts raw simulator outputs to normalized [0, 1] state vectors."""

    def __init__(
        self,
        max_pv_kw: float = 50.0,
        max_load_kw: float = 20.0,
        max_grid_price: float = 0.30,
    ) -> None:
        """Initialize normalizer with per-dimension maxima.

        Args:
            max_pv_kw: Maximum expected PV output in kW.
            max_load_kw: Maximum expected load in kW.
            max_grid_price: Maximum expected grid price in $/kWh.
        """
        self.max_pv_kw = max(float(max_pv_kw), 1e-9)
        self.max_load_kw = max(float(max_load_kw), 1e-9)
        self.max_grid_price = max(float(max_grid_price), 1e-9)

    def normalize(
        self,
        hour: int,
        soc: float,
        pv_kw: float,
        load_kw: float,
        grid_price: float,
    ) -> np.ndarray:
        """Return normalized float32 state vector of shape (STATE_DIM,).

        Args:
            hour: Hour of day in [0, 23].
            soc: Battery state of charge in [0, 1].
            pv_kw: PV power output in kW.
            load_kw: Load demand in kW.
            grid_price: Grid electricity price in $/kWh.

        Returns:
            Float32 numpy array of shape (STATE_DIM,) with values in [0, 1].
        """
        raw = np.array(
            [
                float(hour) / 23.0,
                float(soc),
                float(pv_kw) / self.max_pv_kw,
                float(load_kw) / self.max_load_kw,
                float(grid_price) / self.max_grid_price,
            ],
            dtype=np.float32,
        )
        return np.clip(raw, 0.0, 1.0).astype(np.float32)

    def validate(self, state: np.ndarray) -> bool:
        """Validate that a state vector is well-formed.

        Args:
            state: Array to validate.

        Returns:
            True if the state is valid.

        Raises:
            ValueError: If shape, dtype, or value range is incorrect.
        """
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"Expected shape ({STATE_DIM},), got {state.shape}"
            )
        if state.dtype != np.float32:
            raise ValueError(
                f"Expected dtype float32, got {state.dtype}"
            )
        if not (np.all(state >= 0.0) and np.all(state <= 1.0)):
            raise ValueError(
                f"All values must be in [0.0, 1.0]; "
                f"got min={float(state.min()):.4f}, max={float(state.max()):.4f}"
            )
        return True


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration.

    Uses a discretized state space with uniform bins. Q-values are stored
    in a multi-dimensional numpy array indexed by bin indices.
    """

    def __init__(
        self,
        n_bins: int = 10,
        n_actions: int = NUM_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.97,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        """Initialize tabular Q-learning agent.

        Args:
            n_bins: Number of discrete bins per state dimension.
            n_actions: Number of discrete actions.
            alpha: TD learning rate in (0, 1].
            gamma: Discount factor in [0, 1).
            epsilon: Initial exploration probability.
            epsilon_min: Minimum (floor) epsilon after decay.
            epsilon_decay: Multiplicative per-step decay factor.
            seed: RNG seed for reproducibility.
        """
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self._rng = np.random.default_rng(seed)
        shape: tuple[int, ...] = (n_bins,) * STATE_DIM + (n_actions,)
        self.q_table: np.ndarray = self._rng.uniform(-0.01, 0.01, shape).astype(np.float64)
        self.training_step: int = 0
        self.episode_rewards: list[float] = []

    def _discretize(self, state: np.ndarray) -> tuple[int, ...]:
        """Map continuous [0, 1] state to discrete bin indices.

        Args:
            state: Normalized state vector of shape (STATE_DIM,).

        Returns:
            Tuple of integer bin indices in [0, n_bins - 1].
        """
        clipped = np.clip(state, 0.0, 1.0)
        indices = np.floor(clipped * self.n_bins).clip(0, self.n_bins - 1).astype(int)
        return tuple(int(i) for i in indices)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action via epsilon-greedy or greedy policy.

        Args:
            state: Normalized state vector of shape (STATE_DIM,).
            training: If True, apply epsilon-greedy; if False, pure greedy.

        Returns:
            Integer action index in [0, n_actions).
        """
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        idx = self._discretize(state)
        return int(np.argmax(self.q_table[idx]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Apply Bellman Q-update and return absolute TD error.

        Args:
            state: Current normalized state.
            action: Action taken (integer index).
            reward: Scalar reward received from the environment.
            next_state: Next normalized state.
            done: True if the episode terminated after this step.

        Returns:
            Absolute TD error (non-negative float).
        """
        reward = float(np.nan_to_num(reward))
        idx = self._discretize(state)
        next_idx = self._discretize(next_state)
        old_q = float(self.q_table[idx][action])
        max_next_q = float(np.max(self.q_table[next_idx]))
        bootstrap = 0.0 if done else self.gamma * max_next_q
        target = reward + bootstrap
        td_error = target - old_q
        self.q_table[idx][action] = old_q + self.alpha * td_error
        self.training_step += 1
        log.debug(
            "Q-update step=%d action=%d reward=%.4f td_error=%.6f",
            self.training_step, action, reward, td_error,
        )
        return abs(td_error)

    def decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay with floor at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode_reward(self, total_reward: float) -> None:
        """Append cumulative episode reward to history.

        Args:
            total_reward: Sum of rewards over a completed episode.
        """
        self.episode_rewards.append(float(total_reward))

    def get_stats(self) -> dict[str, float]:
        """Return current training statistics.

        Returns:
            Dict with keys: epsilon, training_step, mean_reward_last_10,
            q_table_mean, q_table_std.
        """
        recent = self.episode_rewards[-10:] if self.episode_rewards else [0.0]
        return {
            "epsilon": float(self.epsilon),
            "training_step": float(self.training_step),
            "mean_reward_last_10": float(np.mean(recent)),
            "q_table_mean": float(np.mean(self.q_table)),
            "q_table_std": float(np.std(self.q_table)),
        }

    def save(self, path: str) -> None:
        """Save agent state to a .npz file.

        Args:
            path: Destination file path (parent directory created if absent).
        """
        path_str = str(path)
        parent = os.path.dirname(path_str)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        np.savez(
            path_str,
            q_table=self.q_table,
            epsilon=np.array(self.epsilon),
            training_step=np.array(self.training_step),
            episode_rewards=np.array(self.episode_rewards),
        )
        log.info("QLearningAgent saved to %s", path_str)

    def load(self, path: str) -> None:
        """Restore agent state from a .npz file.

        Args:
            path: Source file path.
        """
        path_str = str(path)
        data = np.load(path_str, allow_pickle=True)
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"])
        self.training_step = int(data["training_step"])
        self.episode_rewards = list(data["episode_rewards"])
        log.info("QLearningAgent loaded from %s", path_str)


class _QNetwork(nn.Module):
    """Fully-connected Q-value network for DQN."""

    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int) -> None:
        """Build the network architecture.

        Args:
            state_dim: Dimension of the input state vector.
            hidden_dim: Width of each hidden layer.
            n_actions: Number of output Q-values (one per action).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states.

        Args:
            x: Input tensor of shape (batch_size, state_dim).

        Returns:
            Q-value tensor of shape (batch_size, n_actions).
        """
        return self.net(x)


class DQNAgent:
    """Deep Q-Network agent with experience replay and a target network.

    Shares the same public interface as QLearningAgent.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        alpha: float = 1e-3,
        gamma: float = 0.97,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_update_freq: int = 50,
        seed: int = 42,
    ) -> None:
        """Initialize DQN agent.

        Args:
            state_dim: Dimension of the normalized state vector.
            n_actions: Number of discrete actions.
            hidden_dim: Hidden layer width in the Q-network.
            alpha: Adam optimizer learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration probability.
            epsilon_min: Minimum epsilon after decay.
            epsilon_decay: Multiplicative per-step decay factor.
            batch_size: Mini-batch size for gradient updates.
            replay_capacity: Maximum number of experiences in the buffer.
            target_update_freq: Steps between target network syncs.
            seed: Random seed for reproducibility.
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self._policy_net: nn.Module = _QNetwork(state_dim, hidden_dim, n_actions).to(self._device)
        self._target_net: nn.Module = _QNetwork(state_dim, hidden_dim, n_actions).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=alpha)
        self._replay_buffer: collections.deque[_Experience] = collections.deque(
            maxlen=replay_capacity
        )

        self.training_step: int = 0
        self.episode_rewards: list[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action via epsilon-greedy or greedy policy.

        Args:
            state: Normalized state vector of shape (STATE_DIM,).
            training: If True, apply epsilon-greedy; if False, pure greedy.

        Returns:
            Integer action index in [0, n_actions).
        """
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        self._policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            q_vals = self._policy_net(state_t)
        self._policy_net.train()
        return int(q_vals.argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Push experience to replay buffer and optionally run gradient update.

        Args:
            state: Current normalized state.
            action: Action taken.
            reward: Scalar reward received.
            next_state: Next normalized state.
            done: True if the episode terminated after this step.

        Returns:
            Mean absolute TD error, or 0.0 if buffer has fewer than batch_size samples.
        """
        reward = float(np.nan_to_num(reward))
        self._replay_buffer.append(
            (state.copy(), int(action), reward, next_state.copy(), bool(done))
        )
        self.training_step += 1

        if len(self._replay_buffer) < self.batch_size:
            return 0.0

        # Sample mini-batch from replay buffer
        buffer_list = list(self._replay_buffer)
        indices = self._rng.choice(len(buffer_list), size=self.batch_size, replace=False)
        batch = [buffer_list[i] for i in indices]

        states_np = np.array([t[0] for t in batch], dtype=np.float32)
        actions_np = np.array([t[1] for t in batch], dtype=np.int64)
        rewards_np = np.array([t[2] for t in batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in batch], dtype=np.float32)
        dones_np = np.array([float(t[4]) for t in batch], dtype=np.float32)

        states_t = torch.from_numpy(states_np).to(self._device)
        actions_t = torch.from_numpy(actions_np).to(self._device)
        rewards_t = torch.from_numpy(rewards_np).to(self._device)
        next_states_t = torch.from_numpy(next_states_np).to(self._device)
        dones_t = torch.from_numpy(dones_np).to(self._device)

        # Q(s, a) from policy net
        self._policy_net.train()
        q_vals = self._policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values from frozen target net
        with torch.no_grad():
            q_next = self._target_net(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_vals, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Periodically sync target net to policy net
        if self.training_step % self.target_update_freq == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
            log.debug("Target net synced at step %d", self.training_step)

        td_errors = (targets - q_vals.detach()).abs()
        mean_td = float(td_errors.mean().item())
        log.debug("DQN update step=%d loss=%.6f mean_td=%.6f", self.training_step, loss.item(), mean_td)
        return mean_td

    def decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay with floor at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode_reward(self, total_reward: float) -> None:
        """Append cumulative episode reward to history.

        Args:
            total_reward: Sum of rewards over a completed episode.
        """
        self.episode_rewards.append(float(total_reward))

    def get_stats(self) -> dict[str, float]:
        """Return current training statistics.

        Returns:
            Dict with keys: epsilon, training_step, mean_reward_last_10,
            q_table_mean, q_table_std, replay_buffer_size.
        """
        recent = self.episode_rewards[-10:] if self.episode_rewards else [0.0]
        return {
            "epsilon": float(self.epsilon),
            "training_step": float(self.training_step),
            "mean_reward_last_10": float(np.mean(recent)),
            "q_table_mean": 0.0,
            "q_table_std": 0.0,
            "replay_buffer_size": float(len(self._replay_buffer)),
        }

    def save(self, path: str) -> None:
        """Save agent state to a file via torch.save.

        Args:
            path: Destination file path (parent directory created if absent).
        """
        path_str = str(path)
        parent = os.path.dirname(path_str)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "policy_net_state": self._policy_net.state_dict(),
                "target_net_state": self._target_net.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "episode_rewards": self.episode_rewards,
            },
            path_str,
        )
        log.info("DQNAgent saved to %s", path_str)

    def load(self, path: str) -> None:
        """Restore agent state from a file saved by torch.save.

        Args:
            path: Source file path.
        """
        path_str = str(path)
        data = torch.load(path_str, map_location=self._device, weights_only=False)
        self._policy_net.load_state_dict(data["policy_net_state"])
        self._target_net.load_state_dict(data["target_net_state"])
        self._optimizer.load_state_dict(data["optimizer_state"])
        self.epsilon = float(data["epsilon"])
        self.training_step = int(data["training_step"])
        self.episode_rewards = list(data["episode_rewards"])
        log.info("DQNAgent loaded from %s", path_str)


def create_agent(
    agent_type: str = "qlearning",
    **kwargs: Any,
) -> QLearningAgent | DQNAgent:
    """Instantiate and return the requested agent type.

    Args:
        agent_type: Either "qlearning" or "dqn".
        **kwargs: Passed directly to the agent constructor.

    Returns:
        A configured agent instance.

    Raises:
        ValueError: If agent_type is not recognized.
    """
    if agent_type == "qlearning":
        return QLearningAgent(**kwargs)
    if agent_type == "dqn":
        return DQNAgent(**kwargs)
    raise ValueError(
        f"Unknown agent_type {agent_type!r}. Expected 'qlearning' or 'dqn'."
    )


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    _log = _logging.getLogger(__name__)

    _log.info("=== Phase 2 Agent Validation ===")

    norm = StateNormalizer()
    state = norm.normalize(hour=13, soc=0.65, pv_kw=32.0, load_kw=8.5, grid_price=0.14)
    _log.info("Normalized state: %s", state)
    assert norm.validate(state), "State validation failed"

    _log.info("--- Q-Learning Agent ---")
    q_agent = QLearningAgent(n_bins=10, seed=42)
    for i in range(100):
        action = q_agent.select_action(state)
        reward = 1.0 if action in (0, 1) else -0.5
        q_agent.update(state, action, reward, state, done=False)
    q_agent.decay_epsilon()
    q_agent.record_episode_reward(-12.5)
    stats = q_agent.get_stats()
    _log.info("Stats after 100 steps: %s", stats)
    assert stats["training_step"] == 100
    assert stats["epsilon"] < 1.0
    assert all(np.isfinite(v) for v in stats.values())

    _log.info("--- DQN Agent ---")
    dqn = DQNAgent(hidden_dim=64, batch_size=16, replay_capacity=200, seed=42)
    for i in range(80):
        ns = norm.normalize(i % 24, 0.5, 20.0, 10.0, 0.12)
        dqn.update(state, i % 4, float(i % 3 - 1), ns, done=(i == 79))
    dqn_action = dqn.select_action(state, training=False)
    assert dqn_action in {0, 1, 2, 3}
    _log.info("DQN greedy action: %s", ACTION_DESCRIPTIONS[dqn_action])

    _log.info("--- Factory ---")
    assert isinstance(create_agent("qlearning", n_bins=5), QLearningAgent)
    assert isinstance(create_agent("dqn", hidden_dim=32), DQNAgent)

    print("\nPhase 2 validation PASSED")
