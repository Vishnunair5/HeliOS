"""Full pytest coverage for agent.py — Phase 2."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agent import (
    ACTION_DESCRIPTIONS,
    NUM_ACTIONS,
    STATE_DIM,
    DQNAgent,
    QLearningAgent,
    StateNormalizer,
    create_agent,
)


# ─────────────────────────────────────────────────────────────────────────────
# TestStateNormalizer
# ─────────────────────────────────────────────────────────────────────────────


class TestStateNormalizer:

    def test_output_shape(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(12, 0.5, 25.0, 10.0, 0.14)
        assert state.shape == (STATE_DIM,)

    def test_output_dtype(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(12, 0.5, 25.0, 10.0, 0.14)
        assert state.dtype == np.float32

    def test_all_values_in_unit_range(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(12, 0.5, 25.0, 10.0, 0.14)
        assert np.all(state >= 0.0) and np.all(state <= 1.0)

    def test_hour_normalization(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(23, 0.0, 0.0, 0.0, 0.0)
        assert state[0] == pytest.approx(1.0, abs=1e-6)

    def test_night_pv_zero(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(2, 0.5, 0.0, 5.0, 0.08)
        assert state[2] == 0.0

    @pytest.mark.boundary
    def test_clipping_above_max(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(12, 0.5, 9999.0, 10.0, 0.14)
        assert state[2] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.boundary
    def test_clipping_below_zero(self, normalizer: StateNormalizer) -> None:
        state = normalizer.normalize(12, -0.5, 25.0, 10.0, 0.14)
        assert state[1] == pytest.approx(0.0, abs=1e-6)

    def test_validate_passes_on_valid_state(
        self, normalizer: StateNormalizer, sample_state: np.ndarray
    ) -> None:
        assert normalizer.validate(sample_state) is True

    @pytest.mark.boundary
    def test_validate_raises_on_wrong_shape(self, normalizer: StateNormalizer) -> None:
        with pytest.raises(ValueError):
            normalizer.validate(np.zeros(3, dtype=np.float32))

    @pytest.mark.boundary
    def test_validate_raises_on_out_of_range(self, normalizer: StateNormalizer) -> None:
        with pytest.raises(ValueError):
            normalizer.validate(np.array([1.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32))

    @pytest.mark.boundary
    def test_validate_raises_on_wrong_dtype(self, normalizer: StateNormalizer) -> None:
        with pytest.raises(ValueError):
            normalizer.validate(np.zeros(STATE_DIM, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# TestQLearningAgent
# ─────────────────────────────────────────────────────────────────────────────


class TestQLearningAgent:

    def test_q_table_shape(self, q_agent: QLearningAgent) -> None:
        assert q_agent.q_table.shape == (5, 5, 5, 5, 5, NUM_ACTIONS)

    def test_discretize_output_type(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        result = q_agent._discretize(sample_state)
        assert isinstance(result, tuple) and len(result) == STATE_DIM

    @pytest.mark.boundary
    def test_discretize_bounds(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        indices = q_agent._discretize(sample_state)
        assert all(0 <= i <= q_agent.n_bins - 1 for i in indices)

    @pytest.mark.boundary
    def test_discretize_zero_state(self, q_agent: QLearningAgent) -> None:
        indices = q_agent._discretize(np.zeros(STATE_DIM, dtype=np.float32))
        assert all(i == 0 for i in indices)

    @pytest.mark.boundary
    def test_discretize_one_state(self, q_agent: QLearningAgent) -> None:
        indices = q_agent._discretize(np.ones(STATE_DIM, dtype=np.float32))
        assert all(i == q_agent.n_bins - 1 for i in indices)

    def test_select_action_valid_range(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        action = q_agent.select_action(sample_state)
        assert action in set(range(NUM_ACTIONS))

    def test_select_action_greedy_is_deterministic(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        actions = [q_agent.select_action(sample_state, training=False) for _ in range(10)]
        assert len(set(actions)) == 1

    def test_select_action_explores_all_actions_during_training(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        # epsilon=1.0 → pure random; 200 draws with 4 actions should cover all
        actions = {q_agent.select_action(sample_state, training=True) for _ in range(200)}
        assert actions == set(range(NUM_ACTIONS))

    def test_update_returns_td_error(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        td = q_agent.update(sample_state, 0, 1.0, sample_state, False)
        assert isinstance(td, float) and td >= 0.0

    @pytest.mark.physics
    def test_update_changes_q_value(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        idx = q_agent._discretize(sample_state)
        old_q = float(q_agent.q_table[idx][0])
        q_agent.update(sample_state, 0, 10.0, sample_state, False)
        assert q_agent.q_table[idx][0] != old_q

    def test_update_increments_training_step(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        before = q_agent.training_step
        q_agent.update(sample_state, 0, 1.0, sample_state, False)
        assert q_agent.training_step == before + 1

    @pytest.mark.physics
    def test_terminal_state_no_bootstrap(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        idx = q_agent._discretize(sample_state)
        old_q = float(q_agent.q_table[idx][0])
        td_returned = q_agent.update(sample_state, 0, 1.0, sample_state, done=True)
        expected_td = abs(1.0 - old_q)
        assert abs(td_returned - expected_td) < 1e-6

    def test_epsilon_decay(self, q_agent: QLearningAgent) -> None:
        before = q_agent.epsilon
        q_agent.decay_epsilon()
        assert q_agent.epsilon < before

    @pytest.mark.boundary
    def test_epsilon_floor(self) -> None:
        agent = QLearningAgent(epsilon=0.05, epsilon_min=0.05, epsilon_decay=0.995, n_bins=5, seed=0)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.05, abs=1e-9)

    def test_get_stats_keys(self, q_agent: QLearningAgent) -> None:
        stats = q_agent.get_stats()
        assert set(stats.keys()) == {
            "epsilon",
            "training_step",
            "mean_reward_last_10",
            "q_table_mean",
            "q_table_std",
        }

    def test_save_and_load(
        self, q_agent: QLearningAgent, sample_state: np.ndarray, tmp_path: pytest.TempPathFactory
    ) -> None:
        for _ in range(5):
            q_agent.update(sample_state, 0, 1.0, sample_state, False)
        save_path = str(tmp_path / "q.npz")
        q_agent.save(save_path)
        new_agent = QLearningAgent(n_bins=5, seed=42)
        new_agent.load(save_path)
        assert np.allclose(q_agent.q_table, new_agent.q_table)

    def test_record_episode_reward(self, q_agent: QLearningAgent) -> None:
        q_agent.record_episode_reward(5.0)
        q_agent.record_episode_reward(-3.5)
        assert len(q_agent.episode_rewards) == 2
        assert q_agent.episode_rewards[0] == pytest.approx(5.0)
        assert q_agent.episode_rewards[1] == pytest.approx(-3.5)

    @pytest.mark.numerical
    def test_no_nan_in_q_table_after_updates(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        rng = np.random.default_rng(7)
        for _ in range(50):
            reward = float(rng.uniform(-10.0, 10.0))
            q_agent.update(sample_state, int(rng.integers(0, NUM_ACTIONS)), reward, sample_state, False)
        assert np.all(np.isfinite(q_agent.q_table))


# ─────────────────────────────────────────────────────────────────────────────
# TestDQNAgent
# ─────────────────────────────────────────────────────────────────────────────


class TestDQNAgent:

    def test_select_action_valid_range(
        self, dqn_agent: DQNAgent, sample_state: np.ndarray
    ) -> None:
        action = dqn_agent.select_action(sample_state)
        assert action in set(range(NUM_ACTIONS))

    def test_update_returns_zero_before_buffer_full(
        self, dqn_agent: DQNAgent, sample_state: np.ndarray
    ) -> None:
        td = dqn_agent.update(sample_state, 0, 1.0, sample_state, False)
        assert td == pytest.approx(0.0)

    def test_update_returns_td_error_after_buffer_full(
        self, replay_filled_dqn: DQNAgent, sample_state: np.ndarray
    ) -> None:
        td = replay_filled_dqn.update(sample_state, 0, 1.0, sample_state, False)
        assert td >= 0.0

    @pytest.mark.boundary
    def test_replay_buffer_max_capacity(self) -> None:
        agent = DQNAgent(hidden_dim=32, batch_size=8, replay_capacity=100, seed=1)
        rng = np.random.default_rng(1)
        for _ in range(200):
            s = rng.random(STATE_DIM).astype(np.float32)
            ns = rng.random(STATE_DIM).astype(np.float32)
            agent.update(s, 0, 0.0, ns, False)
        assert len(agent._replay_buffer) == 100

    @pytest.mark.physics
    def test_target_net_syncs(
        self, replay_filled_dqn: DQNAgent, sample_state: np.ndarray
    ) -> None:
        rng = np.random.default_rng(99)
        for _ in range(replay_filled_dqn.target_update_freq):
            s = rng.random(STATE_DIM).astype(np.float32)
            ns = rng.random(STATE_DIM).astype(np.float32)
            replay_filled_dqn.update(s, 0, 1.0, ns, False)
        policy_params = dict(replay_filled_dqn._policy_net.named_parameters())
        for name, target_param in replay_filled_dqn._target_net.named_parameters():
            assert torch.allclose(target_param, policy_params[name]), (
                f"Parameter {name} not synced between target and policy net"
            )

    @pytest.mark.physics
    def test_policy_net_weights_change_after_training(
        self, replay_filled_dqn: DQNAgent, sample_state: np.ndarray
    ) -> None:
        snapshot = {
            name: param.data.clone()
            for name, param in replay_filled_dqn._policy_net.named_parameters()
        }
        rng = np.random.default_rng(77)
        for _ in range(5):
            s = rng.random(STATE_DIM).astype(np.float32)
            ns = rng.random(STATE_DIM).astype(np.float32)
            replay_filled_dqn.update(s, 0, 1.0, ns, False)
        changed = any(
            not torch.allclose(param.data, snapshot[name])
            for name, param in replay_filled_dqn._policy_net.named_parameters()
        )
        assert changed

    def test_greedy_action_deterministic(
        self, replay_filled_dqn: DQNAgent, sample_state: np.ndarray
    ) -> None:
        actions = [
            replay_filled_dqn.select_action(sample_state, training=False) for _ in range(10)
        ]
        assert len(set(actions)) == 1

    def test_epsilon_decay(self, dqn_agent: DQNAgent) -> None:
        before = dqn_agent.epsilon
        dqn_agent.decay_epsilon()
        assert dqn_agent.epsilon < before

    def test_save_and_load(
        self, dqn_agent: DQNAgent, sample_state: np.ndarray, tmp_path: pytest.TempPathFactory
    ) -> None:
        for _ in range(5):
            dqn_agent.update(sample_state, 0, 1.0, sample_state, False)
        save_path = str(tmp_path / "dqn.pt")
        dqn_agent.save(save_path)
        new_agent = DQNAgent(hidden_dim=32, batch_size=8, replay_capacity=100, seed=42)
        new_agent.load(save_path)
        original_action = dqn_agent.select_action(sample_state, training=False)
        loaded_action = new_agent.select_action(sample_state, training=False)
        assert original_action == loaded_action

    def test_get_stats_has_replay_buffer_size(self, dqn_agent: DQNAgent) -> None:
        assert "replay_buffer_size" in dqn_agent.get_stats()

    @pytest.mark.numerical
    def test_no_nan_in_network_output(
        self, replay_filled_dqn: DQNAgent, sample_state: np.ndarray
    ) -> None:
        with torch.no_grad():
            state_t = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0)
            q_vals = replay_filled_dqn._policy_net(state_t)
        assert torch.all(torch.isfinite(q_vals))


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentFactory
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentFactory:

    def test_creates_qlearning_agent(self) -> None:
        assert isinstance(create_agent("qlearning"), QLearningAgent)

    def test_creates_dqn_agent(self) -> None:
        assert isinstance(create_agent("dqn"), DQNAgent)

    @pytest.mark.boundary
    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            create_agent("ppo")

    def test_kwargs_forwarded(self) -> None:
        agent = create_agent("qlearning", n_bins=7)
        assert isinstance(agent, QLearningAgent)
        assert agent.q_table.shape[0] == 7


# ─────────────────────────────────────────────────────────────────────────────
# TestMDPProperties
# ─────────────────────────────────────────────────────────────────────────────


class TestMDPProperties:

    @pytest.mark.physics
    def test_q_values_improve_with_repeated_positive_reward(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        idx = q_agent._discretize(sample_state)
        initial_q = float(q_agent.q_table[idx][1])
        for _ in range(100):
            q_agent.update(sample_state, 1, 10.0, sample_state, done=False)
        assert q_agent.q_table[idx][1] > initial_q

    @pytest.mark.physics
    def test_q_values_suppress_negative_reward_action(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        for _ in range(100):
            q_agent.update(sample_state, 3, -10.0, sample_state, done=False)
        action = q_agent.select_action(sample_state, training=False)
        assert action != 3

    def test_agent_reaches_epsilon_min_after_decay(self) -> None:
        agent = QLearningAgent(
            n_bins=5, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99, seed=0
        )
        for _ in range(500):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(agent.epsilon_min, abs=1e-9)

    @pytest.mark.physics
    def test_state_normalizer_and_agent_pipeline(
        self,
        normalizer: StateNormalizer,
        q_agent: QLearningAgent,
    ) -> None:
        state = normalizer.normalize(12, 0.6, 30.0, 10.0, 0.14)
        action = q_agent.select_action(state)
        assert action in set(range(NUM_ACTIONS))

    @pytest.mark.numerical
    def test_no_nan_reward_propagation(
        self, q_agent: QLearningAgent, sample_state: np.ndarray
    ) -> None:
        q_agent.update(sample_state, 0, float("nan"), sample_state, False)
        assert np.all(np.isfinite(q_agent.q_table))
