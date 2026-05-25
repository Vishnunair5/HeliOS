import pytest
import numpy as np
from physics_engine import PVArray, BatteryBESS, EnvironmentProfiler


# ── Phase 1 fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def pv_array() -> PVArray:
    return PVArray()


@pytest.fixture
def bess() -> BatteryBESS:
    return BatteryBESS(soc_init=0.5)


@pytest.fixture
def bess_full() -> BatteryBESS:
    return BatteryBESS(soc_init=0.95)


@pytest.fixture
def bess_empty() -> BatteryBESS:
    return BatteryBESS(soc_init=0.10)


@pytest.fixture
def profiler() -> EnvironmentProfiler:
    return EnvironmentProfiler(dt_hours=1.0, seed=42)


@pytest.fixture
def env_profiles(profiler: EnvironmentProfiler) -> dict:
    return profiler.generate_all()


# ── Phase 2 fixtures ──────────────────────────────────────────────────────────

from agent import QLearningAgent, DQNAgent, StateNormalizer, create_agent, STATE_DIM


@pytest.fixture
def normalizer() -> StateNormalizer:
    return StateNormalizer()


@pytest.fixture
def q_agent() -> QLearningAgent:
    return QLearningAgent(n_bins=5, seed=42)


@pytest.fixture
def dqn_agent() -> DQNAgent:
    return DQNAgent(hidden_dim=32, batch_size=8, replay_capacity=100, seed=42)


@pytest.fixture
def sample_state() -> np.ndarray:
    """Mid-day state: hour=12, soc=0.6, pv=30kW, load=10kW, price=$0.14."""
    return np.array([12 / 23, 0.6, 30 / 50, 10 / 20, 0.14 / 0.30], dtype=np.float32)


@pytest.fixture
def night_state() -> np.ndarray:
    """Night state: hour=2, soc=0.4, pv=0kW, load=5kW, price=$0.08."""
    return np.array([2 / 23, 0.4, 0.0, 5 / 20, 0.08 / 0.30], dtype=np.float32)


@pytest.fixture
def replay_filled_dqn() -> DQNAgent:
    """DQNAgent with replay buffer pre-filled with 100 random transitions."""
    agent = DQNAgent(hidden_dim=32, batch_size=8, replay_capacity=100, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = rng.random(STATE_DIM).astype(np.float32)
        ns = rng.random(STATE_DIM).astype(np.float32)
        agent.update(
            s,
            int(rng.integers(0, 4)),
            float(rng.uniform(-1, 1)),
            ns,
            False,
        )
    return agent


# ── Phase 3 fixtures ──────────────────────────────────────────────────────────

from simulator import MicrogridSimulator, RewardCalculator, PowerRouter, StepResult


@pytest.fixture
def reward_calc() -> RewardCalculator:
    return RewardCalculator()


@pytest.fixture
def trained_q_agent() -> QLearningAgent:
    """A Q-agent pre-trained for 10 episodes on the simulator."""
    agent = QLearningAgent(n_bins=10, seed=0)
    sim = MicrogridSimulator(agent=agent, training=True, seed=0)
    for _ in range(10):
        sim.run_episode()
    return agent


@pytest.fixture
def sim_q(q_agent: QLearningAgent) -> MicrogridSimulator:
    """A fresh simulator backed by a Q-learning agent."""
    return MicrogridSimulator(agent=q_agent, training=True, seed=42)


@pytest.fixture
def sim_dqn(dqn_agent: DQNAgent) -> MicrogridSimulator:
    """A fresh simulator backed by a DQN agent."""
    return MicrogridSimulator(agent=dqn_agent, training=True, seed=42)


@pytest.fixture
def completed_episode(sim_q: MicrogridSimulator) -> list:
    """A fully run episode log (list of StepResult)."""
    return sim_q.run_episode()


@pytest.fixture
def power_router(bess: BatteryBESS) -> PowerRouter:
    return PowerRouter(battery=bess, dt_hours=1.0)


# ── Phase 4 fixtures ──────────────────────────────────────────────────────────

import pandas as pd
from dashboard import LogLoader, ChartBuilder, ScorecardBuilder


@pytest.fixture
def sample_log_df() -> pd.DataFrame:
    """A valid 24-row simulation log DataFrame matching StepResult fields."""
    hours = list(range(24))
    rng = np.random.default_rng(42)

    irr = np.array([max(0.0, 950 * np.exp(-0.5 * ((h - 13) / 3) ** 2)) for h in hours])
    pv = irr * 0.04

    df = pd.DataFrame({
        "step":                  hours,
        "hour":                  hours,
        "action":                rng.integers(0, 4, 24),
        "action_description":    ["PV->Load; excess->Battery"] * 24,
        "pv_power_kw":           pv,
        "load_kw":               rng.uniform(3, 12, 24),
        "grid_import_kw":        np.clip(rng.uniform(0, 5, 24), 0, None),
        "grid_export_kw":        np.clip(rng.uniform(0, 3, 24), 0, None),
        "battery_charge_kw":     np.clip(rng.uniform(0, 4, 24), 0, None),
        "battery_discharge_kw":  np.clip(rng.uniform(0, 4, 24), 0, None),
        "soc":                   np.clip(rng.uniform(0.15, 0.90, 24), 0.1, 0.95),
        "soh":                   np.linspace(1.0, 0.9995, 24),
        "battery_temp_k":        rng.uniform(298, 310, 24),
        "grid_price":            np.where(
                                     np.array(hours) < 9, 0.08,
                                     np.where(np.array(hours) < 21, 0.24, 0.08)
                                 ),
        "cost_grid_import":      rng.uniform(0, 0.5, 24),
        "revenue_grid_export":   rng.uniform(0, 0.1, 24),
        "soc_penalty":           np.zeros(24),
        "thermal_penalty":       np.zeros(24),
        "reward":                rng.uniform(-1, 0.5, 24),
        "td_error":              rng.uniform(0, 0.5, 24),
        "cumulative_cost":       np.cumsum(rng.uniform(0, 0.5, 24)),
        "cumulative_revenue":    np.cumsum(rng.uniform(0, 0.1, 24)),
    })
    return df


@pytest.fixture
def sample_log_csv(sample_log_df: pd.DataFrame, tmp_path) -> str:
    """Write the sample DataFrame to a temp CSV and return the path."""
    path = str(tmp_path / "test_sim.csv")
    sample_log_df.to_csv(path, index=False)
    return path


@pytest.fixture
def chart_builder() -> ChartBuilder:
    """A fresh ChartBuilder instance."""
    return ChartBuilder()


@pytest.fixture
def scorecard_builder() -> ScorecardBuilder:
    """A fresh ScorecardBuilder instance."""
    return ScorecardBuilder()


@pytest.fixture
def preprocessed_df(sample_log_df: pd.DataFrame) -> pd.DataFrame:
    """Sample DataFrame preprocessed through LogLoader.preprocess()."""
    loader = LogLoader.__new__(LogLoader)
    return loader.preprocess(sample_log_df)


# ── Phase 5 fixtures ──────────────────────────────────────────────────────────

import os
from main import TrainingConfig, TrainingMetrics, Trainer, Evaluator


@pytest.fixture
def default_config() -> TrainingConfig:
    """Minimal training config for fast tests."""
    return TrainingConfig(
        agent_type="qlearning",
        n_episodes=5,
        n_bins=5,
        seed=42,
        eval_episodes=2,
        launch_dashboard=False,
    )


@pytest.fixture
def dqn_config() -> TrainingConfig:
    """Minimal DQN training config for fast tests."""
    return TrainingConfig(
        agent_type="dqn",
        n_episodes=5,
        seed=0,
        eval_episodes=2,
        launch_dashboard=False,
    )


@pytest.fixture
def trained_trainer(default_config: TrainingConfig) -> Trainer:
    """A Trainer that has completed its full training run."""
    t = Trainer(default_config)
    t.run()
    return t


@pytest.fixture
def training_metrics_stub() -> TrainingMetrics:
    """Pre-filled TrainingMetrics with 20 episodes of synthetic data."""
    m = TrainingMetrics()
    rng = np.random.default_rng(0)
    for _ in range(20):
        m.record(
            summary={
                "mean_reward": float(rng.uniform(-1, 0.5)),
                "net_cost": float(rng.uniform(-0.5, 0.5)),
                "grid_dependence_pct": float(rng.uniform(10, 80)),
                "final_soc": float(rng.uniform(0.2, 0.9)),
            },
            epsilon=float(rng.uniform(0.05, 1.0)),
            mean_td=float(rng.uniform(0, 0.5)),
        )
    return m


# ── Marker registration ───────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "boundary: edge case and boundary condition tests")
    config.addinivalue_line("markers", "physics: physics or RL correctness tests")
    config.addinivalue_line("markers", "numerical: numerical stability and NaN/Inf tests")
    config.addinivalue_line("markers", "integration: cross-module integration tests")
