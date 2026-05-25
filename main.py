"""Entry point for the Solar Microgrid Power Optimizer.

Orchestrates training, evaluation, logging, and optional dashboard launch.
Only this module may import from all other project modules.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from agent import QLearningAgent, DQNAgent, create_agent
from simulator import MicrogridSimulator

log = logging.getLogger(__name__)

AgentType = QLearningAgent | DQNAgent

# ── Module-level constants ────────────────────────────────────────────────────
DEFAULT_EPISODES: int   = 100
DEFAULT_AGENT: str      = "qlearning"
DEFAULT_N_BINS: int     = 10
DEFAULT_DT_HOURS: float = 1.0
DEFAULT_SEED: int       = 42
EVAL_EPISODES: int      = 5
PRINT_EVERY: int        = 10


# ── TrainingConfig ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Typed, validated configuration container for a training run."""

    agent_type: str        = DEFAULT_AGENT
    n_episodes: int        = DEFAULT_EPISODES
    n_bins: int            = DEFAULT_N_BINS
    dt_hours: float        = DEFAULT_DT_HOURS
    seed: int              = DEFAULT_SEED
    eval_episodes: int     = EVAL_EPISODES
    launch_dashboard: bool = False
    checkpoint_dir: str    = "./checkpoints"
    log_dir: str           = "./logs"

    def validate(self) -> None:
        """Validate all configuration values.

        Raises:
            ValueError: If any field is outside its valid range.
        """
        if self.agent_type not in ("qlearning", "dqn"):
            raise ValueError(
                f"agent_type must be 'qlearning' or 'dqn', got '{self.agent_type}'"
            )
        if self.n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {self.n_episodes}")
        if self.dt_hours not in (1.0, 0.25):
            raise ValueError(f"dt_hours must be 1.0 or 0.25, got {self.dt_hours}")
        if not (0 < self.n_bins <= 50):
            raise ValueError(f"n_bins must be in (0, 50], got {self.n_bins}")
        if self.eval_episodes < 1:
            raise ValueError(f"eval_episodes must be >= 1, got {self.eval_episodes}")


# ── TrainingMetrics ───────────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    """Accumulates per-episode training statistics."""

    episode_rewards:   list[float] = field(default_factory=list)
    episode_net_costs: list[float] = field(default_factory=list)
    episode_grid_dep:  list[float] = field(default_factory=list)
    episode_final_soc: list[float] = field(default_factory=list)
    epsilon_history:   list[float] = field(default_factory=list)
    td_error_history:  list[float] = field(default_factory=list)

    def record(
        self,
        summary: dict[str, float],
        epsilon: float,
        mean_td: float,
    ) -> None:
        """Append one episode's metrics to all series.

        Args:
            summary: Dict returned by MicrogridSimulator.get_episode_summary().
            epsilon: Current agent exploration probability.
            mean_td: Mean absolute TD error for the episode.
        """
        self.episode_rewards.append(float(summary["mean_reward"]))
        self.episode_net_costs.append(float(summary["net_cost"]))
        self.episode_grid_dep.append(float(summary["grid_dependence_pct"]))
        self.episode_final_soc.append(float(summary["final_soc"]))
        self.epsilon_history.append(float(epsilon))
        self.td_error_history.append(float(mean_td))

    def rolling_mean(self, series: list[float], window: int = 10) -> list[float]:
        """Compute a causal rolling mean over the given series.

        Args:
            series: Sequence of float values.
            window: Number of preceding values to average (inclusive).

        Returns:
            List of the same length as series with rolling mean values.
        """
        result: list[float] = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            result.append(float(np.mean(series[start : i + 1])))
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Serialize all metric lists to a DataFrame.

        Returns:
            DataFrame with 6 columns and one row per recorded episode.
        """
        return pd.DataFrame({
            "episode_reward":    self.episode_rewards,
            "net_cost":          self.episode_net_costs,
            "grid_dependence_pct": self.episode_grid_dep,
            "final_soc":         self.episode_final_soc,
            "epsilon":           self.epsilon_history,
            "mean_td_error":     self.td_error_history,
        })

    def save(self, path: str) -> None:
        """Write the metrics DataFrame to a CSV file.

        Args:
            path: Destination file path. Parent directory is created if absent.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        log.info("Training metrics saved to %s", path)

    def summary_stats(self) -> dict[str, float]:
        """Return final training summary statistics.

        Returns:
            Dict with keys: best_net_cost, worst_net_cost, mean_net_cost_last_10,
            best_mean_reward, mean_grid_dep_last_10, final_epsilon,
            total_td_error_mean.
        """
        costs = self.episode_net_costs
        rewards = self.episode_rewards
        td = self.td_error_history
        grid_dep = self.episode_grid_dep
        eps = self.epsilon_history
        return {
            "best_net_cost":        float(min(costs)) if costs else 0.0,
            "worst_net_cost":       float(max(costs)) if costs else 0.0,
            "mean_net_cost_last_10": float(np.mean(costs[-10:])) if costs else 0.0,
            "best_mean_reward":     float(max(rewards)) if rewards else 0.0,
            "mean_grid_dep_last_10": float(np.mean(grid_dep[-10:])) if grid_dep else 0.0,
            "final_epsilon":        float(eps[-1]) if eps else 1.0,
            "total_td_error_mean":  float(np.mean(td)) if td else 0.0,
        }


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full multi-episode training loop."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialise trainer, agent, simulator, and metrics accumulator.

        Args:
            config: Validated training configuration.
        """
        self._config = config
        self._logger = logging.getLogger(__name__)

        if config.agent_type == "qlearning":
            self._agent: AgentType = QLearningAgent(
                n_bins=config.n_bins,
                alpha=0.3,
                epsilon_decay=0.9,
                seed=config.seed,
            )
        else:
            self._agent = DQNAgent(seed=config.seed)

        self._sim = MicrogridSimulator(
            agent=self._agent,
            dt_hours=config.dt_hours,
            training=True,
            seed=config.seed,
        )
        self._metrics = TrainingMetrics()

    def run(self) -> TrainingMetrics:
        """Execute the full training loop across all configured episodes.

        Returns:
            Populated TrainingMetrics instance with one entry per episode.
        """
        os.makedirs(self._config.checkpoint_dir, exist_ok=True)

        for episode in range(1, self._config.n_episodes + 1):
            results = self._sim.run_episode()
            summary = self._sim.get_episode_summary()
            mean_td = float(np.mean([r.td_error for r in results]))

            self._metrics.record(
                summary=summary,
                epsilon=self._agent.get_stats()["epsilon"],
                mean_td=mean_td,
            )

            if episode % PRINT_EVERY == 0:
                self._log_progress(episode, summary)

            if episode % 25 == 0:
                self._save_checkpoint(episode)

        return self._metrics

    def _log_progress(self, episode: int, summary: dict[str, float]) -> None:
        """Log a single formatted progress line at INFO level.

        Args:
            episode: Current episode number (1-indexed).
            summary: Episode summary from MicrogridSimulator.get_episode_summary().
        """
        self._logger.info(
            "Episode %3d/%d | NetCost=$%.2f | GridDep=%.1f%% | ε=%.3f | MeanR=%.4f",
            episode,
            self._config.n_episodes,
            summary["net_cost"],
            summary["grid_dependence_pct"],
            self._agent.get_stats()["epsilon"],
            summary["mean_reward"],
        )

    def _save_checkpoint(self, episode: int) -> None:
        """Persist agent weights to a timestamped checkpoint file.

        Args:
            episode: Current episode number used in the filename.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".npz" if isinstance(self._agent, QLearningAgent) else ".pt"
        fname = f"{self._config.agent_type}_ep{episode:04d}_{ts}{ext}"
        path = os.path.join(self._config.checkpoint_dir, fname)
        self._agent.save(path)
        self._logger.info("Checkpoint saved: %s", path)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """Runs the trained agent in greedy mode and reports evaluation results."""

    def __init__(
        self,
        agent: AgentType,
        config: TrainingConfig,
    ) -> None:
        """Initialise evaluator with a trained agent and configuration.

        Args:
            agent: Trained QLearningAgent or DQNAgent instance.
            config: Training configuration (seed, dt_hours, eval_episodes).
        """
        self._agent = agent
        self._config = config
        self._logger = logging.getLogger(__name__)

    def evaluate(self) -> dict[str, float]:
        """Run eval_episodes greedy episodes and return averaged metrics.

        The agent is NOT updated or modified during evaluation. Each episode
        uses an identical seeded environment for reproducibility.

        Returns:
            Dict of averaged get_episode_summary() keys plus 'eval_episodes'.
        """
        sim = MicrogridSimulator(
            agent=self._agent,
            dt_hours=self._config.dt_hours,
            training=False,
            seed=self._config.seed,
        )

        all_summaries: list[dict[str, float]] = []
        for _ in range(self._config.eval_episodes):
            state = sim.reset()
            for _ in range(sim._n_steps):
                action = self._agent.select_action(state, training=False)
                next_state, _reward, _done = sim.step(action)
                state = next_state
            all_summaries.append(sim.get_episode_summary())

        keys = list(all_summaries[0].keys())
        avg: dict[str, float] = {
            k: float(np.mean([s[k] for s in all_summaries]))
            for k in keys
        }
        avg["eval_episodes"] = float(self._config.eval_episodes)
        self._logger.info(
            "Evaluation complete: %d episodes | avg_net_cost=%.4f",
            self._config.eval_episodes,
            avg.get("net_cost", 0.0),
        )
        return avg

    def print_report(
        self,
        metrics: dict[str, float],
        training_metrics: TrainingMetrics,
    ) -> None:
        """Print a formatted final report to stdout.

        This is the only sanctioned multi-line print() call outside __main__.

        Args:
            metrics: Dict returned by evaluate().
            training_metrics: Populated TrainingMetrics from a training run.
        """
        W = 54
        sep = "-" * W
        ts = training_metrics.summary_stats()

        def _head(text: str) -> str:
            return f"|{text:^{W}}|"

        def _row(label: str, value: str) -> str:
            content = f"  {label:<18}: {value}"
            return f"|{content:<{W}}|"

        def _section(text: str) -> str:
            return f"|  {text:<{W - 2}}|"

        total_pv = metrics.get("total_pv_kwh", 0.0)
        grid_export = metrics.get("grid_export_kwh", 0.0)
        pv_self_use = float(np.clip(
            (total_pv - grid_export) / max(total_pv, 1e-9) * 100.0,
            0.0, 100.0,
        ))

        print(f"+{sep}+")
        print(_head("SOLAR MICROGRID OPTIMIZER - FINAL REPORT"))
        print(f"+{sep}+")
        print(_row("Agent Type", self._config.agent_type))
        print(_row("Training Episodes", str(self._config.n_episodes)))
        print(_row("Eval Episodes", str(self._config.eval_episodes)))
        print(f"+{sep}+")
        print(_section("TRAINING SUMMARY"))
        print(_row("Best Net Cost", f"${ts['best_net_cost']:.2f}"))
        print(_row("Mean Net Cost(10)", f"${ts['mean_net_cost_last_10']:.2f}"))
        print(_row("Final Epsilon", f"{ts['final_epsilon']:.3f}"))
        print(f"+{sep}+")
        eval_label = f"EVALUATION RESULTS (greedy, {self._config.eval_episodes} episodes)"
        print(_head(eval_label))
        print(_row("Avg Net Cost", f"${metrics.get('net_cost', 0.0):.2f}"))
        print(_row("Avg Grid Dep.", f"{metrics.get('grid_dependence_pct', 0.0):.1f}%"))
        print(_row("Avg PV Self-Use", f"{pv_self_use:.1f}%"))
        print(_row("Avg Final SoC", f"{metrics.get('final_soc', 0.0):.2f}"))
        print(f"+{sep}+")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> TrainingConfig:
    """Parse CLI arguments and return a validated TrainingConfig.

    Returns:
        A fully validated TrainingConfig instance.
    """
    parser = argparse.ArgumentParser(
        description="Solar Microgrid Power Optimizer — Training & Evaluation"
    )
    parser.add_argument("--agent",     type=str,   default=DEFAULT_AGENT)
    parser.add_argument("--episodes",  type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--bins",      type=int,   default=DEFAULT_N_BINS)
    parser.add_argument("--dt",        type=float, default=DEFAULT_DT_HOURS)
    parser.add_argument("--seed",      type=int,   default=DEFAULT_SEED)
    parser.add_argument("--eval",      type=int,   default=EVAL_EPISODES)
    parser.add_argument("--dashboard", action="store_true")

    args = parser.parse_args()
    config = TrainingConfig(
        agent_type=args.agent,
        n_episodes=args.episodes,
        n_bins=args.bins,
        dt_hours=args.dt,
        seed=args.seed,
        eval_episodes=args.eval,
        launch_dashboard=args.dashboard,
    )
    config.validate()
    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Full system entry point: train, evaluate, log, optionally launch dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _log = logging.getLogger(__name__)

    config = parse_args()
    _log.info("Starting training: %s", config)

    # Training
    trainer = Trainer(config)
    training_metrics = trainer.run()

    os.makedirs(config.log_dir, exist_ok=True)
    metrics_path = os.path.join(
        config.log_dir,
        f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    training_metrics.save(metrics_path)
    _log.info("Training metrics saved: %s", metrics_path)

    # Evaluation
    evaluator = Evaluator(agent=trainer._agent, config=config)
    eval_metrics = evaluator.evaluate()

    eval_log_path = trainer._sim.save_log()
    _log.info("Final eval log saved: %s", eval_log_path)

    # Report
    evaluator.print_report(eval_metrics, training_metrics)

    # Dashboard
    if config.launch_dashboard:
        _log.info("Launching Streamlit dashboard...")
        os.system(f"streamlit run dashboard.py -- {eval_log_path}")


# ── Phase 5 inline validation ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _log = logging.getLogger(__name__)

    _log.info("=== Phase 5 Integration Validation ===")

    config = TrainingConfig(
        agent_type="qlearning",
        n_episodes=10,
        n_bins=10,
        seed=42,
        eval_episodes=3,
        launch_dashboard=False,
    )
    config.validate()
    _log.info("Config validated: %s", config)

    trainer = Trainer(config)
    metrics = trainer.run()
    _log.info("Training complete. Summary: %s", metrics.summary_stats())

    assert len(metrics.episode_rewards) == 10
    assert metrics.summary_stats()["final_epsilon"] < 1.0
    assert all(x == x for x in metrics.episode_rewards)  # no NaN

    evaluator = Evaluator(agent=trainer._agent, config=config)
    eval_metrics = evaluator.evaluate()
    evaluator.print_report(eval_metrics, metrics)

    assert eval_metrics["grid_dependence_pct"] >= 0.0
    assert 0.0 < eval_metrics["final_soc"] <= 1.0

    log_path = trainer._sim.save_log()
    _log.info("Final log: %s", log_path)
    assert os.path.exists(log_path)

    from dashboard import LogLoader, ChartBuilder, ScorecardBuilder

    df = LogLoader(log_path).load()
    assert len(df) == 24
    builder = ChartBuilder()
    figs = [
        builder.build_generation_vs_load(df),
        builder.build_battery_soc(df),
        builder.build_cumulative_economics(df),
        builder.build_reward_trace(df),
        builder.build_action_distribution(df),
        builder.build_grid_price_overlay(df),
    ]
    assert all(len(f.data) > 0 for f in figs), "One or more charts empty"
    _log.info("All %d charts validated.", len(figs))

    sc = ScorecardBuilder()
    sc_metrics = sc.compute_metrics(df)
    assert all(isinstance(v, (int, float, str)) for v in sc_metrics.values())

    print("\nPhase 5 validation PASSED")
    print("Run 'python main.py --episodes 100 --dashboard' for full training.")
