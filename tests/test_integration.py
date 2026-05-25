"""Phase 5 end-to-end integration tests.

Every test here crosses at least two module boundaries.
All tests carry @pytest.mark.integration. Secondary markers applied where relevant.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from main import (
    TrainingConfig,
    TrainingMetrics,
    Trainer,
    Evaluator,
    DEFAULT_EPISODES,
    DEFAULT_AGENT,
)


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestTrainingConfig:
    """Validation logic for TrainingConfig."""

    def test_default_config_validates(self) -> None:
        TrainingConfig().validate()  # must not raise

    @pytest.mark.boundary
    def test_invalid_agent_type_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(agent_type="ppo").validate()

    @pytest.mark.boundary
    def test_zero_episodes_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(n_episodes=0).validate()

    @pytest.mark.boundary
    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(dt_hours=0.5).validate()

    @pytest.mark.boundary
    def test_invalid_bins_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(n_bins=0).validate()

    @pytest.mark.boundary
    def test_invalid_eval_episodes_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingConfig(eval_episodes=0).validate()


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestTrainingMetrics:
    """Unit-level tests for TrainingMetrics (wired through Trainer calls)."""

    def test_record_appends_all_series(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        for lst in (
            training_metrics_stub.episode_rewards,
            training_metrics_stub.episode_net_costs,
            training_metrics_stub.episode_grid_dep,
            training_metrics_stub.episode_final_soc,
            training_metrics_stub.epsilon_history,
            training_metrics_stub.td_error_history,
        ):
            assert len(lst) == 20

    def test_rolling_mean_correct_length(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        result = training_metrics_stub.rolling_mean(
            training_metrics_stub.episode_rewards, window=5
        )
        assert len(result) == 20

    @pytest.mark.numerical
    def test_rolling_mean_values_finite(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        result = training_metrics_stub.rolling_mean(
            training_metrics_stub.episode_rewards, window=5
        )
        assert all(np.isfinite(v) for v in result)

    def test_to_dataframe_shape(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        df = training_metrics_stub.to_dataframe()
        assert df.shape == (20, 6)

    def test_save_creates_file(
        self, training_metrics_stub: TrainingMetrics, tmp_path: pytest.TempPathFactory
    ) -> None:
        path = str(tmp_path / "metrics.csv")
        training_metrics_stub.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_summary_stats_keys(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        stats = training_metrics_stub.summary_stats()
        expected = {
            "best_net_cost", "worst_net_cost", "mean_net_cost_last_10",
            "best_mean_reward", "mean_grid_dep_last_10",
            "final_epsilon", "total_td_error_mean",
        }
        assert expected.issubset(stats.keys())

    @pytest.mark.numerical
    def test_summary_stats_finite(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        stats = training_metrics_stub.summary_stats()
        for k, v in stats.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Non-finite stat: {k}={v}"

    @pytest.mark.physics
    def test_best_net_cost_is_minimum(
        self, training_metrics_stub: TrainingMetrics
    ) -> None:
        stats = training_metrics_stub.summary_stats()
        assert stats["best_net_cost"] == min(
            training_metrics_stub.episode_net_costs
        )


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestTrainer:
    """Tests for the multi-episode training loop."""

    def test_trainer_runs_correct_episode_count(
        self, default_config: TrainingConfig
    ) -> None:
        metrics = Trainer(default_config).run()
        assert len(metrics.episode_rewards) == default_config.n_episodes

    def test_trainer_creates_checkpoint_dir(
        self, default_config: TrainingConfig, tmp_path: pytest.TempPathFactory
    ) -> None:
        default_config.checkpoint_dir = str(tmp_path / "ckpts")
        Trainer(default_config).run()
        assert os.path.isdir(default_config.checkpoint_dir)

    def test_trainer_saves_checkpoint(
        self, default_config: TrainingConfig, tmp_path: pytest.TempPathFactory
    ) -> None:
        default_config.n_episodes = 25
        default_config.checkpoint_dir = str(tmp_path / "ckpts")
        Trainer(default_config).run()
        files = os.listdir(default_config.checkpoint_dir)
        ckpt_files = [f for f in files if f.endswith(".npz") or f.endswith(".pt")]
        assert len(ckpt_files) >= 1

    def test_trainer_metrics_record_per_episode(
        self, trained_trainer: Trainer
    ) -> None:
        m = trained_trainer._metrics
        n = trained_trainer._config.n_episodes
        for lst in (
            m.episode_rewards, m.episode_net_costs, m.episode_grid_dep,
            m.episode_final_soc, m.epsilon_history, m.td_error_history,
        ):
            assert len(lst) == n

    @pytest.mark.physics
    def test_trainer_epsilon_decreases_monotonically(
        self, trained_trainer: Trainer
    ) -> None:
        eps = trained_trainer._metrics.epsilon_history
        for i in range(1, len(eps)):
            assert eps[i] <= eps[i - 1] + 1e-9, (
                f"Epsilon increased at episode {i}: {eps[i-1]} → {eps[i]}"
            )

    @pytest.mark.physics
    def test_trainer_net_cost_trend(self, default_config: TrainingConfig) -> None:
        default_config.n_episodes = 50
        default_config.n_bins = 10
        metrics = Trainer(default_config).run()
        first_10 = float(np.mean(metrics.episode_net_costs[:10]))
        last_10 = float(np.mean(metrics.episode_net_costs[-10:]))
        assert last_10 <= first_10, (
            f"No improvement: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )

    @pytest.mark.numerical
    def test_trainer_no_nan_in_metrics(self, trained_trainer: Trainer) -> None:
        m = trained_trainer._metrics
        for lst in (
            m.episode_rewards, m.episode_net_costs, m.episode_grid_dep,
            m.episode_final_soc, m.epsilon_history, m.td_error_history,
        ):
            assert all(np.isfinite(v) for v in lst)

    def test_dqn_trainer_completes(self, dqn_config: TrainingConfig) -> None:
        metrics = Trainer(dqn_config).run()
        assert len(metrics.episode_rewards) == dqn_config.n_episodes


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestEvaluator:
    """Tests for the greedy evaluation loop."""

    def test_evaluator_runs_correct_episodes(
        self, trained_trainer: Trainer, default_config: TrainingConfig
    ) -> None:
        ev = Evaluator(trained_trainer._agent, default_config)
        m = ev.evaluate()
        assert int(m["eval_episodes"]) == default_config.eval_episodes

    @pytest.mark.numerical
    def test_evaluator_metrics_finite(
        self, trained_trainer: Trainer, default_config: TrainingConfig
    ) -> None:
        m = Evaluator(trained_trainer._agent, default_config).evaluate()
        for k, v in m.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Non-finite eval metric: {k}={v}"

    @pytest.mark.physics
    def test_evaluator_grid_dep_in_range(
        self, trained_trainer: Trainer, default_config: TrainingConfig
    ) -> None:
        m = Evaluator(trained_trainer._agent, default_config).evaluate()
        assert 0.0 <= m["grid_dependence_pct"] <= 100.0

    def test_evaluator_greedy_produces_consistent_results(
        self, trained_trainer: Trainer, default_config: TrainingConfig
    ) -> None:
        ev = Evaluator(trained_trainer._agent, default_config)
        m1 = ev.evaluate()
        m2 = ev.evaluate()
        assert abs(m1["net_cost"] - m2["net_cost"]) < 1e-3, (
            f"Greedy eval not deterministic: {m1['net_cost']:.6f} vs {m2['net_cost']:.6f}"
        )

    def test_print_report_does_not_raise(
        self,
        trained_trainer: Trainer,
        default_config: TrainingConfig,
        training_metrics_stub: TrainingMetrics,
        capsys: pytest.CaptureFixture,
    ) -> None:
        ev = Evaluator(trained_trainer._agent, default_config)
        eval_metrics = ev.evaluate()
        ev.print_report(eval_metrics, training_metrics_stub)
        captured = capsys.readouterr()
        assert "FINAL REPORT" in captured.out
        assert "EVALUATION RESULTS" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestEndToEnd:
    """Full pipeline tests crossing all five module boundaries."""

    def test_full_qlearning_pipeline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from dashboard import LogLoader, ChartBuilder

        config = TrainingConfig(
            agent_type="qlearning",
            n_episodes=10,
            n_bins=5,
            seed=0,
            eval_episodes=2,
            log_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        metrics = trainer.run()
        evaluator = Evaluator(trainer._agent, config)
        eval_m = evaluator.evaluate()
        log_path = trainer._sim.save_log(str(tmp_path / "final.csv"))

        df = LogLoader(log_path).load()
        charts = [
            ChartBuilder().build_generation_vs_load(df),
            ChartBuilder().build_battery_soc(df),
            ChartBuilder().build_cumulative_economics(df),
        ]
        assert all(len(fig.data) > 0 for fig in charts)
        assert eval_m["grid_dependence_pct"] >= 0.0
        assert metrics.summary_stats()["final_epsilon"] < 1.0

    def test_full_dqn_pipeline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from dashboard import LogLoader, ChartBuilder

        config = TrainingConfig(
            agent_type="dqn",
            n_episodes=5,
            seed=0,
            eval_episodes=2,
            log_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        metrics = trainer.run()
        evaluator = Evaluator(trainer._agent, config)
        eval_m = evaluator.evaluate()
        log_path = trainer._sim.save_log(str(tmp_path / "dqn_final.csv"))

        df = LogLoader(log_path).load()
        assert len(df) == 24
        assert eval_m["grid_dependence_pct"] >= 0.0
        assert len(metrics.episode_rewards) == config.n_episodes

    def test_log_csv_schema_matches_dashboard_expectations(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from dashboard import LogLoader, COL_HOUR, COL_PV, COL_LOAD, COL_GRID_IMPORT
        from dashboard import COL_GRID_EXPORT, COL_BATT_CHARGE, COL_BATT_DISCHARGE
        from dashboard import COL_SOC, COL_SOH, COL_REWARD, COL_CUMCOST
        from dashboard import COL_CUMREV, COL_GRID_PRICE, COL_ACTION, COL_ACTION_DESC

        config = TrainingConfig(
            agent_type="qlearning",
            n_episodes=3,
            n_bins=5,
            seed=7,
            log_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        trainer.run()
        log_path = trainer._sim.save_log(str(tmp_path / "schema_test.csv"))

        loader = LogLoader(log_path)
        df_raw = loader.load()
        assert loader.validate(df_raw) is True

        required_cols = [
            COL_HOUR, COL_PV, COL_LOAD, COL_GRID_IMPORT, COL_GRID_EXPORT,
            COL_BATT_CHARGE, COL_BATT_DISCHARGE, COL_SOC, COL_SOH,
            COL_REWARD, COL_CUMCOST, COL_CUMREV, COL_GRID_PRICE,
            COL_ACTION, COL_ACTION_DESC,
        ]
        for col in required_cols:
            assert col in df_raw.columns, f"Missing column: {col}"

    def test_agent_outperforms_random_baseline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from simulator import MicrogridSimulator
        from agent import QLearningAgent

        # Train agent for 50 episodes
        config = TrainingConfig(
            agent_type="qlearning",
            n_episodes=50,
            n_bins=10,
            seed=42,
            eval_episodes=5,
            log_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        trainer.run()
        ev = Evaluator(trainer._agent, config)
        trained_result = ev.evaluate()
        trained_net_cost = trained_result["net_cost"]

        # Random baseline: QLearningAgent fixed at epsilon=1.0 (fully random)
        random_agent = QLearningAgent(n_bins=10, seed=99)
        random_agent.epsilon = 1.0
        random_agent.epsilon_decay = 1.0  # no decay
        random_agent.epsilon_min = 1.0

        sim = MicrogridSimulator(agent=random_agent, training=True, seed=42)
        random_costs: list[float] = []
        for _ in range(5):
            sim.run_episode()
            s = sim.get_episode_summary()
            random_costs.append(s["net_cost"])
        random_net_cost = float(np.mean(random_costs))

        assert trained_net_cost <= random_net_cost, (
            f"Trained ({trained_net_cost:.4f}) not better than random ({random_net_cost:.4f})"
        )

    def test_checkpoint_restore_preserves_policy(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from agent import QLearningAgent

        config = TrainingConfig(
            agent_type="qlearning",
            n_episodes=10,
            n_bins=5,
            seed=42,
            eval_episodes=3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        trainer.run()

        # Save checkpoint manually
        ckpt_path = str(tmp_path / "restore_test.npz")
        trainer._agent.save(ckpt_path)

        # Original agent evaluation
        ev_orig = Evaluator(trainer._agent, config)
        orig_result = ev_orig.evaluate()

        # Restored agent
        restored = QLearningAgent(n_bins=5, seed=42)
        restored.load(ckpt_path)
        ev_restored = Evaluator(restored, config)
        restored_result = ev_restored.evaluate()

        assert abs(orig_result["net_cost"] - restored_result["net_cost"]) < 1e-3, (
            f"Checkpoint restore diverged: {orig_result['net_cost']:.6f} vs "
            f"{restored_result['net_cost']:.6f}"
        )

    def test_simulation_log_roundtrip_preserves_rewards(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        config = TrainingConfig(
            agent_type="qlearning",
            n_episodes=1,
            n_bins=5,
            seed=42,
            log_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        trainer.run()
        original_log = list(trainer._sim._log)

        log_path = trainer._sim.save_log(str(tmp_path / "roundtrip.csv"))
        reloaded = trainer._sim.load_log(log_path)

        assert len(reloaded) == len(original_log)
        for orig, loaded in zip(original_log, reloaded):
            assert abs(orig.reward - loaded.reward) < 1e-6, (
                f"Reward mismatch at step {orig.step}: {orig.reward} vs {loaded.reward}"
            )

    @pytest.mark.physics
    def test_physics_consistency_across_full_run(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from dashboard import LogLoader

        config = TrainingConfig(
            agent_type="qlearning",
            n_episodes=5,
            n_bins=5,
            seed=42,
            log_dir=str(tmp_path),
        )
        trainer = Trainer(config)
        trainer.run()
        log_path = trainer._sim.save_log(str(tmp_path / "physics.csv"))
        df = LogLoader(log_path).load()

        assert (df["pv_power_kw"] >= 0).all(), "Negative PV power"
        assert (df["grid_import_kw"] >= 0).all(), "Negative grid import"
        assert (df["soc"] >= 0.09).all(), "SoC below floor"
        assert (df["soc"] <= 0.96).all(), "SoC above ceiling"
        assert (df["soh"] > 0.0).all(), "Non-positive SoH"
        assert (df["soh"] <= 1.0).all(), "SoH above 1.0"

    def test_all_modules_importable_together(self) -> None:
        import physics_engine  # noqa: F401
        import agent            # noqa: F401
        import simulator        # noqa: F401
        import dashboard        # noqa: F401
        import main             # noqa: F401
