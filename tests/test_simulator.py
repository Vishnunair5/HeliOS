"""Tests for simulator.py — Phase 3."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from physics_engine import BatteryBESS, PVArray
from agent import QLearningAgent
from simulator import (
    MicrogridSimulator,
    RewardCalculator,
    PowerRouter,
    StepResult,
)


# ── TestRewardCalculator ──────────────────────────────────────────────────────

class TestRewardCalculator:

    def _call(
        self,
        calc: RewardCalculator,
        grid_import_kw: float = 0.0,
        grid_export_kw: float = 0.0,
        grid_price: float = 0.14,
        soc: float = 0.5,
        battery_temp_k: float = 300.0,
        prev_battery_temp_k: float = 300.0,
        dt_hours: float = 1.0,
    ) -> dict[str, float]:
        return calc.compute(
            grid_import_kw=grid_import_kw,
            grid_export_kw=grid_export_kw,
            grid_price=grid_price,
            soc=soc,
            battery_temp_k=battery_temp_k,
            prev_battery_temp_k=prev_battery_temp_k,
            dt_hours=dt_hours,
        )

    def test_reward_keys(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc)
        assert set(result.keys()) == {
            "reward", "cost_import", "revenue_export", "soc_penalty", "thermal_penalty"
        }

    def test_zero_import_zero_cost(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, grid_import_kw=0.0, grid_export_kw=0.0)
        assert result["cost_import"] == 0.0

    @pytest.mark.physics
    def test_import_cost_scales_with_price(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, grid_import_kw=5.0, grid_price=0.20, dt_hours=1.0)
        assert abs(result["cost_import"] - 1.0) < 1e-6

    @pytest.mark.physics
    def test_export_revenue_positive(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, grid_export_kw=3.0)
        assert result["revenue_export"] > 0.0

    @pytest.mark.physics
    def test_soc_penalty_zero_within_bounds(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, soc=0.5, battery_temp_k=300.0, prev_battery_temp_k=300.0)
        assert result["soc_penalty"] == 0.0

    @pytest.mark.boundary
    def test_soc_penalty_nonzero_below_floor(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, soc=0.05)
        assert result["soc_penalty"] > 0.0

    @pytest.mark.boundary
    def test_soc_penalty_nonzero_above_ceiling(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, soc=0.98)
        assert result["soc_penalty"] > 0.0

    @pytest.mark.physics
    def test_thermal_penalty_zero_for_small_rise(self, reward_calc: RewardCalculator) -> None:
        result = self._call(
            reward_calc, battery_temp_k=301.0, prev_battery_temp_k=300.0
        )
        assert result["thermal_penalty"] == 0.0

    @pytest.mark.physics
    def test_thermal_penalty_nonzero_for_large_rise(self, reward_calc: RewardCalculator) -> None:
        result = self._call(
            reward_calc, battery_temp_k=305.0, prev_battery_temp_k=300.0
        )
        assert result["thermal_penalty"] > 0.0

    @pytest.mark.physics
    def test_reward_negative_under_heavy_import(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc, grid_import_kw=10.0, grid_price=0.24, dt_hours=1.0)
        assert result["reward"] < 0.0

    @pytest.mark.physics
    def test_reward_improves_with_export(self, reward_calc: RewardCalculator) -> None:
        r_no_export = self._call(reward_calc, grid_export_kw=0.0)
        r_with_export = self._call(reward_calc, grid_export_kw=5.0)
        assert r_with_export["reward"] > r_no_export["reward"]

    def test_explain_returns_string(self, reward_calc: RewardCalculator) -> None:
        result = self._call(reward_calc)
        s = reward_calc.explain(result)
        assert isinstance(s, str) and len(s) > 0

    @pytest.mark.numerical
    def test_no_nan_in_reward(self, reward_calc: RewardCalculator) -> None:
        result = self._call(
            reward_calc,
            grid_import_kw=50.0,
            soc=0.0,
            battery_temp_k=400.0,
            prev_battery_temp_k=300.0,
        )
        for v in result.values():
            assert math.isfinite(v)


# ── TestPowerRouter ───────────────────────────────────────────────────────────

class TestPowerRouter:

    @pytest.mark.physics
    def test_action0_charges_battery_on_pv_surplus(self, power_router: PowerRouter) -> None:
        flows = power_router.route(0, pv_kw=10.0, load_kw=3.0, grid_price=0.14)
        assert flows["battery_charge_kw"] > 0.0
        assert flows["grid_import_kw"] == 0.0

    @pytest.mark.physics
    def test_action0_imports_on_pv_deficit(self, power_router: PowerRouter) -> None:
        flows = power_router.route(0, pv_kw=1.0, load_kw=8.0, grid_price=0.14)
        assert flows["grid_import_kw"] > 0.0
        assert flows["battery_charge_kw"] == 0.0

    @pytest.mark.physics
    def test_action1_exports_on_pv_surplus(self, power_router: PowerRouter) -> None:
        flows = power_router.route(1, pv_kw=15.0, load_kw=5.0, grid_price=0.14)
        assert flows["grid_export_kw"] > 0.0
        assert flows["battery_charge_kw"] == 0.0

    @pytest.mark.physics
    def test_action1_imports_on_pv_deficit(self, power_router: PowerRouter) -> None:
        flows = power_router.route(1, pv_kw=2.0, load_kw=10.0, grid_price=0.14)
        assert flows["grid_import_kw"] > 0.0
        assert flows["grid_export_kw"] == 0.0

    @pytest.mark.physics
    def test_action2_discharges_battery(self, power_router: PowerRouter) -> None:
        flows = power_router.route(2, pv_kw=0.0, load_kw=5.0, grid_price=0.14)
        assert flows["battery_discharge_kw"] > 0.0

    @pytest.mark.boundary
    def test_action2_respects_soc_floor(self, bess_empty: BatteryBESS) -> None:
        router = PowerRouter(battery=bess_empty, dt_hours=1.0)
        flows = router.route(2, pv_kw=0.0, load_kw=20.0, grid_price=0.14)
        assert flows["soc"] >= bess_empty.soc_min - 1e-9

    @pytest.mark.physics
    def test_action3_charges_battery_from_grid(self, power_router: PowerRouter) -> None:
        flows = power_router.route(3, pv_kw=0.0, load_kw=5.0, grid_price=0.08)
        assert flows["battery_charge_kw"] > 0.0
        assert flows["grid_import_kw"] > 0.0

    @pytest.mark.numerical
    def test_all_power_flows_non_negative(self, bess: BatteryBESS) -> None:
        for action in range(4):
            router = PowerRouter(battery=BatteryBESS(soc_init=0.5), dt_hours=1.0)
            flows = router.route(action, pv_kw=5.0, load_kw=3.0, grid_price=0.14)
            assert flows["grid_import_kw"] >= 0.0
            assert flows["grid_export_kw"] >= 0.0
            assert flows["battery_charge_kw"] >= 0.0
            assert flows["battery_discharge_kw"] >= 0.0

    @pytest.mark.physics
    def test_max_charge_rate_tapers_near_full(self, power_router: PowerRouter) -> None:
        assert power_router._max_charge_rate(0.5) > power_router._max_charge_rate(0.90)

    @pytest.mark.physics
    def test_no_simultaneous_charge_and_discharge(self, bess: BatteryBESS) -> None:
        for action in range(4):
            router = PowerRouter(battery=BatteryBESS(soc_init=0.5), dt_hours=1.0)
            flows = router.route(action, pv_kw=5.0, load_kw=3.0, grid_price=0.14)
            assert not (
                flows["battery_charge_kw"] > 0.0 and flows["battery_discharge_kw"] > 0.0
            ), f"Action {action} produced simultaneous charge and discharge"


# ── TestMicrogridSimulator ────────────────────────────────────────────────────

class TestMicrogridSimulator:

    def test_reset_returns_valid_state(self, sim_q: MicrogridSimulator) -> None:
        state = sim_q.reset()
        assert state.shape == (5,)
        assert float(state.min()) >= 0.0
        assert float(state.max()) <= 1.0

    def test_reset_clears_log(
        self, completed_episode: list, sim_q: MicrogridSimulator
    ) -> None:
        sim_q.reset()
        assert len(sim_q._log) == 0

    def test_reset_resets_cumulative_counters(self, sim_q: MicrogridSimulator) -> None:
        sim_q.run_episode()
        sim_q.reset()
        assert sim_q._cumulative_cost == 0.0

    def test_step_returns_correct_types(self, sim_q: MicrogridSimulator) -> None:
        sim_q.reset()
        next_state, reward, done = sim_q.step(0)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_next_state_valid(self, sim_q: MicrogridSimulator) -> None:
        sim_q.reset()
        next_state, _, _ = sim_q.step(0)
        assert next_state.shape == (5,)
        assert float(next_state.min()) >= 0.0
        assert float(next_state.max()) <= 1.0

    def test_step_done_false_before_last(self, sim_q: MicrogridSimulator) -> None:
        sim_q.reset()
        _, _, done = sim_q.step(0)
        assert done is False

    def test_step_done_true_at_last_step(self, sim_q: MicrogridSimulator) -> None:
        sim_q.reset()
        done = False
        for _ in range(24):
            _, _, done = sim_q.step(0)
        assert done is True

    def test_run_episode_returns_24_steps(self, completed_episode: list) -> None:
        assert len(completed_episode) == 24

    @pytest.mark.numerical
    def test_step_result_fields_finite(self, completed_episode: list) -> None:
        from dataclasses import fields as dc_fields
        float_fields = [
            f.name for f in dc_fields(StepResult)
            if f.name not in ("action_description",)
            and f.name not in ("step", "hour", "action")
        ]
        for r in completed_episode:
            for fname in float_fields:
                val = getattr(r, fname)
                assert math.isfinite(float(val)), f"Non-finite value in {fname}: {val}"

    @pytest.mark.boundary
    def test_soc_stays_within_bounds_entire_episode(
        self, completed_episode: list
    ) -> None:
        for r in completed_episode:
            assert 0.10 - 1e-9 <= r.soc <= 0.95 + 1e-9, (
                f"SoC={r.soc:.4f} out of bounds at step {r.step}"
            )

    def test_all_actions_are_valid(self, completed_episode: list) -> None:
        for r in completed_episode:
            assert r.action in {0, 1, 2, 3}

    @pytest.mark.physics
    def test_pv_power_zero_at_night(self, completed_episode: list) -> None:
        night_hours = {0, 1, 2, 22, 23}
        for r in completed_episode:
            if r.hour in night_hours:
                assert r.pv_power_kw == pytest.approx(0.0, abs=1e-6), (
                    f"Non-zero PV at night hour {r.hour}: {r.pv_power_kw}"
                )

    @pytest.mark.physics
    def test_pv_power_positive_at_noon(self, completed_episode: list) -> None:
        noon_hours = {11, 12, 13}
        for r in completed_episode:
            if r.hour in noon_hours:
                assert r.pv_power_kw > 0.0, (
                    f"Zero PV at daytime hour {r.hour}"
                )

    def test_cumulative_cost_non_negative(self, completed_episode: list) -> None:
        for r in completed_episode:
            assert r.cumulative_cost >= 0.0

    @pytest.mark.physics
    def test_cumulative_cost_monotonically_non_decreasing(
        self, completed_episode: list
    ) -> None:
        for i in range(1, len(completed_episode)):
            assert completed_episode[i].cumulative_cost >= (
                completed_episode[i - 1].cumulative_cost - 1e-9
            ), f"Cumulative cost decreased at step {i}"

    def test_episode_summary_keys(
        self, sim_q: MicrogridSimulator, completed_episode: list
    ) -> None:
        summary = sim_q.get_episode_summary()
        expected = {
            "total_cost", "total_revenue", "net_cost",
            "total_pv_kwh", "total_load_kwh",
            "grid_import_kwh", "grid_export_kwh",
            "grid_dependence_pct", "final_soc", "final_soh",
            "mean_reward", "min_soc",
        }
        assert set(summary.keys()) == expected

    @pytest.mark.physics
    def test_grid_dependence_pct_in_range(self, sim_q: MicrogridSimulator) -> None:
        sim_q.run_episode()
        summary = sim_q.get_episode_summary()
        assert 0.0 <= summary["grid_dependence_pct"] <= 100.0

    @pytest.mark.physics
    def test_energy_balance_approximate(self, completed_episode: list) -> None:
        for r in completed_episode:
            lhs = r.pv_power_kw + r.grid_import_kw + r.battery_discharge_kw
            rhs = r.load_kw + r.grid_export_kw + r.battery_charge_kw
            assert abs(lhs - rhs) <= 0.5, (
                f"Energy balance violated at step {r.step}: "
                f"lhs={lhs:.3f}, rhs={rhs:.3f}, diff={abs(lhs-rhs):.3f}"
            )

    def test_save_log_creates_file(
        self, sim_q: MicrogridSimulator, completed_episode: list, tmp_path: object
    ) -> None:
        path = str(tmp_path / "test.csv")  # type: ignore[operator]
        sim_q.save_log(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_load_log_roundtrip(
        self, sim_q: MicrogridSimulator, completed_episode: list, tmp_path: object
    ) -> None:
        path = str(tmp_path / "roundtrip.csv")  # type: ignore[operator]
        sim_q.save_log(path)
        loaded = sim_q.load_log(path)
        assert len(loaded) == 24
        assert abs(loaded[0].reward - completed_episode[0].reward) < 1e-6

    def test_agent_training_step_increments(self, sim_q: MicrogridSimulator) -> None:
        agent: QLearningAgent = sim_q._agent  # type: ignore[assignment]
        before = agent.training_step
        sim_q.run_episode()
        assert agent.training_step == before + 24


# ── TestSimulatorIntegration ──────────────────────────────────────────────────

class TestSimulatorIntegration:

    @pytest.mark.integration
    def test_q_agent_net_cost_decreases_over_training(self) -> None:
        agent = QLearningAgent(n_bins=10, seed=0)
        sim = MicrogridSimulator(agent=agent, training=True, seed=0)

        sim.run_episode()
        first_net_cost = sim.get_episode_summary()["net_cost"]

        for _ in range(29):
            sim.run_episode()
        last_net_cost = sim.get_episode_summary()["net_cost"]

        assert last_net_cost <= first_net_cost + 0.5, (
            f"Net cost did not improve: ep1={first_net_cost:.4f}, ep30={last_net_cost:.4f}"
        )

    @pytest.mark.integration
    def test_dqn_agent_completes_full_episode(self, sim_dqn: MicrogridSimulator) -> None:
        results = sim_dqn.run_episode()
        assert len(results) == 24

    @pytest.mark.integration
    def test_simulator_physics_engine_agreement(self, sim_q: MicrogridSimulator) -> None:
        from physics_engine import EnvironmentProfiler, PVArray
        pv = PVArray()
        profiler = EnvironmentProfiler(dt_hours=1.0, seed=42)
        profiles = profiler.generate_all()

        results = sim_q.run_episode()
        for r in results:
            G = float(profiles["irradiance"][r.step])
            T = float(profiles["temperature"][r.step])
            expected_kw = pv.get_power(G, T) / 1000.0
            if expected_kw > 0.0:
                assert abs(r.pv_power_kw - expected_kw) / max(expected_kw, 1e-9) < 0.01, (
                    f"PV mismatch at step {r.step}: "
                    f"sim={r.pv_power_kw:.4f}, expected={expected_kw:.4f}"
                )

    @pytest.mark.integration
    def test_repeated_episodes_do_not_accumulate_state(
        self, sim_q: MicrogridSimulator
    ) -> None:
        for _ in range(3):
            sim_q.run_episode()
        assert len(sim_q._log) == 24
