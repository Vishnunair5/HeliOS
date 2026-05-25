"""Central orchestration layer for the solar microgrid simulation.

Wires together physics_engine and agent modules. Owns the 24-hour step loop,
reward computation, power routing, and structured logging.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from typing import Union

import numpy as np

from physics_engine import PVArray, BatteryBESS, EnvironmentProfiler
from agent import (
    QLearningAgent,
    DQNAgent,
    StateNormalizer,
    ACTION_DESCRIPTIONS,
    STATE_DIM,
)

# Module-level constants
LOG_DIR: str = "./logs"
CHECKPOINT_DIR: str = "./checkpoints"
MAX_PV_KW: float = 50.0
MAX_LOAD_KW: float = 20.0
MAX_GRID_PRICE: float = 0.30
GRID_EXPORT_RATE: float = 0.07
SOC_PENALTY_WEIGHT: float = 5.0
THERMAL_PENALTY_WEIGHT: float = 0.1

AgentType = Union[QLearningAgent, DQNAgent]

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Outcome of a single simulation timestep."""

    step: int
    hour: int
    action: int
    action_description: str
    pv_power_kw: float
    load_kw: float
    grid_import_kw: float
    grid_export_kw: float
    battery_charge_kw: float
    battery_discharge_kw: float
    soc: float
    soh: float
    battery_temp_k: float
    grid_price: float
    cost_grid_import: float
    revenue_grid_export: float
    soc_penalty: float
    thermal_penalty: float
    reward: float
    td_error: float
    cumulative_cost: float
    cumulative_revenue: float


class RewardCalculator:
    """Computes the composite reward signal for each simulation step."""

    def __init__(
        self,
        soc_min: float = 0.10,
        soc_max: float = 0.95,
        soc_penalty_weight: float = SOC_PENALTY_WEIGHT,
        thermal_penalty_weight: float = THERMAL_PENALTY_WEIGHT,
        export_rate: float = GRID_EXPORT_RATE,
    ) -> None:
        """Initialize reward calculator with tunable weights.

        Args:
            soc_min: SoC floor below which penalty is applied.
            soc_max: SoC ceiling above which penalty is applied.
            soc_penalty_weight: Scalar multiplier for SoC bound violations.
            thermal_penalty_weight: Scalar multiplier for excessive temperature rise.
            export_rate: Revenue rate for grid export in $/kWh.
        """
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_penalty_weight = soc_penalty_weight
        self.thermal_penalty_weight = thermal_penalty_weight
        self.export_rate = export_rate

    def compute(
        self,
        grid_import_kw: float,
        grid_export_kw: float,
        grid_price: float,
        soc: float,
        battery_temp_k: float,
        prev_battery_temp_k: float,
        dt_hours: float,
    ) -> dict[str, float]:
        """Compute all reward components for a single step.

        Args:
            grid_import_kw: Power drawn from the grid in kW (>= 0).
            grid_export_kw: Power sold to the grid in kW (>= 0).
            grid_price: Grid electricity price in $/kWh.
            soc: Battery state of charge after the step.
            battery_temp_k: Battery temperature after the step in Kelvin.
            prev_battery_temp_k: Battery temperature before the step in Kelvin.
            dt_hours: Timestep duration in hours.

        Returns:
            Dict with keys: reward, cost_import, revenue_export, soc_penalty,
            thermal_penalty.
        """
        cost_import = float(grid_import_kw) * float(grid_price) * float(dt_hours)
        revenue_export = float(grid_export_kw) * self.export_rate * float(dt_hours)

        soc_below = max(0.0, self.soc_min - float(soc))
        soc_above = max(0.0, float(soc) - self.soc_max)
        soc_penalty = self.soc_penalty_weight * (soc_below ** 2 + soc_above ** 2)

        temp_rise = float(battery_temp_k) - float(prev_battery_temp_k)
        thermal_penalty = self.thermal_penalty_weight * max(0.0, temp_rise - 2.0)

        reward = -cost_import + revenue_export - soc_penalty - thermal_penalty

        return {
            "reward": float(reward),
            "cost_import": float(cost_import),
            "revenue_export": float(revenue_export),
            "soc_penalty": float(soc_penalty),
            "thermal_penalty": float(thermal_penalty),
        }

    def explain(self, result: dict[str, float]) -> str:
        """Return a formatted one-line summary of the reward breakdown.

        Args:
            result: Dict returned by compute().

        Returns:
            Human-readable reward breakdown string.
        """
        return (
            f"R={result['reward']:.2f} "
            f"[import={-result['cost_import']:.2f}, "
            f"export=+{result['revenue_export']:.2f}, "
            f"soc_pen={-result['soc_penalty']:.2f}, "
            f"therm_pen={-result['thermal_penalty']:.2f}]"
        )


class PowerRouter:
    """Translates discrete agent actions into physical power flows."""

    def __init__(self, battery: BatteryBESS, dt_hours: float = 1.0) -> None:
        """Initialize router with battery reference and timestep.

        Args:
            battery: BatteryBESS instance to charge/discharge.
            dt_hours: Timestep duration in hours.
        """
        self._battery = battery
        self._dt_hours = dt_hours

    def _max_charge_rate(self, soc: float) -> float:
        """Return SoC-dependent maximum charge rate in kW (CC-CV taper).

        Args:
            soc: Current battery state of charge in [0, 1].

        Returns:
            Maximum charge power in kW.
        """
        full_rate = 5.0
        min_rate = 0.5
        taper_start = 0.80
        taper_end = 0.95
        if soc <= taper_start:
            return full_rate
        if soc >= taper_end:
            return min_rate
        frac = (soc - taper_start) / max(taper_end - taper_start, 1e-9)
        return full_rate - frac * (full_rate - min_rate)

    def route(
        self,
        action: int,
        pv_kw: float,
        load_kw: float,
        grid_price: float,
    ) -> dict[str, float]:
        """Apply action and return power flow dict.

        Args:
            action: Integer in [0, 3] selecting the routing strategy.
            pv_kw: Available PV power in kW.
            load_kw: Current load demand in kW.
            grid_price: Current grid electricity price in $/kWh.

        Returns:
            Dict with keys: grid_import_kw, grid_export_kw, battery_charge_kw,
            battery_discharge_kw, soc, soh, battery_temp_k. All kW >= 0.
        """
        grid_import_kw = 0.0
        grid_export_kw = 0.0
        battery_charge_kw = 0.0
        battery_discharge_kw = 0.0

        current_soc = self._battery.get_state()["soc"]

        if action == 0:
            # PV -> Load; excess PV -> Battery
            net = pv_kw - load_kw
            if net >= 0.0:
                charge_rate = min(net, self._max_charge_rate(current_soc))
                result = self._battery.charge(charge_rate, self._dt_hours)
                battery_charge_kw = float(result["actual_power_kw"])
                grid_import_kw = 0.0
            else:
                grid_import_kw = abs(net)
                battery_charge_kw = 0.0
            grid_export_kw = 0.0

        elif action == 1:
            # PV -> Load; excess PV -> Grid
            net = pv_kw - load_kw
            if net >= 0.0:
                grid_export_kw = net
                grid_import_kw = 0.0
            else:
                grid_import_kw = abs(net)
                grid_export_kw = 0.0
            battery_charge_kw = 0.0

        elif action == 2:
            # Battery -> Load; Grid supplements
            result = self._battery.discharge(load_kw, self._dt_hours)
            energy_delivered_kwh = float(result["energy_delivered_kwh"])
            delivered_kw = energy_delivered_kwh / max(self._dt_hours, 1e-9)
            battery_discharge_kw = delivered_kw
            shortfall = load_kw - delivered_kw
            grid_import_kw = max(0.0, shortfall)
            grid_export_kw = 0.0
            battery_charge_kw = 0.0

        elif action == 3:
            # Grid -> Load; Grid -> Battery (arbitrage)
            arbitrage_charge_kw = 2.0
            result = self._battery.charge(arbitrage_charge_kw, self._dt_hours)
            battery_charge_kw = float(result["actual_power_kw"])
            grid_import_kw = load_kw + battery_charge_kw
            grid_export_kw = 0.0
            battery_discharge_kw = 0.0

        else:
            log.warning("Unknown action %d; defaulting to action 0 behavior.", action)
            net = pv_kw - load_kw
            if net >= 0.0:
                charge_rate = min(net, self._max_charge_rate(current_soc))
                result = self._battery.charge(charge_rate, self._dt_hours)
                battery_charge_kw = float(result["actual_power_kw"])
            else:
                grid_import_kw = abs(net)

        batt_state = self._battery.get_state()
        return {
            "grid_import_kw": max(0.0, grid_import_kw),
            "grid_export_kw": max(0.0, grid_export_kw),
            "battery_charge_kw": max(0.0, battery_charge_kw),
            "battery_discharge_kw": max(0.0, battery_discharge_kw),
            "soc": batt_state["soc"],
            "soh": batt_state["soh"],
            "battery_temp_k": batt_state["temperature"],
        }


class MicrogridSimulator:
    """Orchestrates the 24-hour simulation loop for the solar microgrid."""

    def __init__(
        self,
        agent: AgentType,
        dt_hours: float = 1.0,
        training: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize simulator with an RL agent.

        Args:
            agent: QLearningAgent or DQNAgent instance.
            dt_hours: Timestep duration in hours. Default 1.0 (hourly).
            training: If True, agent uses epsilon-greedy; otherwise greedy.
            seed: RNG seed for environment profiles.
        """
        self._agent = agent
        self._dt_hours = dt_hours
        self._training = training
        self._seed = seed
        self._n_steps = int(round(24.0 / dt_hours))

        self._pv = PVArray()
        self._normalizer = StateNormalizer(
            max_pv_kw=MAX_PV_KW,
            max_load_kw=MAX_LOAD_KW,
            max_grid_price=MAX_GRID_PRICE,
        )
        self._reward_calc = RewardCalculator()

        self._bess: BatteryBESS = BatteryBESS()
        self._profiler: EnvironmentProfiler = EnvironmentProfiler(
            dt_hours=dt_hours, seed=seed
        )
        self._profiles: dict[str, np.ndarray] = self._profiler.generate_all()
        self._router: PowerRouter = PowerRouter(self._bess, dt_hours)

        self._log: list[StepResult] = []
        self._cumulative_cost: float = 0.0
        self._cumulative_revenue: float = 0.0
        self._current_step: int = 0

        self._logger = logging.getLogger(__name__)

    def reset(self) -> np.ndarray:
        """Reset simulator to initial state and return normalized state at step 0.

        Returns:
            Normalized state vector of shape (STATE_DIM,) with values in [0, 1].
        """
        self._bess = BatteryBESS()
        self._profiler = EnvironmentProfiler(dt_hours=self._dt_hours, seed=self._seed)
        self._profiles = self._profiler.generate_all()
        self._router = PowerRouter(self._bess, self._dt_hours)
        self._log = []
        self._cumulative_cost = 0.0
        self._cumulative_revenue = 0.0
        self._current_step = 0
        return self._build_state(0)

    def _build_state(self, step: int) -> np.ndarray:
        """Construct normalized state vector for the given step.

        Args:
            step: Timestep index in [0, n_steps - 1].

        Returns:
            Float32 numpy array of shape (STATE_DIM,) in [0, 1].
        """
        hour = int(round(step * self._dt_hours)) % 24
        batt_state = self._bess.get_state()
        soc = batt_state["soc"]
        pv_kw = float(self._pv.get_power(
            float(self._profiles["irradiance"][step]),
            float(self._profiles["temperature"][step]),
        )) / 1000.0
        load_kw = float(self._profiles["load"][step])
        # look-ahead grid price (next step, or current if at last step)
        price_step = min(step + 1, self._n_steps - 1)
        grid_price = float(self._profiles["grid_price"][price_step])

        return self._normalizer.normalize(hour, soc, pv_kw, load_kw, grid_price)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Execute one simulation timestep.

        Args:
            action: Integer in [0, 3] selected by the agent.

        Returns:
            Tuple of (next_state, reward, done).
        """
        s = self._current_step
        hour = int(round(s * self._dt_hours)) % 24

        G = float(self._profiles["irradiance"][s])
        T = float(self._profiles["temperature"][s])
        load_kw = float(self._profiles["load"][s])
        grid_price = float(self._profiles["grid_price"][s])

        pv_power_w = float(self._pv.get_power(G, T))
        pv_kw = pv_power_w / 1000.0

        prev_temp_k = self._bess.get_state()["temperature"]

        flows = self._router.route(action, pv_kw, load_kw, grid_price)

        reward_dict = self._reward_calc.compute(
            grid_import_kw=flows["grid_import_kw"],
            grid_export_kw=flows["grid_export_kw"],
            grid_price=grid_price,
            soc=flows["soc"],
            battery_temp_k=flows["battery_temp_k"],
            prev_battery_temp_k=prev_temp_k,
            dt_hours=self._dt_hours,
        )

        self._cumulative_cost += reward_dict["cost_import"]
        self._cumulative_revenue += reward_dict["revenue_export"]

        self._current_step += 1
        done = self._current_step >= self._n_steps

        next_state = self._build_state(min(self._current_step, self._n_steps - 1))

        result = StepResult(
            step=s,
            hour=hour,
            action=action,
            action_description=ACTION_DESCRIPTIONS.get(action, "unknown"),
            pv_power_kw=pv_kw,
            load_kw=load_kw,
            grid_import_kw=flows["grid_import_kw"],
            grid_export_kw=flows["grid_export_kw"],
            battery_charge_kw=flows["battery_charge_kw"],
            battery_discharge_kw=flows["battery_discharge_kw"],
            soc=flows["soc"],
            soh=flows["soh"],
            battery_temp_k=flows["battery_temp_k"],
            grid_price=grid_price,
            cost_grid_import=reward_dict["cost_import"],
            revenue_grid_export=reward_dict["revenue_export"],
            soc_penalty=reward_dict["soc_penalty"],
            thermal_penalty=reward_dict["thermal_penalty"],
            reward=reward_dict["reward"],
            td_error=0.0,
            cumulative_cost=self._cumulative_cost,
            cumulative_revenue=self._cumulative_revenue,
        )
        self._log.append(result)

        self._logger.debug(
            "Step %02d | action=%d | pv=%.2fkW | load=%.2fkW | "
            "import=%.2fkW | export=%.2fkW | soc=%.3f | reward=%.4f",
            s, action, pv_kw, load_kw,
            flows["grid_import_kw"], flows["grid_export_kw"],
            flows["soc"], reward_dict["reward"],
        )

        return next_state, reward_dict["reward"], done

    def run_episode(self) -> list[StepResult]:
        """Run a complete 24-step episode.

        Returns:
            List of StepResult objects, one per timestep.
        """
        state = self.reset()
        episode_rewards: list[float] = []

        for _ in range(self._n_steps):
            action = self._agent.select_action(state, training=self._training)
            next_state, reward, done = self.step(action)

            td_error = self._agent.update(state, action, reward, next_state, done)
            # Store td_error back into the last log entry
            self._log[-1].td_error = float(td_error)

            episode_rewards.append(reward)
            state = next_state

        self._agent.decay_epsilon()
        self._agent.record_episode_reward(float(sum(episode_rewards)))

        self._logger.info(
            "Episode complete | steps=%d | total_reward=%.4f | "
            "net_cost=%.4f | final_soc=%.3f",
            len(self._log),
            sum(episode_rewards),
            self._cumulative_cost - self._cumulative_revenue,
            self._log[-1].soc if self._log else 0.0,
        )
        return self._log

    def get_episode_summary(self) -> dict[str, float]:
        """Compute summary metrics from the most recently completed episode.

        Returns:
            Dict with 12 summary keys covering cost, energy, and performance.
        """
        if not self._log:
            return {k: 0.0 for k in (
                "total_cost", "total_revenue", "net_cost",
                "total_pv_kwh", "total_load_kwh",
                "grid_import_kwh", "grid_export_kwh",
                "grid_dependence_pct", "final_soc", "final_soh",
                "mean_reward", "min_soc",
            )}

        total_cost = sum(r.cost_grid_import for r in self._log)
        total_revenue = sum(r.revenue_grid_export for r in self._log)
        total_pv_kwh = sum(r.pv_power_kw * self._dt_hours for r in self._log)
        total_load_kwh = sum(r.load_kw * self._dt_hours for r in self._log)
        grid_import_kwh = sum(r.grid_import_kw * self._dt_hours for r in self._log)
        grid_export_kwh = sum(r.grid_export_kw * self._dt_hours for r in self._log)
        grid_dependence_pct = (
            grid_import_kwh / max(total_load_kwh, 1e-9) * 100.0
        )
        final_soc = self._log[-1].soc
        final_soh = self._log[-1].soh
        mean_reward = float(np.mean([r.reward for r in self._log]))
        min_soc = min(r.soc for r in self._log)

        return {
            "total_cost": float(total_cost),
            "total_revenue": float(total_revenue),
            "net_cost": float(total_cost - total_revenue),
            "total_pv_kwh": float(total_pv_kwh),
            "total_load_kwh": float(total_load_kwh),
            "grid_import_kwh": float(grid_import_kwh),
            "grid_export_kwh": float(grid_export_kwh),
            "grid_dependence_pct": float(np.clip(grid_dependence_pct, 0.0, 100.0)),
            "final_soc": float(final_soc),
            "final_soh": float(final_soh),
            "mean_reward": float(mean_reward),
            "min_soc": float(min_soc),
        }

    def save_log(self, path: str | None = None) -> str:
        """Serialize episode log to CSV.

        Args:
            path: Destination path. If None, auto-generates timestamped filename
                under LOG_DIR.

        Returns:
            Absolute path of the written CSV file.
        """
        if path is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(LOG_DIR, f"simulation_{ts}.csv")
        else:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        field_names = [f.name for f in fields(StepResult)]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            for r in self._log:
                writer.writerow(asdict(r))

        self._logger.info("Episode log saved to %s (%d rows)", path, len(self._log))
        return path

    def load_log(self, path: str) -> list[StepResult]:
        """Read a CSV log previously written by save_log().

        Args:
            path: Path to the CSV file.

        Returns:
            List of reconstructed StepResult objects.
        """
        results: list[StepResult] = []
        field_names = {f.name: f.type for f in fields(StepResult)}

        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                kwargs: dict = {}
                for fname, ftype in field_names.items():
                    raw = row[fname]
                    if fname in ("step", "hour", "action"):
                        kwargs[fname] = int(raw)
                    elif fname == "action_description":
                        kwargs[fname] = str(raw)
                    else:
                        kwargs[fname] = float(raw)
                results.append(StepResult(**kwargs))

        self._logger.info("Loaded %d rows from %s", len(results), path)
        return results


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    _log = _logging.getLogger(__name__)

    from agent import create_agent

    _log.info("=== Phase 3 Simulator Validation ===")

    agent = create_agent("qlearning", n_bins=10, seed=42)
    sim = MicrogridSimulator(agent=agent, training=True, seed=42)

    _log.info("--- Running 5 training episodes ---")
    for ep in range(5):
        results = sim.run_episode()
        summary = sim.get_episode_summary()
        _log.info(
            "Episode %2d | Net cost: $%.4f | Grid dep: %.1f%% | "
            "Mean reward: %.4f | ε: %.4f",
            ep + 1,
            summary["net_cost"],
            summary["grid_dependence_pct"],
            summary["mean_reward"],
            agent.get_stats()["epsilon"],
        )
        assert len(results) == 24, f"Expected 24 steps, got {len(results)}"
        assert all(
            0.10 <= r.soc <= 0.95 for r in results
        ), "SoC out of bounds"

    _log.info("--- Saving simulation log ---")
    path = sim.save_log()
    _log.info("Log saved to: %s", path)

    _log.info("--- Reloading log ---")
    loaded = sim.load_log(path)
    assert len(loaded) == 24
    assert abs(loaded[0].reward - results[0].reward) < 1e-6

    _log.info("--- Episode summary ---")
    summary = sim.get_episode_summary()
    for k, v in summary.items():
        _log.info("  %s: %.4f", k, v)

    assert summary["grid_dependence_pct"] >= 0.0
    assert summary["final_soc"] >= 0.10

    print("\nPhase 3 validation PASSED")
