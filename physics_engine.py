"""High-fidelity physics models for solar microgrid simulation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# Physical constants
BOLTZMANN_K: float = 1.380649e-23    # J/K
ELECTRON_Q: float = 1.602176634e-19  # C
STC_IRRADIANCE: float = 1000.0       # W/m²
STC_TEMPERATURE: float = 298.15      # K


class PVArray:
    """Photovoltaic array modeled via Single-Diode Equivalent Circuit."""

    def __init__(
        self,
        n_series: int = 10,
        n_parallel: int = 5,
        I_sc_ref: float = 9.0,
        V_oc_ref: float = 0.65,
        alpha_isc: float = 0.0004,
        beta_voc: float = -0.0023,
        R_s: float = 0.3,
        R_sh: float = 200.0,
        n_ideal: float = 1.3,
        T_ref: float = 298.15,
        G_ref: float = 1000.0,
    ) -> None:
        self.n_series = n_series
        self.n_parallel = n_parallel
        self.I_sc_ref = I_sc_ref
        self.V_oc_ref = V_oc_ref
        self.alpha_isc = alpha_isc
        self.beta_voc = beta_voc
        self.R_s = R_s
        self.R_sh = R_sh
        self.n_ideal = n_ideal
        self.T_ref = T_ref
        self.G_ref = G_ref

    def _thermal_voltage(self, T: float) -> float:
        """V_th = k*T / q (per cell)."""
        return BOLTZMANN_K * T / ELECTRON_Q

    def _light_current(self, G: float, T: float) -> float:
        """I_L = I_sc_ref * (G/G_ref) * [1 + alpha_isc*(T - T_ref)]"""
        return self.I_sc_ref * (G / self.G_ref) * (1.0 + self.alpha_isc * (T - self.T_ref))

    def _saturation_current(self, T: float) -> float:
        """Bandgap-corrected diode saturation current relative to T_ref."""
        V_th_ref = self._thermal_voltage(self.T_ref)
        V_th_T = self._thermal_voltage(T)
        # I_0 at reference: derived from open-circuit condition at STC
        # I_0_ref = I_sc_ref / (exp(V_oc_ref / (n * V_th_ref)) - 1)
        I_0_ref = self.I_sc_ref / (
            np.exp(self.V_oc_ref / (self.n_ideal * V_th_ref)) - 1.0
        )
        # Temperature scaling with bandgap energy (Si: ~1.12 eV)
        E_g = 1.12  # eV
        E_g_J = E_g * ELECTRON_Q
        exponent = (E_g_J / (self.n_ideal * BOLTZMANN_K)) * (1.0 / self.T_ref - 1.0 / T)
        I_0 = I_0_ref * (T / self.T_ref) ** 3 * np.exp(exponent)
        return float(I_0)

    def _iv_residual(self, I: float, V: float, G: float, T: float) -> float:
        """Residual of single-diode equation at string level (pvlib convention).

        I: string current (A), V: string/array voltage (V).
        Denominator uses n_ideal * n_series * V_th so R_s/R_sh stay module-level.
        """
        V_th = self._thermal_voltage(T)
        I_L = self._light_current(G, T)
        I_0 = self._saturation_current(T)
        nNsVth = self.n_ideal * self.n_series * V_th
        exponent_arg = min((V + I * self.R_s) / nNsVth, 700.0)
        diode_current = I_0 * (np.exp(exponent_arg) - 1.0)
        shunt_current = (V + I * self.R_s) / self.R_sh
        return I - I_L + diode_current + shunt_current

    def solve_iv_point(self, V: float, G: float, T: float) -> float:
        """Solve for array current at array voltage V using brentq."""
        if G < 1.0:
            return 0.0
        I_L = self._light_current(G, T)
        # Search over string current; V is array voltage = string voltage
        try:
            I_string = brentq(
                lambda i: self._iv_residual(i, V, G, T),
                0.0,
                I_L * 1.05,
                xtol=1e-8,
                maxiter=200,
            )
        except ValueError:
            return 0.0
        return float(I_string * self.n_parallel)

    def compute_mpp(
        self, G: float, T: float, v_points: int = 200
    ) -> dict[str, float]:
        """Sweep voltage 0→V_oc, find MPP. Returns V_mp, I_mp, P_mp, V_oc, I_sc."""
        if G < 1.0:
            return {"V_mp": 0.0, "I_mp": 0.0, "P_mp": 0.0, "V_oc": 0.0, "I_sc": 0.0}

        V_th = self._thermal_voltage(T)
        # Approximate V_oc for sweep range
        V_oc_cell = self.V_oc_ref + self.beta_voc * (T - self.T_ref)
        V_oc_array = V_oc_cell * self.n_series * max(0.5, G / self.G_ref) ** 0.05

        voltages = np.linspace(0.0, V_oc_array * 0.999, v_points)
        currents = np.array([self.solve_iv_point(v, G, T) for v in voltages])
        powers = voltages * currents

        # Refine V_oc: find zero crossing
        # I at V=0 gives I_sc
        I_sc = self.solve_iv_point(0.0, G, T)

        # Find V_oc by bisection on power array zero crossing
        valid_mask = currents > 0
        if valid_mask.any():
            last_valid = np.where(valid_mask)[0][-1]
            V_oc_est = voltages[last_valid]
        else:
            V_oc_est = V_oc_array

        mpp_idx = int(np.argmax(powers))
        V_mp = float(voltages[mpp_idx])
        I_mp = float(currents[mpp_idx])
        P_mp = float(powers[mpp_idx])

        return {
            "V_mp": V_mp,
            "I_mp": I_mp,
            "P_mp": P_mp,
            "V_oc": float(V_oc_est),
            "I_sc": float(I_sc),
        }

    def get_power(self, G: float, T: float) -> float:
        """Return maximum power output in watts."""
        return self.compute_mpp(G, T)["P_mp"]


class BatteryBESS:
    """Battery Energy Storage System with thermal and degradation modeling."""

    def __init__(
        self,
        capacity_kwh: float = 20.0,
        soc_init: float = 0.5,
        soc_min: float = 0.10,
        soc_max: float = 0.95,
        r_int: float = 0.05,
        eta_charge: float = 0.96,
        eta_discharge: float = 0.94,
        v_nominal: float = 48.0,
        thermal_mass: float = 2_000_000.0,
        degradation_rate: float = 0.00003,
    ) -> None:
        self.capacity_kwh = capacity_kwh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.r_int = r_int
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.v_nominal = v_nominal
        self.thermal_mass = thermal_mass
        self.degradation_rate = degradation_rate

        self.soc: float = float(np.clip(soc_init, soc_min, soc_max))
        self.soh: float = 1.0
        self.temperature: float = 298.15
        self._cycle_accumulator: float = 0.0

    def _voc_from_soc(self, soc: float) -> float:
        """Piecewise-linear OCV curve approximating Li-ion behavior."""
        # Breakpoints: (soc, voc_fraction_of_nominal)
        soc_pts = [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.0]
        voc_pts = [0.82, 0.86, 0.90, 0.94, 0.96, 0.98, 0.99, 1.01]
        voc_fraction = float(np.interp(soc, soc_pts, voc_pts))
        return voc_fraction * self.v_nominal

    def charge(self, power_kw: float, dt_hours: float) -> dict[str, float]:
        """Charge battery; enforce SoC ceiling; account for eta and r_int losses."""
        soc_before = self.soc

        # Max energy that can be accepted
        headroom_kwh = (self.soc_max - self.soc) * self.capacity_kwh * self.soh
        max_energy_storable = headroom_kwh  # energy into battery

        # Energy input from source
        energy_input_kwh = power_kw * dt_hours

        # Energy stored after efficiency
        energy_to_store = min(energy_input_kwh * self.eta_charge, max_energy_storable)
        actual_energy_input = energy_to_store / self.eta_charge

        # Actual power drawn from source
        actual_power_kw = actual_energy_input / dt_hours if dt_hours > 0 else 0.0

        # Update SoC
        delta_soc = energy_to_store / (self.capacity_kwh * self.soh)
        self.soc = float(np.clip(self.soc + delta_soc, self.soc_min, self.soc_max))

        # Thermal update: I²R heating
        current_a = (actual_power_kw * 1000.0) / max(self._voc_from_soc(self.soc), 1.0)
        heat_j = current_a ** 2 * self.r_int * dt_hours * 3600.0
        delta_T = heat_j / self.thermal_mass
        self.temperature = float(self.temperature + delta_T)

        self._update_degradation(abs(self.soc - soc_before))

        return {
            "energy_stored_kwh": float(energy_to_store),
            "actual_power_kw": float(actual_power_kw),
            "soc": self.soc,
            "soh": self.soh,
            "temperature": self.temperature,
        }

    def discharge(self, power_kw: float, dt_hours: float) -> dict[str, float]:
        """Discharge battery; enforce SoC floor; account for eta and r_int losses."""
        soc_before = self.soc

        # Max energy available
        available_kwh = (self.soc - self.soc_min) * self.capacity_kwh * self.soh
        max_deliverable = available_kwh * self.eta_discharge

        energy_requested_kwh = power_kw * dt_hours
        energy_delivered = min(energy_requested_kwh, max_deliverable)
        energy_from_battery = energy_delivered / self.eta_discharge if self.eta_discharge > 0 else 0.0

        actual_power_kw = energy_delivered / dt_hours if dt_hours > 0 else 0.0

        # Update SoC
        delta_soc = energy_from_battery / (self.capacity_kwh * self.soh)
        self.soc = float(np.clip(self.soc - delta_soc, self.soc_min, self.soc_max))

        # Thermal update
        current_a = (actual_power_kw * 1000.0) / max(self._voc_from_soc(self.soc), 1.0)
        heat_j = current_a ** 2 * self.r_int * dt_hours * 3600.0
        delta_T = heat_j / self.thermal_mass
        self.temperature = float(self.temperature + delta_T)

        self._update_degradation(abs(self.soc - soc_before))

        return {
            "energy_delivered_kwh": float(energy_delivered),
            "actual_power_kw": float(actual_power_kw),
            "soc": self.soc,
            "soh": self.soh,
            "temperature": self.temperature,
        }

    def _update_degradation(self, delta_soc: float) -> None:
        """Rainflow-lite: accumulate |delta_soc|; decrement SoH per full cycle, scaled by DoD."""
        self._cycle_accumulator += delta_soc
        while self._cycle_accumulator >= 1.0:
            self._cycle_accumulator -= 1.0
            dod = min(delta_soc, 1.0)
            self.soh = max(0.0, self.soh - self.degradation_rate * dod)

    def get_state(self) -> dict[str, float]:
        """Return current battery state."""
        return {
            "soc": self.soc,
            "soh": self.soh,
            "temperature": self.temperature,
            "voc": self._voc_from_soc(self.soc),
        }


class EnvironmentProfiler:
    """Generates 24-hour environmental and economic profiles."""

    def __init__(self, dt_hours: float = 1.0, seed: int = 42) -> None:
        self.dt_hours = dt_hours
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._n_steps = int(24 / dt_hours)

    def irradiance_profile(self) -> np.ndarray:
        """Gaussian bell peaking ~950 W/m² at hour 13 with cloud transients."""
        rng = np.random.default_rng(self.seed)
        hours = np.arange(self._n_steps) * self.dt_hours
        # Gaussian bell curve
        peak = 950.0
        mu = 13.0
        sigma = 3.5
        irr = peak * np.exp(-0.5 * ((hours - mu) / sigma) ** 2)

        # Night mask: before hour 6, after hour 20
        night_mask = (hours < 6.0) | (hours > 20.0)
        irr[night_mask] = 0.0

        # Cloud transients: 3-5 dips
        n_dips = rng.integers(3, 6)
        dip_hours = rng.choice(np.where(~night_mask)[0], size=n_dips, replace=False)
        for h in dip_hours:
            reduction = rng.uniform(0.40, 0.80)
            irr[h] *= (1.0 - reduction)

        return np.clip(irr, 0.0, 1000.0)

    def temperature_profile(self, irradiance: np.ndarray) -> np.ndarray:
        """Sinusoidal ambient + NOCT-based cell offset."""
        hours = np.arange(self._n_steps) * self.dt_hours
        # Sinusoidal ambient: 285K night, 308K peak at 14h
        T_min = 285.0
        T_max = 308.0
        T_amb = T_min + (T_max - T_min) * 0.5 * (1.0 - np.cos(2 * np.pi * (hours - 6) / 24))

        # NOCT cell temperature offset
        NOCT = 45.0  # °C
        G_ref = 1000.0
        T_cell = T_amb + (irradiance / G_ref) * (NOCT - 20.0) * 5.0 / 9.0

        return T_cell

    def load_profile(self) -> np.ndarray:
        """Morning peak 07-09h, evening peak 18-21h, base overnight + noise."""
        rng = np.random.default_rng(self.seed + 1)
        hours = np.arange(self._n_steps)
        load = np.full(self._n_steps, 1.5)  # base kW

        for h in range(self._n_steps):
            hr = hours[h] * self.dt_hours
            if 7 <= hr < 9:
                load[h] += 3.5
            elif 18 <= hr < 21:
                load[h] += 4.0
            elif 12 <= hr < 14:
                load[h] += 1.0

        noise = rng.normal(0.0, 0.15, size=self._n_steps)
        load = load + noise
        return np.maximum(load, 0.5)  # always some base draw

    def grid_pricing_profile(self) -> np.ndarray:
        """3-tier TOU pricing with slight noise."""
        rng = np.random.default_rng(self.seed + 2)
        hours = np.arange(self._n_steps)
        price = np.full(self._n_steps, 0.08)  # off-peak default

        for h in range(self._n_steps):
            hr = hours[h] * self.dt_hours
            if 9 <= hr < 21:
                if 11 <= hr < 14 or 17 <= hr < 20:
                    price[h] = 0.24  # on-peak
                else:
                    price[h] = 0.14  # mid-peak

        noise = rng.normal(0.0, 0.003, size=self._n_steps)
        return np.maximum(price + noise, 0.01)

    def generate_all(self) -> dict[str, np.ndarray]:
        """Generate all four profiles with consistent seed."""
        irr = self.irradiance_profile()
        temp = self.temperature_profile(irr)
        load = self.load_profile()
        price = self.grid_pricing_profile()
        return {
            "irradiance": irr,
            "temperature": temp,
            "load": load,
            "grid_price": price,
        }


if __name__ == "__main__":
    import pandas as pd

    profiler = EnvironmentProfiler(dt_hours=1.0, seed=42)
    profiles = profiler.generate_all()
    pv = PVArray()
    bess = BatteryBESS()

    rows = []
    for t in range(24):
        G = profiles["irradiance"][t]
        T = profiles["temperature"][t]
        mpp = pv.compute_mpp(G, T)
        state = bess.get_state()
        rows.append({
            "Hour": t,
            "G (W/m²)": round(G, 1),
            "T_cell (K)": round(T, 2),
            "P_pv (kW)": round(mpp["P_mp"] / 1000, 3),
            "SoC": round(state["soc"], 4),
            "SoH": round(state["soh"], 6),
            "Grid Price ($/kWh)": round(profiles["grid_price"][t], 4),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    for row in rows:
        assert row["P_pv (kW)"] >= 0.0, f"Negative power at hour {row['Hour']}"
        assert all(map(lambda v: v == v, row.values())), f"NaN at hour {row['Hour']}"

    print("\nPhase 1 validation PASSED")
