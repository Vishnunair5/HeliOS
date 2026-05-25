"""Full pytest coverage for physics_engine.py — Phase 1."""

import math
import numpy as np
import pytest
from physics_engine import PVArray, BatteryBESS, EnvironmentProfiler


class TestPVArray:

    def test_thermal_voltage_at_stc(self, pv_array: PVArray) -> None:
        vth = pv_array._thermal_voltage(298.15)
        assert abs(vth - 0.02569) < 1e-4

    def test_thermal_voltage_scales_with_temperature(self, pv_array: PVArray) -> None:
        assert pv_array._thermal_voltage(350.0) > pv_array._thermal_voltage(298.15)

    def test_light_current_at_stc(self, pv_array: PVArray) -> None:
        il = pv_array._light_current(1000.0, 298.15)
        assert abs(il - pv_array.I_sc_ref) < 1e-6

    def test_light_current_scales_with_irradiance(self, pv_array: PVArray) -> None:
        il_full = pv_array._light_current(1000.0, 298.15)
        il_half = pv_array._light_current(500.0, 298.15)
        assert abs(il_half - il_full / 2.0) < 1e-6

    def test_light_current_zero_at_night(self, pv_array: PVArray) -> None:
        assert pv_array._light_current(0.0, 298.15) == 0.0

    def test_saturation_current_positive(self, pv_array: PVArray) -> None:
        assert pv_array._saturation_current(298.15) > 0.0

    def test_solve_iv_point_at_zero_voltage(self, pv_array: PVArray) -> None:
        I = pv_array.solve_iv_point(V=0.0, G=1000.0, T=298.15)
        expected = pv_array.I_sc_ref * pv_array.n_parallel
        assert abs(I - expected) / expected < 0.05

    def test_solve_iv_point_at_night(self, pv_array: PVArray) -> None:
        assert pv_array.solve_iv_point(V=1.0, G=0.0, T=298.15) == 0.0

    def test_compute_mpp_keys(self, pv_array: PVArray) -> None:
        result = pv_array.compute_mpp(G=800.0, T=300.0)
        assert set(result.keys()) == {"V_mp", "I_mp", "P_mp", "V_oc", "I_sc"}

    def test_compute_mpp_power_positive(self, pv_array: PVArray) -> None:
        result = pv_array.compute_mpp(G=800.0, T=300.0)
        assert result["P_mp"] > 0.0

    def test_compute_mpp_power_at_night(self, pv_array: PVArray) -> None:
        assert pv_array.get_power(G=0.0, T=298.15) == 0.0

    def test_compute_mpp_power_increases_with_irradiance(self, pv_array: PVArray) -> None:
        assert pv_array.get_power(1000.0, 298.15) > pv_array.get_power(500.0, 298.15)

    def test_compute_mpp_power_decreases_with_temperature(self, pv_array: PVArray) -> None:
        assert pv_array.get_power(1000.0, 298.15) > pv_array.get_power(1000.0, 320.0)

    def test_no_nan_in_mpp_sweep(self, pv_array: PVArray) -> None:
        for G in [0.0, 200.0, 500.0, 800.0, 1000.0]:
            for T in [285.0, 298.0, 320.0]:
                result = pv_array.compute_mpp(G, T)
                for key, val in result.items():
                    assert math.isfinite(val), f"Non-finite {key}={val} at G={G}, T={T}"


class TestBatteryBESS:

    def test_initial_state(self, bess: BatteryBESS) -> None:
        assert bess.soc == 0.5
        assert bess.soh == 1.0
        assert bess.temperature == 298.15

    def test_voc_increases_with_soc(self, bess: BatteryBESS) -> None:
        assert bess._voc_from_soc(0.9) > bess._voc_from_soc(0.5) > bess._voc_from_soc(0.1)

    def test_charge_increases_soc(self, bess: BatteryBESS) -> None:
        soc_before = bess.soc
        bess.charge(2.0, 1.0)
        assert bess.soc > soc_before

    def test_charge_result_keys(self, bess: BatteryBESS) -> None:
        result = bess.charge(2.0, 1.0)
        assert set(result.keys()) == {"energy_stored_kwh", "actual_power_kw", "soc", "soh", "temperature"}

    def test_charge_respects_soc_ceiling(self, bess_full: BatteryBESS) -> None:
        bess_full.charge(10.0, 1.0)
        assert bess_full.soc <= bess_full.soc_max

    def test_discharge_decreases_soc(self, bess: BatteryBESS) -> None:
        soc_before = bess.soc
        bess.discharge(2.0, 1.0)
        assert bess.soc < soc_before

    def test_discharge_result_keys(self, bess: BatteryBESS) -> None:
        result = bess.discharge(2.0, 1.0)
        assert set(result.keys()) == {"energy_delivered_kwh", "actual_power_kw", "soc", "soh", "temperature"}

    def test_discharge_respects_soc_floor(self, bess_empty: BatteryBESS) -> None:
        bess_empty.discharge(20.0, 1.0)
        assert bess_empty.soc >= bess_empty.soc_min

    def test_energy_conservation_on_charge(self, bess: BatteryBESS) -> None:
        result = bess.charge(5.0, 1.0)
        assert result["energy_stored_kwh"] <= 5.0

    def test_energy_conservation_on_discharge(self, bess: BatteryBESS) -> None:
        result = bess.discharge(5.0, 1.0)
        assert result["energy_delivered_kwh"] <= 5.0

    def test_soh_degrades_after_cycling(self) -> None:
        b = BatteryBESS(soc_init=0.5, degradation_rate=0.00003)
        for _ in range(50):
            b.charge(100.0, 1.0)   # charge to ceiling
            b.discharge(100.0, 1.0)  # discharge to floor
        assert b.soh < 1.0

    def test_temperature_rises_during_operation(self, bess: BatteryBESS) -> None:
        T_before = bess.temperature
        bess.charge(10.0, 1.0)
        assert bess.temperature > T_before

    def test_no_nan_in_state(self, bess: BatteryBESS) -> None:
        for i in range(24):
            if i % 2 == 0:
                bess.charge(3.0, 1.0)
            else:
                bess.discharge(3.0, 1.0)
        state = bess.get_state()
        for key, val in state.items():
            assert math.isfinite(val), f"Non-finite {key}={val}"


class TestEnvironmentProfiler:

    def test_irradiance_length(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        assert len(irr) == 24

    def test_irradiance_non_negative(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        assert np.all(irr >= 0.0)

    def test_irradiance_clipped_to_1000(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        assert np.all(irr <= 1000.0)

    def test_irradiance_peaks_near_noon(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        peak_idx = int(np.argmax(irr))
        assert 10 <= peak_idx <= 16

    def test_irradiance_zero_at_night(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        for h in [0, 1, 2, 22, 23]:
            assert irr[h] == 0.0, f"Irradiance nonzero at hour {h}: {irr[h]}"

    def test_temperature_length(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        temp = profiler.temperature_profile(irr)
        assert len(temp) == 24

    def test_temperature_in_kelvin_range(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        temp = profiler.temperature_profile(irr)
        assert np.all(temp >= 280.0) and np.all(temp <= 360.0)

    def test_temperature_cell_exceeds_ambient(self, profiler: EnvironmentProfiler) -> None:
        irr = profiler.irradiance_profile()
        temp = profiler.temperature_profile(irr)
        # At peak irradiance hours (10-16), cell temp must exceed pure sinusoidal ambient
        peak_hours = range(10, 17)
        T_min, T_max = 285.0, 308.0
        hours = np.arange(24)
        T_amb = T_min + (T_max - T_min) * 0.5 * (1.0 - np.cos(2 * np.pi * (hours - 6) / 24))
        for h in peak_hours:
            if irr[h] > 100.0:
                assert temp[h] > T_amb[h], f"Cell temp not above ambient at hour {h}"

    def test_load_profile_length(self, profiler: EnvironmentProfiler) -> None:
        assert len(profiler.load_profile()) == 24

    def test_load_positive(self, profiler: EnvironmentProfiler) -> None:
        load = profiler.load_profile()
        assert np.all(load > 0.0)

    def test_load_evening_peak(self, profiler: EnvironmentProfiler) -> None:
        load = profiler.load_profile()
        evening_mean = np.mean(load[18:22])
        night_mean = np.mean(load[2:6])
        assert evening_mean > night_mean

    def test_grid_pricing_length(self, profiler: EnvironmentProfiler) -> None:
        assert len(profiler.grid_pricing_profile()) == 24

    def test_grid_pricing_positive(self, profiler: EnvironmentProfiler) -> None:
        price = profiler.grid_pricing_profile()
        assert np.all(price > 0.0)

    def test_grid_pricing_onpeak_highest(self, profiler: EnvironmentProfiler) -> None:
        price = profiler.grid_pricing_profile()
        # On-peak hours: 11-14 and 17-20
        onpeak = np.concatenate([price[11:14], price[17:20]])
        offpeak = price[0:6]
        assert np.mean(onpeak) > np.mean(offpeak)

    def test_generate_all_keys(self, profiler: EnvironmentProfiler) -> None:
        result = profiler.generate_all()
        assert set(result.keys()) == {"irradiance", "temperature", "load", "grid_price"}

    def test_generate_all_consistent_lengths(self, profiler: EnvironmentProfiler) -> None:
        result = profiler.generate_all()
        lengths = [len(v) for v in result.values()]
        assert len(set(lengths)) == 1

    def test_reproducibility(self) -> None:
        p1 = EnvironmentProfiler(dt_hours=1.0, seed=42)
        p2 = EnvironmentProfiler(dt_hours=1.0, seed=42)
        r1 = p1.generate_all()
        r2 = p2.generate_all()
        for key in r1:
            np.testing.assert_array_equal(r1[key], r2[key])
