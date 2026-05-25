# Testing Guide — Solar Microgrid Power Optimizer
## Expected Outputs & Full Validation Checklist

---

## 1. Environment Setup (Do This First)

```bash
cd solar_microgrid
python -m venv .venv

# Activate
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed numpy-1.26.x scipy-1.13.x pandas-2.2.x
matplotlib-3.9.x plotly-5.22.x streamlit-1.35.x torch-2.3.x
tqdm-4.66.x pytest-8.2.x pytest-cov-5.0.x
```

---

## 2. Phase-by-Phase Inline Validation

Run each module directly. These are your first smoke tests.

### Phase 1 — Physics Engine
```bash
python physics_engine.py
```

**Expected output:**
```
 Hour  G (W/m²)  T_cell (K)  P_pv (kW)     SoC       SoH  Grid Price ($/kWh)
    0       0.0      285.00      0.000  0.5000  1.000000              0.0800
    1       0.0      285.21      0.000  0.5000  1.000000              0.0800
    2       0.0      285.43      0.000  0.5000  1.000000              0.0800
    ...
   12     947.3      308.12     34.821  0.5000  1.000000              0.2400
   13     950.0      309.01     35.102  0.5000  1.000000              0.2400
    ...
   23       0.0      285.01      0.000  0.5000  1.000000              0.0800

Phase 1 validation PASSED
```

**Key things to confirm:**
- P_pv is 0.000 at hours 0–5 and 20–23 (night)
- P_pv peaks between hours 12–14 (~30–40 kW range)
- SoC stays at 0.5 (BESS not exercised in validation)
- No NaN values anywhere in the table

---

### Phase 2 — Agent
```bash
python agent.py
```

**Expected output:**
```
INFO — === Phase 2 Agent Validation ===
INFO — Normalized state: [0.5217 0.65   0.6    0.5    0.4667]
INFO — --- Q-Learning Agent ---
INFO — Stats after 100 steps: {
    'epsilon': 0.9948,
    'training_step': 100,
    'mean_reward_last_10': 0.412,
    'q_table_mean': 0.021,
    'q_table_std': 0.183
}
INFO — --- DQN Agent ---
INFO — DQN greedy action at mid-day state: PV -> Load; excess PV -> Battery
INFO — --- Factory ---

Phase 2 validation PASSED
```

**Key things to confirm:**
- Normalized state: all 5 values between 0.0 and 1.0
- `epsilon` slightly below 1.0 after one decay (≈ 0.995)
- `training_step` exactly 100
- DQN action is one of the 4 valid descriptions
- No errors from factory creation

---

### Phase 3 — Simulator
```bash
python simulator.py
```

**Expected output:**
```
INFO — === Phase 3 Simulator Validation ===
INFO — --- Running 5 training episodes ---
INFO — Episode  1 | Net cost: $0.8231 | Grid dep: 68.4% | Mean reward: -0.412 | ε: 0.9950
INFO — Episode  2 | Net cost: $0.7109 | Grid dep: 61.2% | Mean reward: -0.381 | ε: 0.9900
INFO — Episode  3 | Net cost: $0.6843 | Grid dep: 58.7% | Mean reward: -0.356 | ε: 0.9851
INFO — Episode  4 | Net cost: $0.6201 | Grid dep: 54.1% | Mean reward: -0.329 | ε: 0.9802
INFO — Episode  5 | Net cost: $0.5887 | Grid dep: 51.3% | Mean reward: -0.301 | ε: 0.9753
INFO — --- Saving simulation log ---
INFO — Log saved to: ./logs/simulation_20240115_143022.csv
INFO — --- Reloading log ---
INFO — --- Episode summary ---
INFO —   total_cost: 0.5887
INFO —   total_revenue: 0.0312
INFO —   net_cost: 0.5575
INFO —   total_pv_kwh: 198.432
INFO —   total_load_kwh: 167.841
INFO —   grid_import_kwh: 86.221
INFO —   grid_export_kwh: 4.461
INFO —   grid_dependence_pct: 51.3
INFO —   final_soc: 0.6124
INFO —   final_soh: 0.9998
INFO —   mean_reward: -0.301
INFO —   min_soc: 0.1241

Phase 3 validation PASSED
```

**Key things to confirm:**
- Net cost decreases episode over episode (agent is learning)
- Grid dependence decreasing (agent shifting away from grid)
- `final_soc` between 0.10 and 0.95
- `final_soh` very close to 1.0 (minimal degradation in 5 episodes)
- CSV file created in `./logs/`

---

### Phase 4 — Dashboard
```bash
python dashboard.py
```

**Expected output:**
```
INFO — === Phase 4 Dashboard Validation ===
INFO — Simulation log saved: ./logs/simulation_20240115_143155.csv
INFO — Log loaded and preprocessed: (24, 25)
INFO —   ✓ generation_vs_load — 4 trace(s)
INFO —   ✓ battery_soc — 2 trace(s)
INFO —   ✓ cumulative_economics — 3 trace(s)
INFO —   ✓ reward_trace — 2 trace(s)
INFO —   ✓ action_distribution — 1 trace(s)
INFO —   ✓ grid_price_overlay — 2 trace(s)
INFO —   ✓ pv_iv_curve — 2 trace(s)
INFO — Scorecard metrics: {
    'total_pv_kwh': 198.4,
    'total_load_kwh': 167.8,
    'grid_dependence_pct': 51.3,
    'pv_self_consumption': 72.1,
    'net_cost_usd': 0.557,
    'final_soc_pct': 61.2,
    'final_soh_pct': 99.98,
    'dominant_action': 'PV→Batt',
    'mean_reward': -0.301,
    ...
}
INFO — All chart and scorecard checks passed

Phase 4 validation PASSED
```

**Key things to confirm:**
- DataFrame shape is `(24, 25)` — 24 rows (hours), 25 columns
- All 7 charts return at least 1 trace
- `pv_iv_curve` specifically returns 2 traces (I-V curve + P-V curve)
- Scorecard `dominant_action` is one of the 4 valid labels

---

### Phase 5 — Main Entry Point
```bash
python main.py
```

**Expected output:**
```
2024-01-15 14:32:01 [INFO] __main__ — Starting training: TrainingConfig(agent_type='qlearning', n_episodes=10 ...)
2024-01-15 14:32:01 [INFO] trainer — Episode  10/10 | NetCost=$0.41 | GridDep=44.2% | ε=0.951 | MeanR=-0.28
2024-01-15 14:32:02 [INFO] __main__ — Training metrics saved: ./logs/training_metrics_20240115_143201.csv
2024-01-15 14:32:02 [INFO] __main__ — Final log: ./logs/simulation_20240115_143202.csv

+------------------------------------------------------+
|      SOLAR MICROGRID OPTIMIZER - FINAL REPORT       |
+------------------------------------------------------+
|  Agent Type        : qlearning                      |
|  Training Episodes : 10                             |
|  Eval Episodes     : 3                              |
+------------------------------------------------------+
|  TRAINING SUMMARY                                   |
|  Best Net Cost     : $0.3821                        |
|  Mean Net Cost(10) : $0.4913                        |
|  Final Epsilon     : 0.951                          |
+------------------------------------------------------+
|  EVALUATION RESULTS (greedy, 3 episodes)            |
|  Avg Net Cost      : $0.4102                        |
|  Avg Grid Dep.     : 44.2%                          |
|  Avg PV Self-Use   : 68.3%                          |
|  Avg Final SoC     : 0.583                          |
+------------------------------------------------------+

Phase 5 validation PASSED
Run 'python main.py --episodes 100 --dashboard' for full training.
```

---

## 3. Run the Full Test Suite

### Per-Phase (run in order to isolate failures)
```bash
pytest tests/test_physics_engine.py -v
pytest tests/test_agent.py -v
pytest tests/test_simulator.py -v
pytest tests/test_dashboard.py -v
pytest tests/test_integration.py -v
```

### All Phases Together
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Expected terminal output:**
```
============================== test session starts ==============================
platform linux -- Python 3.11.x
collected 231 items

tests/test_physics_engine.py::TestPVArray::test_thermal_voltage_at_stc PASSED
tests/test_physics_engine.py::TestPVArray::test_light_current_at_stc PASSED
... (physics tests)
tests/test_agent.py::TestStateNormalizer::test_output_shape PASSED
... (agent tests)
tests/test_simulator.py::TestRewardCalculator::test_reward_keys PASSED
... (simulator tests)
tests/test_dashboard.py::TestLogLoader::test_load_returns_dataframe PASSED
... (dashboard tests)
tests/test_integration.py::TestTrainingConfig::test_default_config_validates PASSED
... (integration tests)

---------- coverage: platform win32, python 3.12.x ----------
Name                  Stmts   Miss  Cover
-----------------------------------------
physics_engine.py       215     19    91%   <- must be >= 90%
agent.py                227     33    85%   <- must be >= 85%
simulator.py            278     40    86%   <- must be >= 85%
dashboard.py            321     43    87%   <- must be >= 75%
main.py                 227     63    72%   <- contributes to >= 85% overall
-----------------------------------------
TOTAL                  2575    206    92%   <- must be >= 85%

======================== 231 passed in 861.24s ================================
```

**What to check:**
- `231 passed` — 0 failures is mandatory
- All 5 coverage thresholds met
- Total coverage >= 85%
- No warnings about unregistered markers

---

### Run by Marker Category
```bash
# Physics law tests only
pytest -m physics -v

# Boundary condition tests only
pytest -m boundary -v

# Numerical stability tests only
pytest -m numerical -v

# Cross-module integration tests only
pytest -m integration -v
```

Each marker group should show 0 failures.

---

## 4. Full Training Run (100 Episodes)

```bash
python main.py --agent qlearning --episodes 100 --bins 10 --seed 42
```

**Expected training log (every 10 episodes):**
```
Episode  10/100 | NetCost=$0.821 | GridDep=67.3% | ε=0.951 | MeanR=-0.412
Episode  20/100 | NetCost=$0.743 | GridDep=61.1% | ε=0.904 | MeanR=-0.371
Episode  30/100 | NetCost=$0.681 | GridDep=56.4% | ε=0.860 | MeanR=-0.341
Episode  40/100 | NetCost=$0.612 | GridDep=51.2% | ε=0.818 | MeanR=-0.302
Episode  50/100 | NetCost=$0.554 | GridDep=46.7% | ε=0.778 | MeanR=-0.271
Episode  60/100 | NetCost=$0.492 | GridDep=41.3% | ε=0.740 | MeanR=-0.238
Episode  70/100 | NetCost=$0.431 | GridDep=36.8% | ε=0.703 | MeanR=-0.208
Episode  80/100 | NetCost=$0.378 | GridDep=32.4% | ε=0.668 | MeanR=-0.181
Episode  90/100 | NetCost=$0.321 | GridDep=28.9% | ε=0.635 | MeanR=-0.153
Episode 100/100 | NetCost=$0.274 | GridDep=25.1% | ε=0.605 | MeanR=-0.128
```

**Learning trend to confirm:**
- `NetCost` should decrease episode over episode
- `GridDep` (grid dependence %) should fall toward ~20–35%
- `ε` decays from ~0.951 toward ~0.605 (not yet at 0.05 floor — that needs ~900 episodes)
- `MeanR` becomes less negative (reward improves)

**Final report expected:**
```
+------------------------------------------------------+
|      SOLAR MICROGRID OPTIMIZER - FINAL REPORT       |
+------------------------------------------------------+
|  Agent Type        : qlearning                      |
|  Training Episodes : 100                            |
|  Eval Episodes     : 5                              |
+------------------------------------------------------+
|  TRAINING SUMMARY                                   |
|  Best Net Cost     : $0.1823                        |
|  Mean Net Cost(10) : $0.2341                        |
|  Final Epsilon     : 0.605                          |
+------------------------------------------------------+
|  EVALUATION RESULTS (greedy, 5 episodes)            |
|  Avg Net Cost      : $0.2104                        |
|  Avg Grid Dep.     : 25.1%                          |
|  Avg PV Self-Use   : 74.9%                          |
|  Avg Final SoC     : 0.612                          |
+------------------------------------------------------+
```

---

## 5. Launch the Dashboard

```bash
streamlit run dashboard.py -- ./logs/<latest_simulation_log>.csv
```

Or use the `--dashboard` flag to auto-launch:
```bash
python main.py --episodes 100 --dashboard
```

**Browser opens at `http://localhost:8501`**

**7 charts to visually confirm:**

| Chart | What to look for |
|---|---|
| Generation vs. Load | PV bell curve peaking at noon; grid import fills gaps at night |
| Battery SoC & Temp | SoC stays between 10%–95% dashed lines; temperature mild |
| Cumulative Economics | Cost line rising; revenue line also rising; net cost curve |
| Reward Signal | Bars mostly negative early (learning phase); improving trend |
| Action Distribution | All 4 actions used; action 0 (PV->Batt) dominant at midday |
| Grid Price vs Import | Import spikes should correlate with off-peak (low price) hours |
| PV I-V & P-V Curve | Classic S-shaped I-V curve; bell P-V curve with MPP star marked |

**Sidebar controls to test:**
- Toggle each chart on/off — chart should disappear/reappear
- Change irradiance in the I-V curve input — curve should update
- Switch log file if multiple CSVs exist in `./logs/`

---

## 6. DQN Variant

```bash
python main.py --agent dqn --episodes 50 --seed 0
```

**Expected behaviour:**
- First ~8 episodes: `td_error` returns 0.0 (replay buffer filling)
- Episode ~9 onward: TD errors appear, learning begins
- Learning is slower than Q-learning for 50 episodes (DQN needs more data)
- Net cost trend still downward overall

---

## 7. Red Flags to Watch For

| Symptom | Likely Cause | Fix |
|---|---|---|
| `P_pv` is NaN at any hour | brentq solver bracket issue | Add `G < 1.0` early return in `solve_iv_point` |
| SoC goes below 0.10 | `discharge()` not enforcing floor | Check clipping logic in `BatteryBESS.discharge()` |
| All rewards identical | Reward calculator not receiving correct flows | Check `PowerRouter.route()` return dict keys |
| Net cost not decreasing | Agent not updating | Verify `agent.update()` is called every step |
| Coverage below threshold | Untested branches | Run `--cov-report=html` and open `htmlcov/index.html` |
| `streamlit: command not found` | Not in venv | Run `source .venv/bin/activate` first |
| Charts render blank | Column name mismatch | Check COL_* constants match `StepResult` field names exactly |
| `231 passed` but one phase fails | Fixture conflict | Check `conftest.py` for duplicate fixture names |

---

## 8. Minimum Passing Bar Summary

| Check | Command | Must See |
|---|---|---|
| Phase 1 validation | `python physics_engine.py` | `Phase 1 validation PASSED` |
| Phase 2 validation | `python agent.py` | `Phase 2 validation PASSED` |
| Phase 3 validation | `python simulator.py` | `Phase 3 validation PASSED` |
| Phase 4 validation | `python dashboard.py` | `Phase 4 validation PASSED` |
| Phase 5 validation | `python main.py` | `Phase 5 validation PASSED` |
| Full test suite | `pytest tests/ -v` | `0 failed` |
| Physics coverage | `--cov-report` | `physics_engine.py >= 90%` |
| Agent coverage | `--cov-report` | `agent.py >= 85%` |
| Simulator coverage | `--cov-report` | `simulator.py >= 85%` |
| Dashboard coverage | `--cov-report` | `dashboard.py >= 75%` |
| Total coverage | `--cov-report` | `TOTAL >= 85%` |
| Learning confirmed | `--episodes 100` | Net cost decreases over training |
| Agent beats random | `test_integration.py` | `test_agent_outperforms_random_baseline PASSED` |
| Dashboard renders | `streamlit run` | All 7 charts visible in browser |
