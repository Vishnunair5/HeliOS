# ☀️ Solar Microgrid Power Optimizer

A high-fidelity simulation and reinforcement learning system that optimizes power routing in a solar microgrid — minimizing electricity costs, reducing grid dependence, and maximizing battery longevity across a 24-hour horizon.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

The system simulates a residential/industrial solar microgrid consisting of:
- A **Photovoltaic (PV) array** modeled using the Single-Diode Equivalent Circuit equation
- A **Lithium-ion Battery (BESS)** with real electrochemical dynamics, degradation tracking, and thermal modeling
- A **Utility grid connection** with time-of-use (TOU) pricing

A **Reinforcement Learning agent** (Q-learning or Deep Q-Network) learns to route power optimally by choosing between 4 strategies at each hour:

| Action | Strategy |
|--------|-----------|
| 0 | PV → Load; excess → Battery |
| 1 | PV → Load; excess → Grid (sell) |
| 2 | Battery → Load; Grid supplements |
| 3 | Grid → Load; Grid → Battery (arbitrage) |

Over training, the agent learns **peak shaving**, **load shifting**, and **energy arbitrage** automatically.

---

## Architecture

```
solar_microgrid/
├── physics_engine.py   # PV single-diode model, BESS dynamics, environment profiles
├── agent.py            # Q-learning & DQN agents, state normalizer, MDP definition
├── simulator.py        # 24-hour step loop, reward calculation, power routing, logging
├── dashboard.py        # Streamlit + Plotly telemetry dashboard
├── main.py             # Entry point — training, evaluation, CLI, report
├── requirements.txt    # Pinned dependencies
└── tests/
    ├── test_physics_engine.py
    ├── test_agent.py
    ├── test_simulator.py
    ├── test_dashboard.py
    └── test_integration.py
```

---

## Physics Models

### PV Array — Single-Diode Equivalent Circuit
```
I = I_L - I_0 * [exp((V + I·R_s) / (n·V_th)) - 1] - (V + I·R_s) / R_sh
```
- Light current and saturation current derived dynamically from irradiance (G) and cell temperature (T)
- Maximum Power Point (MPP) solved via voltage sweep at each timestep
- Realistic cloud transients applied to irradiance profile

### Battery BESS — Discrete-Time Electrochemical Model
- State-of-charge dependent open-circuit voltage (OCV curve)
- Internal resistance losses on charge and discharge
- Coulombic efficiency: η_charge = 0.96, η_discharge = 0.94
- Rainflow-lite cycle counting for State of Health (SoH) degradation
- Thermal mass modeling — temperature rises under load

### Reward Function
```
R = - [Grid Import Cost + SoC Violation Penalty + Thermal Cycling Penalty]
    + [Grid Export Revenue]
```

---

## Dashboard

The Streamlit dashboard visualizes the simulation results with 7 interactive Plotly charts:

| Chart | Description |
|-------|-------------|
| Generation vs. Load | PV output vs demand vs grid flows over 24 hours |
| Battery SoC & Temperature | State of charge tracking with reference limit lines |
| Cumulative Economics | Running cost, revenue, and net cost over the episode |
| Reward Signal | Per-step reward bars with rolling mean overlay |
| Action Distribution | Breakdown of which strategies the agent used |
| Grid Price vs Import | TOU pricing bands vs agent import behavior |
| PV I-V & P-V Curve | Real-time physics curve with MPP annotation |

---

## Quickstart

### 1. Clone and set up environment
```bash
git clone https://github.com/Vishnunair5/HeliOS.git
cd HeliOS/solar_microgrid

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Validate each module
```bash
python physics_engine.py   # Phase 1 validation PASSED
python agent.py            # Phase 2 validation PASSED
python simulator.py        # Phase 3 validation PASSED
python dashboard.py        # Phase 4 validation PASSED
python main.py             # Phase 5 validation PASSED
```

### 3. Run training
```bash
# Quick run (10 episodes)
python main.py

# Full training — Q-learning (recommended first run)
python main.py --agent qlearning --episodes 100

# Extended training for best results
python main.py --agent qlearning --episodes 500

# Deep Q-Network variant
python main.py --agent dqn --episodes 100
```

### 4. Launch the dashboard
```bash
python -m streamlit run dashboard.py
```
Opens at `http://localhost:8501` — auto-loads the latest simulation log.

---

## CLI Options

```
python main.py [OPTIONS]

Options:
  --agent       Agent type: qlearning or dqn (default: qlearning)
  --episodes    Number of training episodes (default: 100)
  --bins        Q-table bins per dimension, Q-learning only (default: 10)
  --dt          Timestep in hours: 1.0 or 0.25 (default: 1.0)
  --seed        Random seed for reproducibility (default: 42)
  --eval        Number of greedy evaluation episodes (default: 5)
  --dashboard   Launch Streamlit dashboard after training
```

---

## Run Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# By category
pytest -m physics -v      # Physical law verification
pytest -m boundary -v     # Edge case / boundary tests
pytest -m numerical -v    # NaN / Inf stability tests
pytest -m integration -v  # Cross-module integration tests
```

**Coverage targets:**

| Module | Minimum |
|--------|---------|
| `physics_engine.py` | ≥ 90% |
| `agent.py` | ≥ 85% |
| `simulator.py` | ≥ 85% |
| `dashboard.py` | ≥ 75% |
| Overall | ≥ 85% |

---

## What Good Training Looks Like

After 500 episodes the agent should show clear improvement:

| Metric | Episode 1 | Episode 500 |
|--------|-----------|-------------|
| Grid Dependence | ~90% | ~25–35% |
| Net Cost | ~$8.00 | ~$0.20–0.50 |
| Mean Reward | ~-0.40 | ~-0.10 |
| PV Self-Use | ~10% | ~65–75% |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Physics simulation | NumPy, SciPy |
| Reinforcement learning | PyTorch (DQN), NumPy (Q-learning) |
| Data handling | Pandas |
| Visualization | Plotly, Streamlit |
| Testing | pytest, pytest-cov |

---

## Project Structure Notes

- `simulator.py` is the only module that imports from both `physics_engine.py` and `agent.py`
- `main.py` is the only module that imports from all other modules
- `dashboard.py` reads exclusively from CSV logs — no live simulation coupling
- All simulation logs saved to `./logs/`, checkpoints to `./checkpoints/`

---

## License

MIT License — free to use, modify, and distribute.