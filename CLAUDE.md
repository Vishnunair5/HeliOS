# Solar Microgrid Power Optimizer — CLAUDE.md

## Section 1 — Project Overview
High-fidelity solar microgrid power optimizer using physics-based modeling, tabular/DQN RL agent, and real-time dashboard.

## Section 2 — Module Architecture & Import Rules

| Module | Purpose | May import from |
|--------|---------|----------------|
| `physics_engine.py` | Physical models | numpy, scipy only |
| `agent.py` | RL agent | numpy, torch only |
| `simulator.py` | Simulation loop | physics_engine, agent |
| `dashboard.py` | Streamlit UI | simulator, physics_engine |
| `main.py` | Entry point | all modules |

**Cross-module constraint:** `agent.py` must NOT import from `simulator.py`, `dashboard.py`, or `main.py`.

## Section 3 — Naming Conventions

- Variables: `snake_case`, unit suffix where ambiguous (`power_kw`, `temp_k`, `dt_hours`)
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` at module level
- Test functions: `test_<what_is_being_tested>`
- Test classes: `Test<ClassName>`
- Private methods: `_leading_underscore`

## Section 4 — Code Quality Standards

- Full Python type annotations on all parameters and return types
- Google-style docstrings on all public classes and methods
- All divisions guard against zero denominator (guard with `max(val, 1e-9)`)
- Inputs clipped at physical boundaries; no silent NaN propagation
- Use `float(np.nan_to_num(x))` to sanitize external numeric inputs before arithmetic
- No `pass`, `TODO`, or stub methods in production code

## Section 5 — Physics Standards

- PVArray: Single-Diode Equivalent Circuit (pvlib string-level convention, `nNsVth = n_ideal * n_series * V_th`)
- BatteryBESS: Piecewise-linear OCV curve, rainflow-lite degradation
- SI units throughout: W, J, K, A, V — kW only at the interface layer
- Edge cases always handled: G=0 (night), SoC at floor/ceiling, T extremes

## Section 6 — RL Agent Separation of Concerns

- Agent observes normalized state → selects action → receives reward → updates Q-values
- **Agent NEVER computes rewards internally.** Reward is computed externally by `simulator.py`.
- Checkpoints saved to `./checkpoints/` directory
- Both `QLearningAgent` and `DQNAgent` expose identical public method signatures

## Section 7 — Testing Standards

### Markers
Register in `conftest.py` via `pytest_configure`:
```
boundary   — edge cases and boundary conditions
physics    — physics or RL correctness tests
numerical  — numerical stability and NaN/Inf tests
```

### Coverage Targets

| Module | Minimum coverage |
|--------|-----------------|
| `physics_engine.py` | ≥ 90% |
| `agent.py` | ≥ 85% |
| `simulator.py` | ≥ 85% |

### Phase Completion Gates
- `python <module>.py` prints `"Phase N validation PASSED"`
- `pytest tests/test_<module>.py` reports 0 failures, 0 errors
- Coverage threshold met
- All prior phase tests still pass

### Test Rules
- No smoke tests — every assertion checks specific numerical or structural outcomes
- No bare `assert result is not None`
- Never modify a test to make it pass — fix the implementation

## Section 8 — Dependencies
See `requirements.txt`. No external dependencies beyond that list.

## Section 9 — Logging

- Use `logging` module in all production code paths
- `print()` only inside `if __name__ == "__main__":` blocks
- Log training steps at DEBUG; warnings/errors at appropriate levels
- Module logger: `log = logging.getLogger(__name__)`

## Phase Status
- [x] Phase 1 — Foundation, Physics Engine & Tests
- [x] Phase 2 — Optimization Agent & Tests
- [ ] Phase 3 — Simulator
- [ ] Phase 4 — Dashboard
- [ ] Phase 5 — Integration & Optimization
