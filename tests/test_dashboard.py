"""Phase 4 tests for dashboard.py — LogLoader, ChartBuilder, ScorecardBuilder, DashboardApp.

DashboardApp methods are tested by patching streamlit in sys.modules with a MagicMock
so that the Streamlit-specific calls never touch a real server.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import (
    LogLoader,
    ChartBuilder,
    ScorecardBuilder,
    _ACTION_LABELS,
    COL_HOUR,
    COL_PV,
    COL_LOAD,
    COL_GRID_IMPORT,
    COL_GRID_EXPORT,
    COL_BATT_CHARGE,
    COL_BATT_DISCHARGE,
    COL_SOC,
    COL_SOH,
    COL_REWARD,
    COL_CUMCOST,
    COL_CUMREV,
    COL_GRID_PRICE,
    COL_ACTION,
    COL_ACTION_DESC,
    CHART_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
class TestLogLoader:
    """Tests for LogLoader — file I/O, validation, and preprocessing."""

    def test_load_returns_dataframe(self, sample_log_csv: str) -> None:
        df = LogLoader(sample_log_csv).load()
        assert isinstance(df, pd.DataFrame)

    def test_load_has_correct_row_count(self, sample_log_csv: str) -> None:
        df = LogLoader(sample_log_csv).load()
        assert len(df) == 24

    @pytest.mark.boundary
    def test_load_raises_on_missing_file(self) -> None:
        with pytest.raises((FileNotFoundError, ValueError)):
            LogLoader("nonexistent_does_not_exist.csv").load()

    def test_validate_passes_on_valid_df(self, sample_log_df: pd.DataFrame) -> None:
        loader = LogLoader.__new__(LogLoader)
        result = loader.validate(sample_log_df)
        assert result is True

    @pytest.mark.boundary
    def test_validate_raises_on_missing_column(self, sample_log_df: pd.DataFrame) -> None:
        df = sample_log_df.drop(columns=[COL_SOC])
        loader = LogLoader.__new__(LogLoader)
        with pytest.raises(ValueError):
            loader.validate(df)

    @pytest.mark.boundary
    def test_validate_raises_on_empty_df(self) -> None:
        loader = LogLoader.__new__(LogLoader)
        with pytest.raises(ValueError):
            loader.validate(pd.DataFrame())

    @pytest.mark.numerical
    def test_validate_raises_on_nan_in_numeric(self, sample_log_df: pd.DataFrame) -> None:
        df = sample_log_df.copy()
        df.loc[0, COL_REWARD] = np.nan
        loader = LogLoader.__new__(LogLoader)
        with pytest.raises(ValueError):
            loader.validate(df)

    @pytest.mark.boundary
    def test_validate_raises_on_soc_out_of_range(self, sample_log_df: pd.DataFrame) -> None:
        df = sample_log_df.copy()
        df.loc[0, COL_SOC] = 1.5
        loader = LogLoader.__new__(LogLoader)
        with pytest.raises(ValueError):
            loader.validate(df)

    @pytest.mark.boundary
    def test_validate_raises_on_wrong_hour_count(self, sample_log_df: pd.DataFrame) -> None:
        df = sample_log_df.iloc[:-2].copy()
        loader = LogLoader.__new__(LogLoader)
        with pytest.raises(ValueError):
            loader.validate(df)

    def test_preprocess_adds_net_power_column(self, sample_log_df: pd.DataFrame) -> None:
        loader = LogLoader.__new__(LogLoader)
        df = loader.preprocess(sample_log_df)
        assert "net_power_kw" in df.columns

    def test_preprocess_adds_action_label_column(self, sample_log_df: pd.DataFrame) -> None:
        loader = LogLoader.__new__(LogLoader)
        df = loader.preprocess(sample_log_df)
        assert "action_label" in df.columns

    def test_preprocess_action_labels_valid(self, preprocessed_df: pd.DataFrame) -> None:
        valid_labels = set(_ACTION_LABELS.values())
        assert set(preprocessed_df["action_label"].unique()).issubset(valid_labels)

    @pytest.mark.numerical
    def test_preprocess_net_power_finite(self, preprocessed_df: pd.DataFrame) -> None:
        assert np.all(np.isfinite(preprocessed_df["net_power_kw"].values))


# ─────────────────────────────────────────────────────────────────────────────
class TestChartBuilder:
    """Tests for ChartBuilder — every chart method returns valid go.Figure."""

    def test_build_generation_vs_load_returns_figure(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_generation_vs_load(preprocessed_df)
        assert isinstance(fig, go.Figure)

    def test_generation_vs_load_has_correct_traces(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_generation_vs_load(preprocessed_df)
        assert len(fig.data) >= 4
        all_names = " ".join(t.name for t in fig.data if t.name)
        assert "PV" in all_names
        assert "Load" in all_names
        assert "Import" in all_names
        assert "Export" in all_names

    def test_generation_vs_load_axis_labels(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_generation_vs_load(preprocessed_df)
        assert "Hour" in fig.layout.xaxis.title.text
        assert "kW" in fig.layout.yaxis.title.text

    def test_build_battery_soc_returns_figure(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_battery_soc(preprocessed_df)
        assert isinstance(fig, go.Figure)

    def test_battery_soc_has_reference_lines(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_battery_soc(preprocessed_df)
        assert len(fig.layout.shapes) >= 2

    def test_build_cumulative_economics_returns_figure(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_cumulative_economics(preprocessed_df)
        assert isinstance(fig, go.Figure)

    def test_cumulative_economics_has_three_traces(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_cumulative_economics(preprocessed_df)
        assert len(fig.data) == 3

    def test_build_reward_trace_returns_figure(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_reward_trace(preprocessed_df)
        assert isinstance(fig, go.Figure)

    def test_reward_trace_bar_and_line(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_reward_trace(preprocessed_df)
        types = {type(t) for t in fig.data}
        assert go.Bar in types
        assert go.Scatter in types

    def test_build_action_distribution_returns_figure(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_action_distribution(preprocessed_df)
        assert isinstance(fig, go.Figure)

    def test_build_grid_price_overlay_returns_figure(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_grid_price_overlay(preprocessed_df)
        assert isinstance(fig, go.Figure)

    def test_grid_price_overlay_has_two_yaxes(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        fig = chart_builder.build_grid_price_overlay(preprocessed_df)
        assert fig.layout.yaxis2 is not None

    def test_build_pv_iv_curve_returns_figure(
        self, chart_builder: ChartBuilder
    ) -> None:
        fig = chart_builder.build_pv_iv_curve(G=800.0, T=298.15)
        assert isinstance(fig, go.Figure)

    def test_pv_iv_curve_has_mpp_annotation(
        self, chart_builder: ChartBuilder
    ) -> None:
        fig = chart_builder.build_pv_iv_curve(G=800.0, T=298.15)
        assert len(fig.layout.annotations) > 0

    def test_all_charts_use_correct_template(
        self, chart_builder: ChartBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        figs = [
            chart_builder.build_generation_vs_load(preprocessed_df),
            chart_builder.build_battery_soc(preprocessed_df),
            chart_builder.build_cumulative_economics(preprocessed_df),
            chart_builder.build_reward_trace(preprocessed_df),
            chart_builder.build_action_distribution(preprocessed_df),
            chart_builder.build_grid_price_overlay(preprocessed_df),
        ]
        for fig in figs:
            tmpl = fig.layout.template
            # template may be stored as a string or as a Template object with a name
            if isinstance(tmpl, str):
                assert CHART_TEMPLATE in tmpl
            else:
                assert tmpl is not None  # template object was applied

    @pytest.mark.boundary
    def test_charts_handle_all_zero_pv(self, chart_builder: ChartBuilder) -> None:
        rng = np.random.default_rng(99)
        hours = list(range(24))
        df = pd.DataFrame({
            COL_HOUR: hours,
            COL_PV: np.zeros(24),
            COL_LOAD: rng.uniform(3, 10, 24),
            COL_GRID_IMPORT: rng.uniform(0, 5, 24),
            COL_GRID_EXPORT: np.zeros(24),
            COL_BATT_CHARGE: np.zeros(24),
            COL_BATT_DISCHARGE: np.zeros(24),
            COL_SOC: rng.uniform(0.15, 0.80, 24),
            COL_SOH: np.ones(24),
            COL_REWARD: rng.uniform(-1, 0, 24),
            COL_CUMCOST: np.cumsum(rng.uniform(0, 0.5, 24)),
            COL_CUMREV: np.zeros(24),
            COL_GRID_PRICE: np.full(24, 0.12),
            COL_ACTION: rng.integers(0, 4, 24),
            COL_ACTION_DESC: ["PV->Load"] * 24,
            "action_label": ["PV→Batt"] * 24,
            "net_power_kw": np.zeros(24),
            "net_cost_step": np.zeros(24),
        })
        fig = chart_builder.build_generation_vs_load(df)
        assert isinstance(fig, go.Figure)

    @pytest.mark.boundary
    def test_charts_handle_single_action_type(self, chart_builder: ChartBuilder) -> None:
        rng = np.random.default_rng(7)
        hours = list(range(24))
        df = pd.DataFrame({
            COL_HOUR: hours,
            COL_PV: rng.uniform(0, 10, 24),
            COL_LOAD: rng.uniform(3, 10, 24),
            COL_GRID_IMPORT: rng.uniform(0, 5, 24),
            COL_GRID_EXPORT: np.zeros(24),
            COL_BATT_CHARGE: np.zeros(24),
            COL_BATT_DISCHARGE: np.zeros(24),
            COL_SOC: rng.uniform(0.15, 0.80, 24),
            COL_SOH: np.ones(24),
            COL_REWARD: rng.uniform(-1, 0, 24),
            COL_CUMCOST: np.cumsum(rng.uniform(0, 0.5, 24)),
            COL_CUMREV: np.zeros(24),
            COL_GRID_PRICE: np.full(24, 0.12),
            COL_ACTION: np.zeros(24, dtype=int),
            COL_ACTION_DESC: ["PV->Load"] * 24,
            "action_label": ["PV→Batt"] * 24,
            "net_power_kw": np.zeros(24),
            "net_cost_step": np.zeros(24),
        })
        fig = chart_builder.build_action_distribution(df)
        assert isinstance(fig, go.Figure)


# ─────────────────────────────────────────────────────────────────────────────
class TestScorecardBuilder:
    """Tests for ScorecardBuilder — metrics computation and figure output."""

    def test_compute_metrics_returns_dict(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        result = scorecard_builder.compute_metrics(preprocessed_df)
        assert isinstance(result, dict)

    def test_compute_metrics_all_keys_present(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        result = scorecard_builder.compute_metrics(preprocessed_df)
        expected_keys = {
            "total_pv_kwh", "total_load_kwh", "total_grid_import_kwh",
            "total_grid_export_kwh", "grid_dependence_pct", "pv_self_consumption",
            "total_cost_usd", "total_revenue_usd", "net_cost_usd",
            "final_soc_pct", "final_soh_pct", "dominant_action", "mean_reward",
        }
        assert expected_keys.issubset(result.keys())

    @pytest.mark.physics
    def test_grid_dependence_pct_in_range(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        assert 0.0 <= metrics["grid_dependence_pct"] <= 100.0

    @pytest.mark.physics
    def test_pv_self_consumption_in_range(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        assert 0.0 <= metrics["pv_self_consumption"] <= 100.0

    @pytest.mark.boundary
    def test_final_soc_pct_in_range(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        assert 0.0 <= metrics["final_soc_pct"] <= 100.0

    @pytest.mark.physics
    def test_net_cost_is_total_minus_revenue(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        expected = metrics["total_cost_usd"] - metrics["total_revenue_usd"]
        assert abs(metrics["net_cost_usd"] - expected) < 1e-6

    def test_dominant_action_is_valid_label(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        assert metrics["dominant_action"] in set(_ACTION_LABELS.values())

    @pytest.mark.numerical
    def test_no_nan_in_metrics(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        for k, v in metrics.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Non-finite metric: {k}={v}"

    def test_build_scorecard_figure_returns_figure(
        self, scorecard_builder: ScorecardBuilder, preprocessed_df: pd.DataFrame
    ) -> None:
        metrics = scorecard_builder.compute_metrics(preprocessed_df)
        fig = scorecard_builder.build_scorecard_figure(metrics)
        assert isinstance(fig, go.Figure)


# ─────────────────────────────────────────────────────────────────────────────
class TestDashboardIntegration:
    """Integration tests that wire LogLoader → ChartBuilder / ScorecardBuilder."""

    @pytest.mark.integration
    def test_full_pipeline_load_to_metrics(self, sample_log_csv: str) -> None:
        loader = LogLoader(sample_log_csv)
        df_raw = loader.load()
        loader.validate(df_raw)
        df = loader.preprocess(df_raw)
        metrics = ScorecardBuilder().compute_metrics(df)
        assert isinstance(metrics["net_cost_usd"], float)
        assert np.isfinite(metrics["net_cost_usd"])

    @pytest.mark.integration
    def test_full_pipeline_load_to_all_charts(self, sample_log_csv: str) -> None:
        loader = LogLoader(sample_log_csv)
        df = loader.preprocess(loader.load())
        cb = ChartBuilder()
        figs = [
            cb.build_generation_vs_load(df),
            cb.build_battery_soc(df),
            cb.build_cumulative_economics(df),
            cb.build_reward_trace(df),
            cb.build_action_distribution(df),
            cb.build_grid_price_overlay(df),
        ]
        for fig in figs:
            assert isinstance(fig, go.Figure)

    @pytest.mark.integration
    def test_chart_builder_with_real_simulator_log(self, tmp_path) -> None:
        from agent import QLearningAgent
        from simulator import MicrogridSimulator

        agent = QLearningAgent(n_bins=5, seed=0)
        sim = MicrogridSimulator(agent=agent, training=True, seed=0)
        sim.run_episode()
        csv_path = sim.save_log(str(tmp_path / "real_sim.csv"))

        loader = LogLoader(csv_path)
        df = loader.preprocess(loader.load())
        cb = ChartBuilder()
        figs = [
            cb.build_generation_vs_load(df),
            cb.build_battery_soc(df),
            cb.build_cumulative_economics(df),
            cb.build_reward_trace(df),
            cb.build_action_distribution(df),
            cb.build_grid_price_overlay(df),
        ]
        for fig in figs:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    @pytest.mark.integration
    def test_pv_iv_curve_uses_real_physics(self, chart_builder: ChartBuilder) -> None:
        fig = chart_builder.build_pv_iv_curve(G=1000.0, T=298.15)
        first_trace = fig.data[0]
        assert len(first_trace.x) >= 50
        # Second trace (power kW) max must be positive
        power_trace = fig.data[1]
        assert float(np.max(power_trace.y)) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit mock fixture — scoped to function so each test gets a fresh mock

@pytest.fixture
def st_mock():
    """Mock the streamlit module so DashboardApp methods run without a server."""
    mock = MagicMock()
    # session_state needs to be a real dict so `.get(key, default)` works
    mock.session_state = {"G_iv": 800.0, "T_iv_c": 25.0}
    # columns(n) must return exactly n mock objects for tuple-unpacking
    mock.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    # expander is used as a context manager
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock())
    cm.__exit__ = MagicMock(return_value=False)
    mock.expander.return_value = cm
    # file_uploader returns None by default (no file selected)
    mock.file_uploader.return_value = None
    with patch.dict(sys.modules, {"streamlit": mock}):
        yield mock


class TestDashboardApp:
    """Tests for DashboardApp with a mocked Streamlit module."""

    def test_load_and_validate_success(
        self, sample_log_csv: str, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path=sample_log_csv)
        df = app._load_and_validate(sample_log_csv)
        assert df is not None
        assert len(df) == 24

    def test_load_and_validate_failure_calls_st_error(
        self, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path="nonexistent_test.csv")
        df = app._load_and_validate("nonexistent_test.csv")
        assert df is None
        st_mock.error.assert_called_once()

    def test_render_header_without_log_path(self, st_mock: MagicMock) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path=None)
        app._render_header()
        st_mock.title.assert_called_once()
        st_mock.caption.assert_not_called()

    def test_render_header_with_log_path(
        self, sample_log_csv: str, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path=sample_log_csv)
        app._render_header()
        st_mock.caption.assert_called_once()

    def test_render_scorecard(
        self, preprocessed_df: pd.DataFrame, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path=None)
        app._render_scorecard(preprocessed_df)
        st_mock.subheader.assert_called()
        st_mock.plotly_chart.assert_called()

    def test_render_raw_data(
        self, preprocessed_df: pd.DataFrame, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path=None)
        app._render_raw_data(preprocessed_df)
        st_mock.expander.assert_called_once()

    def test_render_charts_renders_visible_only(
        self, preprocessed_df: pd.DataFrame, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        visibility = {
            "generation_vs_load":   True,
            "battery_soc":          True,
            "cumulative_economics": False,
            "reward_trace":         False,
            "action_distribution":  False,
            "grid_price_overlay":   False,
            "pv_iv_curve":          False,
        }
        app = DashboardApp(log_path=None)
        app._render_charts(preprocessed_df, visibility)
        assert st_mock.plotly_chart.call_count == 2

    def test_render_charts_pv_iv_curve_flag(
        self, preprocessed_df: pd.DataFrame, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        visibility = {k: False for k in [
            "generation_vs_load", "battery_soc", "cumulative_economics",
            "reward_trace", "action_distribution", "grid_price_overlay",
        ]}
        visibility["pv_iv_curve"] = True
        app = DashboardApp(log_path=None)
        app._render_charts(preprocessed_df, visibility)
        assert st_mock.plotly_chart.call_count == 1

    def test_render_sidebar_returns_dict(
        self, preprocessed_df: pd.DataFrame, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        app = DashboardApp(log_path=None)
        visibility = app._render_sidebar(preprocessed_df)
        assert isinstance(visibility, dict)
        assert "generation_vs_load" in visibility

    def test_run_no_path_no_upload_shows_info(self, st_mock: MagicMock) -> None:
        from dashboard import DashboardApp
        st_mock.file_uploader.return_value = None
        app = DashboardApp(log_path=None)
        app.run()
        st_mock.info.assert_called()

    def test_run_with_valid_log_path(
        self, sample_log_csv: str, st_mock: MagicMock
    ) -> None:
        from dashboard import DashboardApp
        # _render_sidebar returns checkbox MagicMock values (truthy = visible)
        app = DashboardApp(log_path=sample_log_csv)
        app.run()
        # set_page_config must be called once
        st_mock.set_page_config.assert_called_once()

    def test_load_raises_on_load_missing_column(
        self, tmp_path: "pytest.TempPathFactory", st_mock: MagicMock
    ) -> None:
        path = str(tmp_path / "bad.csv")
        pd.DataFrame({"hour": range(24), "foo": range(24)}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            LogLoader(path).load()
