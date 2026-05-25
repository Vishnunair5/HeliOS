"""Streamlit + Plotly telemetry dashboard for the solar microgrid simulator.

Reads simulation logs from CSV files produced by simulator.py and renders
interactive charts. Does not run simulations or import agent/simulator classes
directly (except build_pv_iv_curve, which uses PVArray for physics computation).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)

# ── Color palette — CLAUDE.md Section 8. Do not modify. ──────────────────────
COLOR_PV: str      = "#F5A623"   # Solar yellow
COLOR_BATT: str    = "#4A90D9"   # Battery blue
COLOR_GRID: str    = "#E74C3C"   # Grid red
COLOR_SAVINGS: str = "#27AE60"   # Savings green
COLOR_LOAD: str    = "#8E44AD"   # Load purple
COLOR_REWARD: str  = "#2C3E50"   # Reward dark slate

CHART_HEIGHT: int = 380
CHART_TEMPLATE: str = "plotly_white"

# ── Column name mappings — match StepResult field names exactly ───────────────
COL_HOUR: str           = "hour"
COL_PV: str             = "pv_power_kw"
COL_LOAD: str           = "load_kw"
COL_GRID_IMPORT: str    = "grid_import_kw"
COL_GRID_EXPORT: str    = "grid_export_kw"
COL_BATT_CHARGE: str    = "battery_charge_kw"
COL_BATT_DISCHARGE: str = "battery_discharge_kw"
COL_SOC: str            = "soc"
COL_SOH: str            = "soh"
COL_REWARD: str         = "reward"
COL_CUMCOST: str        = "cumulative_cost"
COL_CUMREV: str         = "cumulative_revenue"
COL_GRID_PRICE: str     = "grid_price"
COL_ACTION: str         = "action"
COL_ACTION_DESC: str    = "action_description"

# All columns required in a valid simulation log
_REQUIRED_COLUMNS: list[str] = [
    COL_HOUR, COL_PV, COL_LOAD, COL_GRID_IMPORT, COL_GRID_EXPORT,
    COL_BATT_CHARGE, COL_BATT_DISCHARGE, COL_SOC, COL_SOH, COL_REWARD,
    COL_CUMCOST, COL_CUMREV, COL_GRID_PRICE, COL_ACTION, COL_ACTION_DESC,
]

_NUMERIC_COLUMNS: list[str] = [
    COL_HOUR, COL_PV, COL_LOAD, COL_GRID_IMPORT, COL_GRID_EXPORT,
    COL_BATT_CHARGE, COL_BATT_DISCHARGE, COL_SOC, COL_SOH, COL_REWARD,
    COL_CUMCOST, COL_CUMREV, COL_GRID_PRICE, COL_ACTION,
]

_ACTION_LABELS: dict[int, str] = {
    0: "PV→Batt",
    1: "PV→Grid",
    2: "Batt→Load",
    3: "Grid→Batt",
}


class LogLoader:
    """Reads, validates, and preprocesses a simulation CSV log."""

    def __init__(self, log_path: str) -> None:
        """Initialize loader with path to a simulation CSV.

        Args:
            log_path: Absolute or relative path to the CSV file.
        """
        self.log_path = log_path

    def load(self) -> pd.DataFrame:
        """Read CSV from log_path and cast numeric columns to float64.

        Returns:
            DataFrame with all StepResult columns present.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If required columns are missing.
        """
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        df = pd.read_csv(self.log_path)

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col in _NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        log.info("Loaded %d rows from %s", len(df), self.log_path)
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate structural and physical correctness of the DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            True if all checks pass.

        Raises:
            ValueError: With a specific message for each failed check.
        """
        if df.empty:
            raise ValueError("DataFrame is empty.")

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col in _NUMERIC_COLUMNS:
            if col in df.columns and df[col].isna().any():
                raise ValueError(f"NaN values found in numeric column '{col}'.")

        hours = sorted(df[COL_HOUR].astype(int).tolist())
        if hours != list(range(24)):
            raise ValueError(
                f"Expected hour values 0-23 (24 rows). Got {len(hours)} rows "
                f"with hours: {hours[:5]}..."
            )

        if (df[COL_SOC] < 0.0).any() or (df[COL_SOC] > 1.0).any():
            bad = df.loc[(df[COL_SOC] < 0.0) | (df[COL_SOC] > 1.0), COL_SOC].tolist()
            raise ValueError(f"SoC values out of [0, 1]: {bad[:3]}")

        return True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns to the DataFrame.

        Args:
            df: Raw validated DataFrame.

        Returns:
            Enriched DataFrame with net_power_kw, net_cost_step, action_label columns.
        """
        df = df.copy()

        df["net_power_kw"] = (
            df[COL_PV] + df[COL_BATT_DISCHARGE]
            - df[COL_LOAD] - df[COL_BATT_CHARGE]
        )

        if "cost_grid_import" in df.columns and "revenue_grid_export" in df.columns:
            df["net_cost_step"] = df["cost_grid_import"] - df["revenue_grid_export"]
        else:
            df["net_cost_step"] = df[COL_GRID_IMPORT] * df[COL_GRID_PRICE] * 1.0

        df["action_label"] = (
            df[COL_ACTION].astype(int).map(_ACTION_LABELS).fillna("Unknown")
        )

        return df


class ChartBuilder:
    """Builds Plotly figures from preprocessed simulation DataFrames.

    Stateless — every method takes a DataFrame and returns a go.Figure.
    No Streamlit calls occur inside this class.
    """

    def __init__(self) -> None:
        """Initialize ChartBuilder. Stateless."""

    def build_generation_vs_load(self, df: pd.DataFrame) -> go.Figure:
        """Build time-series overlay of PV, load, grid import, and export.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Plotly Figure with four power traces.
        """
        fig = go.Figure()
        hours = df[COL_HOUR]

        fig.add_trace(go.Scatter(
            x=hours, y=df[COL_PV],
            name="PV Generation",
            fill="tozeroy",
            line=dict(color=COLOR_PV),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=df[COL_LOAD],
            name="Load Demand",
            line=dict(color=COLOR_LOAD, width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=df[COL_GRID_IMPORT],
            name="Grid Import",
            line=dict(color=COLOR_GRID, dash="dash"),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=df[COL_GRID_EXPORT],
            name="Grid Export",
            line=dict(color=COLOR_SAVINGS, dash="dot"),
            mode="lines",
        ))

        # On-peak band: 09:00–21:00
        fig.add_vrect(
            x0=9, x1=21,
            fillcolor="rgba(245, 166, 35, 0.08)",
            layer="below",
            line_width=0,
            annotation_text="On-Peak",
            annotation_position="top left",
        )

        fig.update_layout(
            title="Generation vs. Load Profile",
            xaxis_title="Hour of Day",
            yaxis_title="Power (kW)",
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def build_battery_soc(self, df: pd.DataFrame) -> go.Figure:
        """Build dual-axis chart of SoC (%) and battery temperature.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Plotly Figure with SoC on primary axis and temperature on secondary.
        """
        hours = df[COL_HOUR]
        soc_pct = df[COL_SOC] * 100.0

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hours, y=soc_pct,
            name="SoC (%)",
            fill="tozeroy",
            line=dict(color=COLOR_BATT),
            mode="lines",
            yaxis="y",
        ))

        if "battery_temp_k" in df.columns:
            fig.add_trace(go.Scatter(
                x=hours, y=df["battery_temp_k"],
                name="Temp (K)",
                line=dict(color=COLOR_GRID, width=1.5),
                mode="lines",
                yaxis="y2",
            ))

        # SoC reference lines at 10% and 95%
        fig.add_shape(
            type="line", x0=0, x1=23, y0=10, y1=10,
            line=dict(color=COLOR_GRID, dash="dash", width=1),
            yref="y",
        )
        fig.add_shape(
            type="line", x0=0, x1=23, y0=95, y1=95,
            line=dict(color=COLOR_SAVINGS, dash="dash", width=1),
            yref="y",
        )

        fig.update_layout(
            title="Battery State of Charge & Temperature",
            xaxis_title="Hour of Day",
            yaxis=dict(title="State of Charge (%)", range=[0, 105]),
            yaxis2=dict(title="Temperature (K)", overlaying="y", side="right"),
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def build_cumulative_economics(self, df: pd.DataFrame) -> go.Figure:
        """Build cumulative cost, revenue, and net cost traces.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Plotly Figure with three economic traces.
        """
        hours = df[COL_HOUR]
        cum_cost = df[COL_CUMCOST]
        cum_rev = df[COL_CUMREV]
        net = cum_cost - cum_rev
        net_color = COLOR_SAVINGS if float(net.iloc[-1]) <= 0.0 else COLOR_GRID

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hours, y=cum_cost,
            name="Cumulative Cost ($)",
            line=dict(color=COLOR_GRID, width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=cum_rev,
            name="Cumulative Revenue ($)",
            line=dict(color=COLOR_SAVINGS, width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=net,
            name="Net Cost ($)",
            fill="tozeroy",
            line=dict(color=net_color, width=2),
            mode="lines",
        ))

        fig.update_layout(
            title="Cumulative Economics",
            xaxis_title="Hour of Day",
            yaxis_title="Amount ($)",
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def build_reward_trace(self, df: pd.DataFrame) -> go.Figure:
        """Build per-step reward bar chart with smoothed rolling mean overlay.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Plotly Figure with reward bars and rolling mean line.
        """
        hours = df[COL_HOUR]
        rewards = df[COL_REWARD]
        colors = [COLOR_SAVINGS if r >= 0 else COLOR_GRID for r in rewards]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=hours, y=rewards,
            name="Step Reward",
            marker_color=colors,
        ))

        rolling_mean = rewards.rolling(window=4, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=hours, y=rolling_mean,
            name="Rolling Mean (4h)",
            line=dict(color=COLOR_REWARD, width=2),
            mode="lines",
        ))

        fig.update_layout(
            title="Per-Step Reward Signal",
            xaxis_title="Hour of Day",
            yaxis_title="Reward",
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def build_action_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Build action distribution as a horizontal stacked bar (percentage).

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Plotly Figure showing fraction of steps per action.
        """
        action_colors = [COLOR_PV, COLOR_SAVINGS, COLOR_BATT, COLOR_GRID]
        total = max(len(df), 1)
        fig = go.Figure()

        for action_id, label in _ACTION_LABELS.items():
            count = int((df[COL_ACTION].astype(int) == action_id).sum())
            pct = count / total * 100.0
            fig.add_trace(go.Bar(
                name=label,
                x=[pct],
                y=["Actions"],
                orientation="h",
                marker_color=action_colors[action_id],
                text=[f"{label}: {pct:.1f}%"],
                textposition="inside",
            ))

        fig.update_layout(
            title="Action Distribution",
            xaxis_title="Percentage of Steps (%)",
            yaxis_title="",
            barmode="stack",
            template=CHART_TEMPLATE,
            height=max(CHART_HEIGHT // 2, 200),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def build_grid_price_overlay(self, df: pd.DataFrame) -> go.Figure:
        """Build dual overlay of grid price and grid import with TOU bands.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Plotly Figure with price step-line and import filled area.
        """
        hours = df[COL_HOUR]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hours, y=df[COL_GRID_IMPORT],
            name="Grid Import (kW)",
            fill="tozeroy",
            line=dict(color=COLOR_BATT, width=1),
            mode="lines",
            opacity=0.5,
            yaxis="y2",
        ))

        fig.add_trace(go.Scatter(
            x=hours, y=df[COL_GRID_PRICE],
            name="Grid Price ($/kWh)",
            line=dict(color=COLOR_GRID, width=2, shape="hv"),
            mode="lines",
            yaxis="y",
        ))

        # TOU background bands
        # Off-peak: 0-9, 21-23  Mid-peak shoulder: implicit  On-peak: 9-21
        fig.add_vrect(x0=0, x1=9,
                      fillcolor="rgba(39,174,96,0.06)", layer="below", line_width=0,
                      annotation_text="Off-Peak", annotation_position="top left")
        fig.add_vrect(x0=9, x1=21,
                      fillcolor="rgba(231,76,60,0.06)", layer="below", line_width=0,
                      annotation_text="On-Peak", annotation_position="top left")
        fig.add_vrect(x0=21, x1=23,
                      fillcolor="rgba(39,174,96,0.06)", layer="below", line_width=0)

        fig.update_layout(
            title="Grid Price vs. Import Behavior",
            xaxis_title="Hour of Day",
            yaxis=dict(title="Price ($/kWh)"),
            yaxis2=dict(title="Import (kW)", overlaying="y", side="right"),
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def build_pv_iv_curve(self, G: float, T: float) -> go.Figure:
        """Build I-V and P-V curves by instantiating PVArray directly.

        This is the only ChartBuilder method that imports from physics_engine.py.
        It operates on physics parameters, not simulation state, so the CSV-only
        constraint does not apply here.

        Args:
            G: Solar irradiance in W/m².
            T: Cell temperature in Kelvin.

        Returns:
            Plotly Figure with I-V curve on primary axis and P-V on secondary.
        """
        from physics_engine import PVArray  # intentional local import — see docstring
        pv = PVArray()

        v_points = 200
        mpp = pv.compute_mpp(G, T, v_points=v_points)
        V_oc = mpp["V_oc"] if mpp["V_oc"] > 0 else 6.5

        voltages = np.linspace(0.0, V_oc * 0.999, v_points)
        currents = np.array([pv.solve_iv_point(v, G, T) for v in voltages])
        powers_kw = voltages * currents / 1000.0

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=voltages, y=currents,
            name="Current (A)",
            line=dict(color=COLOR_PV, width=2),
            mode="lines",
            yaxis="y",
        ))

        fig.add_trace(go.Scatter(
            x=voltages, y=powers_kw,
            name="Power (kW)",
            line=dict(color=COLOR_BATT, width=2, dash="dash"),
            mode="lines",
            yaxis="y2",
        ))

        # MPP annotation
        if mpp["V_mp"] > 0:
            fig.add_trace(go.Scatter(
                x=[mpp["V_mp"]],
                y=[mpp["I_mp"]],
                name="MPP",
                mode="markers",
                marker=dict(symbol="star", size=14, color=COLOR_GRID),
                yaxis="y",
            ))
            fig.add_annotation(
                x=mpp["V_mp"],
                y=mpp["I_mp"],
                text=f"MPP: {mpp['P_mp']/1000:.2f} kW",
                showarrow=True,
                arrowhead=2,
                yref="y",
            )

        fig.update_layout(
            title=f"PV I-V & P-V Curve (G={G:.0f} W/m², T={T:.1f} K)",
            xaxis_title="Voltage (V)",
            yaxis=dict(title="Current (A)"),
            yaxis2=dict(title="Power (kW)", overlaying="y", side="right"),
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig


class ScorecardBuilder:
    """Computes summary metrics and builds a scorecard figure. Stateless."""

    def __init__(self) -> None:
        """Initialize ScorecardBuilder. Stateless."""

    def compute_metrics(self, df: pd.DataFrame) -> dict[str, float | str]:
        """Derive 13 summary metrics from a preprocessed DataFrame.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Dict with keys: total_pv_kwh, total_load_kwh, total_grid_import_kwh,
            total_grid_export_kwh, grid_dependence_pct, pv_self_consumption,
            total_cost_usd, total_revenue_usd, net_cost_usd, final_soc_pct,
            final_soh_pct, dominant_action, mean_reward.
        """
        total_pv_kwh = float(df[COL_PV].sum())
        total_load_kwh = float(df[COL_LOAD].sum())
        total_grid_import_kwh = float(df[COL_GRID_IMPORT].sum())
        total_grid_export_kwh = float(df[COL_GRID_EXPORT].sum())

        grid_dependence_pct = float(
            np.clip(total_grid_import_kwh / max(total_load_kwh, 1e-9) * 100.0, 0.0, 100.0)
        )
        pv_self_consumption = float(
            np.clip(
                (total_pv_kwh - total_grid_export_kwh) / max(total_pv_kwh, 1e-9) * 100.0,
                0.0, 100.0,
            )
        )

        if "cost_grid_import" in df.columns:
            total_cost_usd = float(df["cost_grid_import"].sum())
        else:
            total_cost_usd = float((df[COL_GRID_IMPORT] * df[COL_GRID_PRICE]).sum())

        if "revenue_grid_export" in df.columns:
            total_revenue_usd = float(df["revenue_grid_export"].sum())
        else:
            total_revenue_usd = float(df[COL_GRID_EXPORT].sum() * 0.07)

        net_cost_usd = total_cost_usd - total_revenue_usd
        final_soc_pct = float(df[COL_SOC].iloc[-1]) * 100.0
        final_soh_pct = float(df[COL_SOH].iloc[-1]) * 100.0
        mean_reward = float(df[COL_REWARD].mean())

        if "action_label" in df.columns:
            dominant_action = str(df["action_label"].mode().iloc[0])
        else:
            dominant_action = _ACTION_LABELS.get(int(df[COL_ACTION].mode().iloc[0]), "Unknown")

        return {
            "total_pv_kwh": total_pv_kwh,
            "total_load_kwh": total_load_kwh,
            "total_grid_import_kwh": total_grid_import_kwh,
            "total_grid_export_kwh": total_grid_export_kwh,
            "grid_dependence_pct": grid_dependence_pct,
            "pv_self_consumption": pv_self_consumption,
            "total_cost_usd": total_cost_usd,
            "total_revenue_usd": total_revenue_usd,
            "net_cost_usd": net_cost_usd,
            "final_soc_pct": final_soc_pct,
            "final_soh_pct": final_soh_pct,
            "dominant_action": dominant_action,
            "mean_reward": mean_reward,
        }

    def build_scorecard_figure(
        self, metrics: dict[str, float | str]
    ) -> go.Figure:
        """Build a Plotly Table figure rendering scorecard metrics.

        Args:
            metrics: Dict returned by compute_metrics().

        Returns:
            Plotly Figure with a formatted metrics table.
        """
        labels = [
            "Total PV (kWh)", "Total Load (kWh)", "Grid Import (kWh)", "Grid Export (kWh)",
            "Grid Dependence (%)", "PV Self-Consumption (%)",
            "Total Cost ($)", "Total Revenue ($)", "Net Cost ($)",
            "Final SoC (%)", "Final SoH (%)", "Dominant Action", "Mean Reward",
        ]
        keys = [
            "total_pv_kwh", "total_load_kwh", "total_grid_import_kwh", "total_grid_export_kwh",
            "grid_dependence_pct", "pv_self_consumption",
            "total_cost_usd", "total_revenue_usd", "net_cost_usd",
            "final_soc_pct", "final_soh_pct", "dominant_action", "mean_reward",
        ]

        values: list[str] = []
        for k in keys:
            v = metrics.get(k, "N/A")
            if isinstance(v, float):
                values.append(f"{v:.3f}")
            else:
                values.append(str(v))

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color=COLOR_BATT,
                font=dict(color="white", size=13),
                align="left",
            ),
            cells=dict(
                values=[labels, values],
                fill_color=[["white", "#f0f4ff"] * 7],
                font=dict(size=12),
                align="left",
            ),
        )])

        fig.update_layout(
            title="Episode Scorecard",
            template=CHART_TEMPLATE,
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig


class DashboardApp:
    """Streamlit application assembling all charts and the scorecard."""

    def __init__(self, log_path: str | None = None) -> None:
        """Initialize dashboard application.

        Args:
            log_path: Path to a simulation CSV log. If None, shows file uploader.
        """
        self._log_path = log_path
        self._chart_builder = ChartBuilder()
        self._scorecard_builder = ScorecardBuilder()

    def run(self) -> None:
        """Launch the Streamlit dashboard. Entry point for streamlit run."""
        import streamlit as st  # local import — only used at runtime

        st.set_page_config(
            page_title="Solar Microgrid Dashboard",
            layout="wide",
            page_icon="☀️",
        )

        self._render_header()

        # Resolve log path
        path = self._log_path
        if path is None:
            uploaded = st.file_uploader("Upload simulation CSV log", type=["csv"])
            if uploaded is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as tmp:
                    tmp.write(uploaded.read())
                    path = tmp.name

        if path is None:
            st.info("Upload a simulation log CSV to begin.")
            return

        df = self._load_and_validate(path)
        if df is None:
            return

        visibility = self._render_sidebar(df)
        self._render_scorecard(df)
        self._render_charts(df, visibility)
        self._render_raw_data(df)

    def _render_header(self) -> None:
        """Render dashboard title, subtitle, and horizontal rule."""
        import streamlit as st

        st.title("Solar Microgrid Power Optimizer")
        st.markdown("**Real-time telemetry dashboard** — reads from simulation CSV log.")
        if self._log_path:
            st.caption(f"Log file: `{self._log_path}`")
        st.markdown("---")

    def _render_sidebar(self, df: pd.DataFrame) -> dict[str, bool]:
        """Render sidebar controls and return visibility flags.

        Args:
            df: Preprocessed simulation DataFrame.

        Returns:
            Dict mapping chart name to bool visibility flag.
        """
        import streamlit as st

        st.sidebar.header("Dashboard Controls")

        chart_toggles = {
            "generation_vs_load":   st.sidebar.checkbox("Generation vs. Load", value=True),
            "battery_soc":          st.sidebar.checkbox("Battery SoC & Temp", value=True),
            "cumulative_economics": st.sidebar.checkbox("Cumulative Economics", value=True),
            "reward_trace":         st.sidebar.checkbox("Reward Signal", value=True),
            "action_distribution":  st.sidebar.checkbox("Action Distribution", value=True),
            "grid_price_overlay":   st.sidebar.checkbox("Grid Price Overlay", value=True),
            "pv_iv_curve":          st.sidebar.checkbox("PV I-V Curve", value=False),
        }

        st.sidebar.markdown("---")
        st.sidebar.subheader("PV I-V Curve Parameters")
        st.sidebar.number_input("Irradiance (W/m²)", min_value=0.0, max_value=1200.0,
                                 value=800.0, step=50.0, key="G_iv")
        st.sidebar.number_input("Temperature (°C)", min_value=-20.0, max_value=80.0,
                                 value=25.0, step=1.0, key="T_iv_c")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Log Selection")
        log_dir = "./logs"
        if os.path.isdir(log_dir):
            csvs = sorted(
                [f for f in os.listdir(log_dir) if f.endswith(".csv")],
                reverse=True,
            )
            if csvs:
                st.sidebar.selectbox("Select log file", csvs, key="selected_log")

        n_steps = len(df)
        st.sidebar.markdown(f"**Steps:** {n_steps} | **dt:** 1.0 h")

        return chart_toggles

    def _render_charts(
        self, df: pd.DataFrame, visibility: dict[str, bool]
    ) -> None:
        """Render visible charts in a 2-column grid layout.

        Args:
            df: Preprocessed simulation DataFrame.
            visibility: Dict of chart name → bool.
        """
        import streamlit as st

        col1, col2 = st.columns(2)
        G_iv = st.session_state.get("G_iv", 800.0)
        T_iv_c = st.session_state.get("T_iv_c", 25.0)
        T_iv_k = float(T_iv_c) + 273.15

        charts: list[tuple[str, Any]] = [
            ("generation_vs_load",   lambda: self._chart_builder.build_generation_vs_load(df)),
            ("battery_soc",          lambda: self._chart_builder.build_battery_soc(df)),
            ("cumulative_economics", lambda: self._chart_builder.build_cumulative_economics(df)),
            ("reward_trace",         lambda: self._chart_builder.build_reward_trace(df)),
            ("action_distribution",  lambda: self._chart_builder.build_action_distribution(df)),
            ("grid_price_overlay",   lambda: self._chart_builder.build_grid_price_overlay(df)),
            ("pv_iv_curve",          lambda: self._chart_builder.build_pv_iv_curve(G_iv, T_iv_k)),
        ]

        cols = [col1, col2]
        idx = 0
        for name, builder in charts:
            if visibility.get(name, False):
                with cols[idx % 2]:
                    fig = builder()
                    st.plotly_chart(fig, use_container_width=True)
                idx += 1

    def _render_scorecard(self, df: pd.DataFrame) -> None:
        """Render key metrics using st.metric widgets and scorecard table.

        Args:
            df: Preprocessed simulation DataFrame.
        """
        import streamlit as st

        metrics = self._scorecard_builder.compute_metrics(df)

        st.subheader("Episode Scorecard")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Net Cost ($)", f"{metrics['net_cost_usd']:.3f}")
        m2.metric("Grid Dependence (%)", f"{metrics['grid_dependence_pct']:.1f}")
        m3.metric("PV Self-Consumption (%)", f"{metrics['pv_self_consumption']:.1f}")
        m4.metric("Final SoC (%)", f"{metrics['final_soc_pct']:.1f}")
        m5.metric("Mean Reward", f"{metrics['mean_reward']:.4f}")

        fig = self._scorecard_builder.build_scorecard_figure(metrics)
        st.plotly_chart(fig, use_container_width=True)

    def _render_raw_data(self, df: pd.DataFrame) -> None:
        """Render collapsible raw data table.

        Args:
            df: Preprocessed simulation DataFrame.
        """
        import streamlit as st

        with st.expander("Raw Simulation Data"):
            st.dataframe(df)

    def _load_and_validate(self, path: str) -> pd.DataFrame | None:
        """Load, validate, and preprocess a CSV log. Returns None on error.

        Args:
            path: Path to simulation CSV.

        Returns:
            Preprocessed DataFrame, or None if loading/validation failed.
        """
        import streamlit as st

        try:
            loader = LogLoader(path)
            df_raw = loader.load()
            loader.validate(df_raw)
            df = loader.preprocess(df_raw)
            return df
        except Exception as exc:
            st.error(f"Failed to load log: {exc}")
            log.error("Dashboard load error: %s", exc)
            return None


def run_dashboard(log_path: str | None = None) -> None:
    """Launch the Streamlit dashboard application.

    Args:
        log_path: Optional path to a simulation CSV log. If None, the app
                  shows a file uploader widget.
    """
    app = DashboardApp(log_path=log_path)
    app.run()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Auto-detect latest log if no path given
    if path is None:
        log_dir = "./logs"
        if os.path.isdir(log_dir):
            csvs = sorted(
                [f for f in os.listdir(log_dir) if f.endswith(".csv")],
                reverse=True
            )
            if csvs:
                path = os.path.join(log_dir, csvs[0])
    
    run_dashboard(path)
