"""Visualization helpers for volatility smile, term structure, and 3D surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_vol_smile(df: pd.DataFrame, output_dir: str = "outputs") -> Path:
    """Plot IV smile by expiration and save to HTML."""
    outdir = Path(output_dir)
    _ensure_dir(outdir)

    fig = px.line(
        df.dropna(subset=["implied_vol"]).sort_values("strike"),
        x="strike",
        y="implied_vol",
        color=df["expiration_date"].dt.strftime("%Y-%m-%d"),
        markers=True,
        title="Volatility Smile by Expiration",
        labels={"color": "expiration", "implied_vol": "Implied Volatility"},
    )

    # Highlight ATM region around moneyness = 1 ± 2%
    atm = df[(np.abs(df["moneyness"] - 1.0) <= 0.02) & df["implied_vol"].notna()]
    fig.add_trace(
        go.Scatter(
            x=atm["strike"],
            y=atm["implied_vol"],
            mode="markers",
            marker=dict(size=8, color="black", symbol="diamond"),
            name="ATM region",
        )
    )

    left_wing = df[(df["moneyness"] > 1.05) & df["implied_vol"].notna()]
    right_wing = df[(df["moneyness"] < 0.95) & df["implied_vol"].notna()]
    for wing, name, color in [(left_wing, "OTM Put Wing", "crimson"), (right_wing, "OTM Call Wing", "royalblue")]:
        fig.add_trace(
            go.Scatter(
                x=wing["strike"],
                y=wing["implied_vol"],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.6),
                name=name,
            )
        )

    path = outdir / "vol_smile.html"
    fig.write_html(path)
    return path


def plot_term_structure(term_df: pd.DataFrame, output_dir: str = "outputs") -> Path:
    """Plot ATM IV term structure and save to HTML."""
    outdir = Path(output_dir)
    _ensure_dir(outdir)

    fig = px.line(
        term_df,
        x="days_to_expiration",
        y="atm_iv",
        markers=True,
        title="ATM Volatility Term Structure",
        labels={"days_to_expiration": "Days to Expiration", "atm_iv": "ATM Implied Volatility"},
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.95,
        text="Upward slope: vol contango | Downward slope: vol backwardation",
        showarrow=False,
    )

    path = outdir / "term_structure.html"
    fig.write_html(path)
    return path


def plot_vol_surface(df: pd.DataFrame, output_dir: str = "outputs", use_log_moneyness: bool = True) -> Path:
    """Interpolate implied vol to grid and plot 3D volatility surface."""
    outdir = Path(output_dir)
    _ensure_dir(outdir)

    clean = df.dropna(subset=["implied_vol"]).copy()
    x = clean["time_to_expiration"].to_numpy()
    y = clean["log_moneyness"].to_numpy() if use_log_moneyness else clean["strike"].to_numpy()
    z = clean["implied_vol"].to_numpy()

    x_grid = np.linspace(x.min(), x.max(), 40)
    y_grid = np.linspace(y.min(), y.max(), 40)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = griddata((x, y), z, (X, Y), method="linear")
    if np.isnan(Z).all():
        Z = griddata((x, y), z, (X, Y), method="nearest")
    else:
        nearest = griddata((x, y), z, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), nearest, Z)

    y_label = "Log-Moneyness ln(K/S)" if use_log_moneyness else "Strike"
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Time to Maturity (Years)",
            yaxis_title=y_label,
            zaxis_title="Implied Volatility",
        ),
    )

    path = outdir / "vol_surface.html"
    fig.write_html(path)
    return path
