"""Implied volatility inversion utilities with robust solver fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from black_scholes import bs_price, bs_vega


@dataclass
class IVSolverConfig:
    """Configuration for implied volatility solving."""

    sigma_lower: float = 1e-6
    sigma_upper: float = 5.0
    tol: float = 1e-8
    max_iter: int = 100
    initial_guess: float = 0.25


def _intrinsic_value(option_type: str, S: float, K: float) -> float:
    return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)


def implied_volatility(
    option_type: str,
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    config: IVSolverConfig | None = None,
) -> float:
    """Compute implied volatility from market price using Newton with Brent fallback.

    The function handles typical edge cases: near-expiry maturities, very low premiums,
    and noisy quotes that may violate no-arbitrage bounds.
    """
    cfg = config or IVSolverConfig()
    option_type = option_type.lower().strip()

    if market_price <= 0 or S <= 0 or K <= 0:
        return np.nan

    if T <= 1e-8:
        return np.nan

    intrinsic = _intrinsic_value(option_type, S, K)
    if market_price < intrinsic - 1e-10:
        return np.nan

    max_price = S if option_type == "call" else K * np.exp(-r * T)
    if market_price > max_price + 1e-10:
        return np.nan

    # 1) Newton iterations (fast near solution)
    sigma = max(cfg.initial_guess, cfg.sigma_lower)
    for _ in range(cfg.max_iter):
        model = bs_price(option_type, S, K, r, sigma, T)
        diff = model - market_price
        if abs(diff) < cfg.tol:
            return float(np.clip(sigma, cfg.sigma_lower, cfg.sigma_upper))

        vega = bs_vega(S, K, r, sigma, T)
        if vega < 1e-10:
            break

        sigma_next = sigma - diff / vega
        if not np.isfinite(sigma_next) or sigma_next <= 0:
            break
        sigma = float(np.clip(sigma_next, cfg.sigma_lower, cfg.sigma_upper))

    # 2) Brent fallback (robust global root finder)
    def objective(vol: float) -> float:
        return bs_price(option_type, S, K, r, vol, T) - market_price

    try:
        f_low = objective(cfg.sigma_lower)
        f_high = objective(cfg.sigma_upper)
        if f_low == 0:
            return cfg.sigma_lower
        if f_high == 0:
            return cfg.sigma_upper
        if f_low * f_high > 0:
            return np.nan

        root = brentq(objective, cfg.sigma_lower, cfg.sigma_upper, xtol=cfg.tol, maxiter=cfg.max_iter)
        return float(root)
    except Exception:
        return np.nan
