"""Black-Scholes pricing functions for European call and put options."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute d1 parameter for Black-Scholes model."""
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    """Compute d2 parameter for Black-Scholes model."""
    return d1 - sigma * np.sqrt(T)


def bs_price(option_type: str, S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Price a European option using Black-Scholes formula.

    Parameters
    ----------
    option_type
        'call' or 'put'.
    S
        Spot price.
    K
        Strike price.
    r
        Continuously compounded risk-free rate.
    sigma
        Volatility (annualized, decimal).
    T
        Time to maturity in years.
    """
    option_type = option_type.lower().strip()

    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute Black-Scholes vega for Newton-based IV solver."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, r, sigma, T)
    return S * norm.pdf(d1) * np.sqrt(T)
