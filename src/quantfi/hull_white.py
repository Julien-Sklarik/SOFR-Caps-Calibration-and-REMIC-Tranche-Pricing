from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.interpolate import interp1d

from .curves import MarketCurves


def caplet_price_hw(p_pmt_start: float, p_pmt_end: float, p_acc_start: float, p_acc_end: float, tau: float, strike: float, sigma_p: float) -> float:
    if sigma_p <= 0:
        return max(p_pmt_start - (1.0 + strike * tau) * p_pmt_end, 0.0)
    h = (1.0 / sigma_p) * np.log((p_acc_end * (1.0 + strike * tau)) / p_acc_start) + 0.5 * sigma_p
    return p_pmt_start * norm.cdf(-h + sigma_p) - (1.0 + strike * tau) * p_pmt_end * norm.cdf(-h)


def sigma_p_hw(kappa: float, sigma: float, t0: float, t1: float, t2: float) -> float:
    dt = t2 - t1
    if kappa <= 0 or sigma <= 0 or dt <= 0:
        return 0.0
    exp_kdt = np.exp(-kappa * dt)
    exp_2kt1 = np.exp(-2.0 * kappa * (t1 - t0))
    term1 = ((1.0 - exp_kdt) / kappa) ** 2 * (sigma ** 2 / (2.0 * kappa)) * (1.0 - exp_2kt1)
    term2 = (sigma ** 2) / (kappa ** 2) * (dt + 2.0 / kappa * exp_kdt - (1.0 / (2.0 * kappa)) * np.exp(-2.0 * kappa * dt) - 3.0 / (2.0 * kappa))
    val = term1 + term2
    return float(np.sqrt(max(val, 0.0)))


def cap_price_hw(params: Tuple[float, float], curves: MarketCurves, maturity_years: int, strike: float, frequency: str = "3M") -> float:
    kappa, sigma = params
    freq_months = int(frequency.replace("M", ""))
    n_periods = int(maturity_years * 12 / freq_months)
    cap_price = 0.0
    t0 = 0.0
    for i in range(n_periods):
        if i == 0:
            p_pmt_start = curves.accrual_df["accrual_dfs"].iloc[i]
            p_acc_start = curves.accrual_df["accrual_dfs"].iloc[i]
            t1 = 0.0
        else:
            p_pmt_start = curves.payment_df["payment_dfs"].iloc[i - 1]
            p_acc_start = curves.accrual_df["accrual_dfs"].iloc[i]
            t1 = curves.payment_df["T"].iloc[i - 1]
        p_pmt_end = curves.payment_df["payment_dfs"].iloc[i]
        p_acc_end = curves.accrual_df["accrual_dfs"].iloc[i + 1]
        tau = curves.payment_df["tau"].iloc[i]
        t2 = curves.payment_df["T"].iloc[i]
        sp = sigma_p_hw(kappa, sigma, t0, t1, t2)
        cap_price += caplet_price_hw(p_pmt_start, p_pmt_end, p_acc_start, p_acc_end, tau, strike, sp)
    return float(cap_price)


def _residuals(params: Tuple[float, float], swap_df: pd.DataFrame, curves: MarketCurves) -> np.ndarray:
    res = []
    for _, row in swap_df.iterrows():
        maturity = int(row["maturity"])
        strike = float(row["calculated_swap_rate"])
        model_price = cap_price_hw(params, curves, maturity, strike, "3M")
        market_price = float(row["calculated_cap_price"]) / 10_000_000.0
        res.append(model_price - market_price)
    return np.asarray(res, dtype=float)


def fit_kappa_sigma(swap_df: pd.DataFrame, curves: MarketCurves, x0=(0.01, 0.01)) -> Tuple[float, float]:
    bounds = ([0.001, 0.001], [0.5, 0.5])
    result = least_squares(_residuals, x0=x0, args=(swap_df, curves), bounds=bounds, verbose=0)
    kappa, sigma = [float(x) for x in result.x]
    return kappa, sigma


def fit_theta_t(monthly_grid: np.ndarray, fwd_rates: np.ndarray, dfwd_dt: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    term1 = dfwd_dt
    term2 = kappa * fwd_rates
    term3 = (sigma ** 2) / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * monthly_grid))
    return term1 + term2 + term3


def prepare_theta_inputs(curves: MarketCurves) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payment_years = np.insert(curves.payment_df["T"].values, 0, 0.0)
    monthly_grid = np.arange(0.0, float(payment_years.max()), 1.0 / 12.0)

    spline = interp1d(payment_years, curves.accrual_df["accrual_dfs"].values, kind="cubic", fill_value="extrapolate")
    monthly_dfs = spline(monthly_grid)
    inst_fwd = -np.gradient(np.log(monthly_dfs), monthly_grid)
    dfwd_dt = np.gradient(inst_fwd, monthly_grid)
    return monthly_grid, inst_fwd, dfwd_dt


def save_fit_json(path: str, kappa: float, sigma: float, mse: float | None = None) -> None:
    obj = dict(kappa=kappa, sigma=sigma, mse=mse)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
