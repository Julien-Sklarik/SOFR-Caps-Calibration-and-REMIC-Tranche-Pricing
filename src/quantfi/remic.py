from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RemicParams:
    seed: int
    n_paths_total: int
    t_months: int
    basis: float
    kappa: float
    sigma: float
    r0: float
    theta: Optional[np.ndarray]
    orig_bal: float
    wac: float
    serv: float
    wam: int
    beta: float
    psa_mult: float
    psa_step: float
    max_ramp: int
    fa_spread: float
    sa_cap: float

    @property
    def note_net(self) -> float:
        return self.wac - self.serv


class RemicPricer:
    def __init__(self, params: RemicParams, rng: Optional[np.random.Generator] = None) -> None:
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng(params.seed)

    def simulate_hw_paths_antithetic(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.p.n_paths_total % 2 == 0, "n_paths_total must be even"
        dt = 1.0 / self.p.basis
        T = self.p.t_months
        theta = self.p.theta if self.p.theta is not None else np.full(T, self.p.kappa * self.p.r0, dtype=float)
        theta = theta[:T]
        half = self.p.n_paths_total // 2
        Z = self.rng.standard_normal(size=(half, T))
        Z = np.vstack([Z, -Z])
        r = np.empty((self.p.n_paths_total, T))
        disc = np.empty((self.p.n_paths_total, T))

        r[:, 0] = self.p.r0 + (theta[0] - self.p.kappa * self.p.r0) * dt + self.p.sigma * np.sqrt(dt) * Z[:, 0]
        disc[:, 0] = np.exp(-r[:, 0] * dt)
        for t in range(1, T):
            r[:, t] = r[:, t - 1] + (theta[t] - self.p.kappa * r[:, t - 1]) * dt + self.p.sigma * np.sqrt(dt) * Z[:, t]
            disc[:, t] = disc[:, t - 1] * np.exp(-r[:, t] * dt)
        return r, disc

    def level_payment(self, balance: float, rate_annual: float, n_months: int) -> float:
        if n_months <= 0:
            return 0.0
        rm = rate_annual / self.p.basis
        if rm == 0.0:
            return balance / n_months
        return balance * rm / (1.0 - (1.0 + rm) ** (-n_months))

    def sched_prin_and_int(self, bal: float, rate_annual: float, n_remaining: int) -> Tuple[float, float]:
        pmt = self.level_payment(bal, rate_annual, n_remaining)
        interest = bal * (rate_annual / self.p.basis)
        principal = max(pmt - interest, 0.0)
        return principal, interest

    def annual_cpr_psa_250(self, pool_month: int, wam: int) -> float:
        ramp = min(pool_month + (360 - wam), self.p.max_ramp)
        return self.p.psa_mult * (self.p.psa_step * ramp)

    @staticmethod
    def smm_from_cpr(cpr: float) -> float:
        return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)

    def effective_smm(self, pool_month: int, wam: int, short_rate_now: float, contract_rate: Optional[float] = None) -> float:
        if contract_rate is None:
            contract_rate = self.p.wac
        smm_250 = self.smm_from_cpr(self.annual_cpr_psa_250(pool_month, wam))
        ten_y = short_rate_now
        x_t = contract_rate - ten_y
        smm = smm_250 * np.exp(self.p.beta * x_t)
        return float(np.clip(smm, 0.0, 1.0))

    def price_fa_sa(self, r_paths: np.ndarray, disc_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_paths, steps = r_paths.shape
        fa_val = np.zeros(n_paths)
        sa_val = np.zeros(n_paths)
        for p in range(n_paths):
            bal = self.p.orig_bal
            n_rem = self.p.wam
            tmax = min(steps, self.p.wam)
            cf_fa = []
            cf_sa = []
            for m in range(1, tmax + 1):
                sched_prin, _ = self.sched_prin_and_int(bal, self.p.note_net, n_rem)
                smm = self.effective_smm(m, self.p.wam, r_paths[p, m - 1])
                prepay = smm * max(bal - sched_prin, 0.0)
                total_prin = sched_prin + prepay

                floater_coupon = max(r_paths[p, m - 1] + self.p.fa_spread, 0.0)
                floater_int = bal * floater_coupon / self.p.basis
                support_coupon = max(self.p.sa_cap - r_paths[p, m - 1], 0.0)
                support_int = bal * support_coupon / self.p.basis

                cf_fa.append(floater_int + total_prin * 0.5)
                cf_sa.append(support_int + total_prin * 0.5)

                bal = max(bal - total_prin, 0.0)
                n_rem = max(n_rem - 1, 0)

            idxs = np.arange(len(cf_fa))
            fa_val[p] = float(np.sum(np.array(cf_fa) * disc_paths[p, idxs]))
            sa_val[p] = float(np.sum(np.array(cf_sa) * disc_paths[p, idxs]))
        return fa_val, sa_val

    @staticmethod
    def summarize_pv(pvs: np.ndarray, notional: float) -> tuple[float, float, float]:
        mean_pv = float(np.mean(pvs))
        std_pv = float(np.std(pvs, ddof=1))
        se_100 = 100.0 * (std_pv / np.sqrt(len(pvs)) / notional)
        price_100 = 100.0 * (mean_pv / notional)
        return mean_pv, price_100, se_100

    @staticmethod
    def apply_oas(disc_paths: np.ndarray, oas_annual: float, basis: float) -> np.ndarray:
        dt = 1.0 / basis
        steps = disc_paths.shape[1]
        t_idx = np.arange(1, steps + 1, dtype=float)
        adj = np.exp(-oas_annual * dt * t_idx)[None, :]
        return disc_paths * adj

    def solve_oas_to_par(self, r_paths: np.ndarray, disc_paths: np.ndarray, target_pv: float, tranche: str, lo=-0.05, hi=0.05, tol=1e-7, iters=64) -> float:
        def price_at(oas: float) -> float:
            adj_disc = self.apply_oas(disc_paths, oas, self.p.basis)
            fa_pv, sa_pv = self.price_fa_sa(r_paths, adj_disc)
            mean_fa = float(np.mean(fa_pv))
            mean_sa = float(np.mean(sa_pv))
            return mean_fa if tranche.upper() == "FA" else mean_sa

        f_lo = price_at(lo) - target_pv
        f_hi = price_at(hi) - target_pv
        if f_lo * f_hi > 0:
            return float("nan")

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f_mid = price_at(mid) - target_pv
            if abs(f_mid) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)
