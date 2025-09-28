import numpy as np
from src.quantfi.remic import RemicParams, RemicPricer

def test_mc_shapes_and_prices():
    p = RemicParams(
        seed=1, n_paths_total=1000, t_months=24, basis=12.0,
        kappa=0.1, sigma=0.01, r0=0.03, theta=None,
        orig_bal=1_000_000.0, wac=0.06, serv=0.01, wam=24,
        beta=0.2, psa_mult=2.5, psa_step=0.002, max_ramp=30,
        fa_spread=0.01, sa_cap=0.05
    )
    pricer = RemicPricer(p)
    r, d = pricer.simulate_hw_paths_antithetic()
    fa, sa = pricer.price_fa_sa(r, d)
    assert fa.shape == sa.shape == (p.n_paths_total,)
    pv, px, se = pricer.summarize_pv(fa, p.orig_bal)
    assert px > 0 and se >= 0
