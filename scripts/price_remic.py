import argparse
import json
import numpy as np
import pandas as pd

from src.quantfi.remic import RemicParams, RemicPricer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=str, default="results/hw_fit.json")
    parser.add_argument("--theta", type=str, default="results/theta_t.json")
    parser.add_argument("--out", type=str, default="results/remic_summary.json")
    args = parser.parse_args()

    with open(args.fit, "r", encoding="utf-8") as f:
        fit = json.load(f)
    kappa = float(fit["kappa"])
    sigma = float(fit["sigma"])

    with open(args.theta, "r", encoding="utf-8") as f:
        th = json.load(f)
    theta_t = np.asarray(th["theta_t"], dtype=float)

    params = RemicParams(
        seed=0,
        n_paths_total=10000,
        t_months=360,
        basis=12.0,
        kappa=kappa,
        sigma=sigma,
        r0=0.05,
        theta=theta_t,
        orig_bal=94894021.0,
        wac=0.0747,
        serv=0.0097,
        wam=354,
        beta=0.38089,
        psa_mult=2.5,
        psa_step=0.002,
        max_ramp=30,
        fa_spread=0.0092,
        sa_cap=0.0558,
    )

    pricer = RemicPricer(params)
    r_paths, disc_paths = pricer.simulate_hw_paths_antithetic()
    fa_pvs, sa_pvs = pricer.price_fa_sa(r_paths, disc_paths)
    fa_pv, fa_px, fa_se = pricer.summarize_pv(fa_pvs, params.orig_bal)
    sa_pv, sa_px, sa_se = pricer.summarize_pv(sa_pvs, params.orig_bal)

    oas_fa = pricer.solve_oas_to_par(r_paths, disc_paths, params.orig_bal, "FA")
    oas_sa = pricer.solve_oas_to_par(r_paths, disc_paths, params.orig_bal, "SA")

    out = {
        "fa": {"pv": fa_pv, "px_100": fa_px, "se_100": fa_se, "oas": oas_fa},
        "sa": {"pv": sa_pv, "px_100": sa_px, "se_100": sa_se, "oas": oas_sa},
        "paths": len(fa_pvs)
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
