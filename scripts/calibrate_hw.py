import argparse
import json
import datetime as dt
import numpy as np
import pandas as pd

from src.quantfi.curves import MarketCurves, attach_schedule_and_forwards
from src.quantfi.hull_white import fit_kappa_sigma, prepare_theta_inputs, fit_theta_t, save_fit_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="data")
    parser.add_argument("--trade_date", type=str, default="2023-04-13")
    parser.add_argument("--settlement_date", type=str, default="2023-04-17")
    parser.add_argument("--out", type=str, default="results/hw_fit.json")
    args = parser.parse_args()

    accrual_csv = f"{args.base_dir}/P_caplet_accruals_{args.trade_date.replace('-', '')}.csv"
    payment_csv = f"{args.base_dir}/P_caplet_payments_{args.trade_date.replace('-', '')}.csv"
    swap_csv = f"{args.base_dir}/cap_sofr_atm_strikes_{args.trade_date.replace('-', '')}.csv"
    bachelier_csv = f"{args.base_dir}/cap_sofr_atm_bachelier_vols_{args.trade_date.replace('-', '')}.csv"
    pv_csv = f"{args.base_dir}/sofr_cap_bloomberg_pv_{args.trade_date.replace('-', '')}.csv"

    curves = MarketCurves.from_csvs(accrual_csv, payment_csv)
    curves = attach_schedule_and_forwards(curves, dt.datetime.strptime(args.settlement_date, "%Y-%m-%d").date())

    swap_df = pd.read_csv(swap_csv, names=["BBG_swap_rates"])
    vols = pd.read_csv(bachelier_csv, names=["bachelier_vols"])["bachelier_vols"].values / 10000.0
    pvs = pd.read_csv(pv_csv, names=["BBG_sofr_caps"])["BBG_sofr_caps"].values

    maturities = np.array([1,2,3,4,5,6,7,8,9,10,12,15,20,25,30], dtype=int)
    swap_df = pd.DataFrame({
        "maturity": maturities,
        "calculated_swap_rate": swap_df["BBG_swap_rates"].values,
        "bachelier_vols": vols,
        "calculated_cap_price": pvs
    })

    kappa, sigma = fit_kappa_sigma(swap_df, curves)

    monthly_grid, inst_fwd, dfwd_dt = prepare_theta_inputs(curves)
    theta_t = fit_theta_t(monthly_grid, inst_fwd, dfwd_dt, kappa, sigma)

    save_fit_json(args.out, kappa, sigma, mse=None)
    with open("results/theta_t.json", "w", encoding="utf-8") as f:
        json.dump({"monthly_grid": monthly_grid.tolist(), "theta_t": theta_t.tolist()}, f, indent=2)

    print(f"kappa {kappa:.6f}  sigma {sigma:.6f}")
    print("wrote results to", args.out)


if __name__ == "__main__":
    main()
