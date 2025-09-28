from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar


def adjust_to_us_business_day(date_like: pd.Timestamp | dt.date) -> pd.Timestamp:
    date = pd.Timestamp(date_like).normalize()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=date - pd.Timedelta(days=7), end=date + pd.Timedelta(days=14)).to_pydatetime()
    while date.weekday() >= 5 or any(date.date() == h.date() for h in holidays):
        date += pd.Timedelta(days=1)
    return date


def days_actual(start: pd.Timestamp | dt.date, end: pd.Timestamp | dt.date) -> int:
    return int((pd.Timestamp(end) - pd.Timestamp(start)).days)


def get_payment_date(accrual_end_date: pd.Timestamp | dt.date) -> pd.Timestamp:
    pay = pd.Timestamp(accrual_end_date)
    for _ in range(2):
        pay = adjust_to_us_business_day(pay + pd.Timedelta(days=1))
    return pay


@dataclass
class MarketCurves:
    accrual_df: pd.DataFrame
    payment_df: pd.DataFrame

    @classmethod
    def from_csvs(cls, accrual_csv: str, payment_csv: str) -> "MarketCurves":
        accrual = pd.read_csv(accrual_csv, names=["accrual_dfs"])
        payment = pd.read_csv(payment_csv, names=["payment_dfs"])
        return cls(accrual_df=accrual, payment_df=payment)

    def discount_by_payment_date(self, pay_date: pd.Timestamp) -> float:
        row = self.payment_df[self.payment_df["payment_date"] == pay_date]
        if row.empty:
            raise KeyError(f"payment date {pay_date} not found")
        return float(row["payment_dfs"].iloc[0])

    def discount_by_accrual_date(self, accr_date: pd.Timestamp) -> float:
        row = self.accrual_df[self.accrual_df["accrual_date"] == accr_date]
        if row.empty:
            raise KeyError(f"accrual date {accr_date} not found")
        return float(row["accrual_dfs"].iloc[0])


def build_quarterly_schedule(settlement_date: dt.date, n_quarters: int = 120) -> Tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    accrual_dates = [adjust_to_us_business_day(settlement_date)]
    payment_dates = []
    for i in range(1, n_quarters + 1):
        accrual_dt = adjust_to_us_business_day(settlement_date + relativedelta(months=3 * i))
        accrual_dates.append(accrual_dt)
        payment_dates.append(get_payment_date(accrual_dt))
    return accrual_dates, payment_dates


def attach_schedule_and_forwards(curves: MarketCurves, settlement_date: dt.date) -> MarketCurves:
    accrual_dates, payment_dates = build_quarterly_schedule(settlement_date)
    curves.accrual_df["accrual_date"] = accrual_dates
    curves.payment_df["payment_date"] = payment_dates

    n = len(payment_dates)
    curves.payment_df["accrual_period_start_date"] = [None] * n
    curves.payment_df["accrual_period_end_date"] = [None] * n
    curves.payment_df["tau"] = np.nan
    curves.payment_df["F_sofr"] = np.nan
    curves.payment_df["T"] = np.nan
    curves.payment_df["T_onethird"] = np.nan

    for i in range(n):
        start = accrual_dates[i]
        end = accrual_dates[i + 1]
        curves.payment_df.at[i, "accrual_period_start_date"] = start
        curves.payment_df.at[i, "accrual_period_end_date"] = end
        tau = days_actual(start, end) / 360.0
        curves.payment_df.at[i, "tau"] = tau

        df_start = lookup_accrual_df(curves, start)
        df_end = lookup_accrual_df(curves, end)
        fwd = (df_start / df_end - 1.0) / tau
        curves.payment_df.at[i, "F_sofr"] = fwd

        T = days_actual(accrual_dates[0], end) / 365.0
        curves.payment_df.at[i, "T"] = T
        if i == 0:
            curves.payment_df.at[i, "T_onethird"] = T / 3.0
        else:
            prev_T = curves.payment_df.at[i - 1, "T"]
            curves.payment_df.at[i, "T_onethird"] = prev_T + (T - prev_T) / 3.0

    return curves


def lookup_accrual_df(curves: MarketCurves, date_like: pd.Timestamp | dt.date) -> float:
    row = curves.accrual_df[curves.accrual_df["accrual_date"] == pd.Timestamp(date_like)]
    if row.empty:
        raise KeyError("accrual date not found")
    return float(row["accrual_dfs"].iloc[0])


def lookup_payment_df(curves: MarketCurves, date_like: pd.Timestamp | dt.date) -> float:
    row = curves.payment_df[curves.payment_df["payment_date"] == pd.Timestamp(date_like)]
    if row.empty:
        raise KeyError("payment date not found")
    return float(row["payment_dfs"].iloc[0])
