You can place your own CSV files here. The code expects single column CSVs with the following names.

## Here are there required columns
1  accrual_dfs for accrual discount factors
2  payment_dfs for payment discount factors
3  BBG_swap_rates for par swap rates used as cap strikes
4  bachelier_vols for normal vols in basis points
5  BBG_sofr_caps for present values in dollars for a ten million notional cap

All files should correspond to one market date. The code builds a quarterly schedule out to ten years and aligns maturities with the rows present in the swap rates file.
