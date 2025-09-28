import numpy as np

def test_theta_monotone_inputs_example():
    monthly_grid = np.arange(0.0, 5.0, 1.0 / 12.0)
    inst_fwd = 0.03 + 0.00 * monthly_grid
    dfwd_dt = np.gradient(inst_fwd, monthly_grid)
    assert monthly_grid.shape == inst_fwd.shape == dfwd_dt.shape
