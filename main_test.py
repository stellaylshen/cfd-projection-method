# main_test.py 
# 專門跑最小驗證 確認數學與離散算子沒壞
# 7x7 toy grid

import numpy as np
from core import setup_mac_grid
from test import (
    test_divergence,
    test_gradient,
    test_div_grad_closure,
    test_projection_mac,
)

if __name__ == "__main__":
    # -------------------------------------------------
    # Layer 1: mathematical / discrete operator checks
    # -------------------------------------------------
    Nx, Ny = 7, 7

    dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid(Nx, Ny)

    print("=== Shapes ===")
    print("p:", p.shape)
    print("u:", u.shape)
    print("v:", v.shape)

    test_divergence(Nx, Ny, dx, dy)
    test_gradient(Nx, Ny, dx, dy)
    test_div_grad_closure(Nx, Ny, dx, dy)
    test_projection_mac(Nx, Ny, dx, dy, dt=1.0)

