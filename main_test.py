# main_test.py 
# 專門跑最小驗證 確認數學與離散算子沒壞
# 5x5 toy grid

import numpy as np
from core import (
    setup_mac_grid_ghost, 
    apply_velocity_bc_mac_ghost,
    compute_divergence_mac_ghost,
    compute_pressure_gradient_mac_ghost,
    project_velocity_mac_ghost,
    face_to_center_velocity_ghost,
    compute_diffusion_predictor_mac_ghost,
    solve_poisson_sor_mac,
    run_ns_projection_mac_ghost,
)
from test import (
    test_divergence,
    test_gradient,
    test_div_grad_closure,
    test_projection_mac,
)

if __name__ == "__main__":
    # =================================================
    # Layer 1: basic MAC-grid / operator sanity checks
    # =================================================
    Nx, Ny = 5, 5

    dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid_ghost(Nx, Ny)

    # Apply lid-driven cavity velocity BCs
    u, v = apply_velocity_bc_mac_ghost(u, v, U_lid=1.0)

    # -------------------------------------------------
    # Check raw array shapes on staggered MAC grid
    # -------------------------------------------------
    print("=== Shapes ===")
    print("p:", p.shape)
    print("u:", u.shape)
    print("v:", v.shape)

    # -------------------------------------------------
    # Unit tests for discrete differential operators
    # -------------------------------------------------

    # test_divergence(Nx, Ny, dx, dy)
    # test_gradient(Nx, Ny, dx, dy)
    # test_div_grad_closure(Nx, Ny, dx, dy)
    # test_projection_mac(Nx, Ny, dx, dy, dt=1.0)

    # -------------------------------------------------
    # Verify physical-domain slicing vs ghost layers
    # -------------------------------------------------
    print("physical u shape =", u[:, 1:-1].shape)
    print("physical v shape =", v[1:-1, :].shape)

    # -------------------------------------------------
    # Inspect ghost-cell BC values directly
    # -------------------------------------------------
    print("u bottom ghost sample:", u[2, 0])
    print("u top ghost sample:", u[2, -1])
    print("v left ghost sample:", v[0, 2])
    print("v right ghost sample:", v[-1, 2])

    # -------------------------------------------------
    # Divergence check for BC-enforced velocity field
    # -------------------------------------------------
    div = compute_divergence_mac_ghost(u, v, dx, dy)

    print("div shape =", div.shape)
    print("max |div| =", abs(div).max())

    # -------------------------------------------------
    # Constant-pressure gradient consistency test
    # Expect: zero gradient everywhere
    # -------------------------------------------------
    p[:, :] = 1.0
    dpdx_u, dpdy_v = compute_pressure_gradient_mac_ghost(p, dx, dy)

    print("dpdx_u shape =", dpdx_u.shape)
    print("dpdy_v shape =", dpdy_v.shape)
    print("max |dpdx| for constant p =", abs(dpdx_u).max())
    print("max |dpdy| for constant p =", abs(dpdy_v).max())

    # -------------------------------------------------
    # Projection test with constant pressure field
    # Expect: velocity remains essentially unchanged
    # -------------------------------------------------
    u_proj, v_proj = project_velocity_mac_ghost(u, v, p, dx, dy, dt=1e-3)

    div_proj = compute_divergence_mac_ghost(u_proj, v_proj, dx, dy)

    print("u_proj shape =", u_proj.shape)
    print("v_proj shape =", v_proj.shape)
    print("max |div_proj| constant p =", abs(div_proj).max())

    # Check whether ghost BC structure survives projection
    print("u top ghost after projection:", u_proj[2, -1])
    print("v left ghost after projection:", v_proj[0, 2])

    # -------------------------------------------------
    # Face-to-center interpolation sanity check
    # -------------------------------------------------
    u_c, v_c = face_to_center_velocity_ghost(u, v)

    print("u_c shape =", u_c.shape)
    print("v_c shape =", v_c.shape)
    print("max |u_c| =", abs(u_c).max())
    print("max |v_c| =", abs(v_c).max())

    # -------------------------------------------------
    # Diffusion-only predictor step test
    # -------------------------------------------------
    u_star, v_star = compute_diffusion_predictor_mac_ghost(
        u, v, dx, dy, dt=5e-3, nu=0.01, U_lid=1.0
    )

    print("u_star shape =", u_star.shape)
    print("v_star shape =", v_star.shape)
    print("max |u_star physical| =", abs(u_star[:, 1:-1]).max())
    print("max |v_star physical| =", abs(v_star[1:-1, :]).max())

    # Inspect lid momentum diffusion into top interior layer
    print("u_star top physical sample =", u_star[2, -2])
    print("u_star top ghost sample =", u_star[2, -1])


    # =================================================
    # Layer 2: full projection consistency check
    # =================================================

    div_star = compute_divergence_mac_ghost(u_star, v_star, dx, dy)

    rhs = div_star / 5e-3

    # Enforce solvability condition for Neumann Poisson
    rhs = rhs - np.mean(rhs)

    print("max |div_star| =", abs(div_star).max())

    # -------------------------------------------------
    # Solve pressure Poisson correction equation
    # -------------------------------------------------
    p_corr = solve_poisson_sor_mac(
        rhs,
        dx,
        dy,
        omega=1.7,
        max_iter=1000,
        tol=1e-6,
        verbose=False,
    )

    # -------------------------------------------------
    # Apply pressure projection correction
    # -------------------------------------------------
    u_corr, v_corr = project_velocity_mac_ghost(
        u_star,
        v_star,
        p_corr,
        dx,
        dy,
        dt=5e-3,
    )

    # -------------------------------------------------
    # IMPORTANT:
    # re-apply ghost BC after projection
    # -------------------------------------------------
    u_corr, v_corr = apply_velocity_bc_mac_ghost(
        u_corr,
        v_corr,
        U_lid=1.0,
    )

    div_corr = compute_divergence_mac_ghost(u_corr, v_corr, dx, dy)

    print("max |div_corr| =", abs(div_corr).max())

    # -------------------------------------------------
    # Final centered-velocity consistency check
    # -------------------------------------------------
    u_c_corr, v_c_corr = face_to_center_velocity_ghost(u_corr, v_corr)

    print("max |u_c_corr| =", abs(u_c_corr).max())
    print("max |v_c_corr| =", abs(v_c_corr).max())

    # -------------------------------------------------
    # Diffusion-Only somke test
    # -------------------------------------------------
    print("\n=== Ghost runner smoke test ===")

    history = run_ns_projection_mac_ghost(
        Nx=5, Ny=5,
        dx=dx, dy=dy,
        nsteps=20,
        dt=5e-3,
        nu=0.01,
        U_lid=1.0,
        print_every=1,
    )

    final = history[-1]

    print("final max |div_new| =", abs(final["div_new"]).max())
    print("final umax =", final["umax"])
    print("final vmax =", final["vmax"])

