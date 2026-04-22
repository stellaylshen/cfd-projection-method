# Core functions for staggered_grid.py
import numpy as np

# =========================================================
# 1. Grid setup (geometry layer)
# =========================================================
# [CORE]
def setup_mac_grid(Nx, Ny, Lx=1.0, Ly=1.0):
    dx = Lx / Nx
    dy = Ly / Ny

    # variables
    p = np.zeros((Nx, Ny))          # cell center
    u = np.zeros((Nx + 1, Ny))      # vertical faces
    v = np.zeros((Nx, Ny + 1))      # horizontal faces

    # coordinates (for debug / visualization)
    x_p = (np.arange(Nx) + 0.5) * dx
    y_p = (np.arange(Ny) + 0.5) * dy
    Xp, Yp = np.meshgrid(x_p, y_p, indexing="ij")

    x_u = np.arange(Nx + 1) * dx
    y_u = (np.arange(Ny) + 0.5) * dy
    Xu, Yu = np.meshgrid(x_u, y_u, indexing="ij")

    x_v = (np.arange(Nx) + 0.5) * dx
    y_v = np.arange(Ny + 1) * dy
    Xv, Yv = np.meshgrid(x_v, y_v, indexing="ij")

    return dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv


# =========================================================
# 2. Operators
# =========================================================
# [CORE]
def compute_divergence_mac(u, v, dx, dy):
    div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy
    return div

# [CORE]
def compute_pressure_gradient_mac(p, dx, dy):
    Nx, Ny = p.shape

    dpdx_u = np.zeros((Nx + 1, Ny))
    dpdy_v = np.zeros((Nx, Ny + 1))

    # interior u-faces
    dpdx_u[1:Nx, :] = (p[1:, :] - p[:-1, :]) / dx

    # interior v-faces
    dpdy_v[:, 1:Ny] = (p[:, 1:] - p[:, :-1]) / dy

    return dpdx_u, dpdy_v

# [CORE]
def compute_laplacian_center(p, dx, dy):
    Nx, Ny = p.shape
    lap = np.zeros_like(p)

    lap[1:-1, 1:-1] = (
        (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx**2
        +
        (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dy**2
    )

    return lap

# [CORE]
def solve_poisson_jacobi_center_simple(rhs, dx, dy, max_iter=20000, tol=1e-10, verbose = True):
    """
    Solve Laplacian(p) = rhs on cell centers using Jacobi iteration.

    Boundary condition for now:
        p = 0 on boundary cells
    """
    Nx, Ny = rhs.shape
    p = np.zeros((Nx, Ny))
    pn = np.zeros_like(p)

    coef = 2.0 * (dx**2 + dy**2)

    for it in range(max_iter):
        pn[:] = p[:]

        p[1:-1, 1:-1] = (
            (dy**2) * (pn[2:, 1:-1] + pn[:-2, 1:-1]) +
            (dx**2) * (pn[1:-1, 2:] + pn[1:-1, :-2]) -
            (dx**2) * (dy**2) * rhs[1:-1, 1:-1]
        ) / coef

        err = np.max(np.abs(p - pn))
        if err < tol:
            if verbose: # 輸出更多細節（詳細模式）
                print(f"Poisson converged at iteration {it}, error = {err:.3e}")
            break
    else:
        print(f"Poisson NOT converged, final error = {err:.3e}")

    return p

# [CORE]
def solve_poisson_jacobi_center_neumann(rhs, u_star, v_star, dx, dy, dt, max_iter=20000, tol=1e-10, verbose = True):
    Nx, Ny = rhs.shape
    p = np.zeros((Nx, Ny))
    pn = np.zeros_like(p)

    coef = 2.0 * (dx**2 + dy**2)

    for it in range(max_iter):
        pn[:] = p[:]   # <- 這行你現在少了

        # 用 predictor 更新 boundary，但更新到 pn 比較乾淨
        pn = apply_pressure_bc_neumann_from_predictor(pn, u_star, v_star, dx, dy, dt)

        p[1:-1, 1:-1] = (
            (dy**2) * (pn[2:, 1:-1] + pn[:-2, 1:-1]) +
            (dx**2) * (pn[1:-1, 2:] + pn[1:-1, :-2]) -
            (dx**2) * (dy**2) * rhs[1:-1, 1:-1]
        ) / coef

        # 邊界再同步一次
        p = apply_pressure_bc_neumann_from_predictor(p, u_star, v_star, dx, dy, dt)

        # 固定 pressure gauge
        p[0, 0] = 0.0

        err = np.max(np.abs(p - pn))
        if err < tol:
            if verbose: # verbose = 冗長(？)
                print(f"Poisson converged at iteration {it}, error = {err:.3e}")
            break
    else:
        print(f"Poisson NOT converged, final error = {err:.3e}")

    return p

# [CORE]
def project_velocity_mac(u_star, v_star, p, dx, dy, dt):
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)

    u_new = u_star.copy()
    v_new = v_star.copy()

    # correct interior faces only
    u_new[1:-1, :] = u_star[1:-1, :] - dt * dpdx_u[1:-1, :]
    v_new[:, 1:-1] = v_star[:, 1:-1] - dt * dpdy_v[:, 1:-1]

    return u_new, v_new


# [CORE]
def apply_velocity_bc_mac(u, v, U_lid=1.0):
    """
    BC interpretation:
    - Top lid velocity is NOT stored directly in the state array.
    - It is treated as an external boundary condition.
    - Its effect should enter the domain via diffusion stencil (ghost-like treatment).

    This version is consistent with:
    - diffusion-based predictor (compute_diffusion_predictor_mac)
    - "momentum gradually diffuses into interior" physics

    Resulting behavior:
    - Top-adjacent interior row can develop velocity gradually
    - No artificial jump at top boundary
    """    
    # u: left/right walls (normal velocity)
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # bottom wall: stationary no-slip-like nearest row
    u[:, 0] = 0.0

    # IMPORTANT:
    # do NOT force u[:, -1] = U_lid anymore
    # top lid influence will enter through diffusion ghost-like treatment

    # v: no penetration on bottom/top
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    # minimal side treatment
    v[0, :] = 0.0
    v[-1, :] = 0.0

    return u, v

# [CORE]
def apply_velocity_bc_mac_ns(u, v, U_lid=1.0):
    """
    BC interpretation:
    - Top lid velocity IS directly stored in the state array.
    - The top row of u is treated as the moving wall itself.

    This version is consistent with:
    - simple lid-driven cavity setups
    - "immediate boundary forcing" (no gradual diffusion needed to create lid motion)

    Resulting behavior:
    - Top row is always fixed to U_lid
    - Creates a sharp velocity jump at the top corners:
        [0 → U_lid] on left
        [U_lid → 0] on right
    - Divergence appears mainly at top corners if interior is still zero

    Important implication:
    - Assumes the discretization treats top row as boundary state,
      NOT as interior influenced by ghost values
    """
    # u
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = U_lid   # <- 直接把 lid 放回 state

    # keep side-wall corners zero
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # v
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0

    return u, v

# [CORE]
def apply_pressure_bc_neumann_from_predictor(p, u_star, v_star, dx, dy, dt):
    """
    Neumann BC for pressure based on predictor velocity (u_star, v_star)

    p: (Nx, Ny)
    u_star: (Nx+1, Ny)
    v_star: (Nx, Ny+1)
    """

    Nx, Ny = p.shape

    # -------------------------
    # Left wall (x = 0)
    # dp/dx = u_star / dt
    # -------------------------
    for j in range(Ny):
        g = u_star[0, j] / dt
        p[0, j] = p[1, j] - g * dx

    # -------------------------
    # Right wall (x = L)
    # -------------------------
    for j in range(Ny):
        g = u_star[-1, j] / dt
        p[-1, j] = p[-2, j] + g * dx

    # -------------------------
    # Bottom wall (y = 0)
    # -------------------------
    for i in range(Nx):
        g = v_star[i, 0] / dt
        p[i, 0] = p[i, 1] - g * dy

    # -------------------------
    # Top wall (y = L)
    # -------------------------
    for i in range(Nx):
        g = v_star[i, -1] / dt
        p[i, -1] = p[i, -2] + g * dy

    return p

# [CORE]
def compute_diffusion_predictor_mac(u, v, dx, dy, dt, nu, U_lid=1.0):
    u_star = u.copy()
    v_star = v.copy()

    # -------------------------------------------------
    # u diffusion
    # u.shape = (Nx+1, Ny)
    # j = 0 ........ Ny-1
    # bottom row j=0 and top row j=Ny-1 will be overwritten
    # with special wall-adjacent formulas
    # -------------------------------------------------

    if u.shape[1] > 2:
        lap_u_bulk = (
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            +
            (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
        u_star[1:-1, 1:-1] += nu * dt * lap_u_bulk

    # top-adjacent row (last u row, adjacent to moving lid)
    j_top = u.shape[1] - 1
    if j_top >= 1:
        d2u_dx2_top = (u[2:, j_top] - 2.0 * u[1:-1, j_top] + u[:-2, j_top]) / dx**2
        d2u_dy2_top = (u[1:-1, j_top - 1] - 3.0 * u[1:-1, j_top] + 2.0 * U_lid) / dy**2
        lap_u_top = d2u_dx2_top + d2u_dy2_top
        u_star[1:-1, j_top] = u[1:-1, j_top] + nu * dt * lap_u_top

    # bottom-adjacent row
    j_bot = 0
    d2u_dx2_bot = (u[2:, j_bot] - 2.0 * u[1:-1, j_bot] + u[:-2, j_bot]) / dx**2
    d2u_dy2_bot = (u[1:-1, j_bot + 1] - 3.0 * u[1:-1, j_bot] + 0.0) / dy**2
    lap_u_bot = d2u_dx2_bot + d2u_dy2_bot
    u_star[1:-1, j_bot] = u[1:-1, j_bot] + nu * dt * lap_u_bot

    # keep left/right walls zero
    u_star[0, :] = 0.0
    u_star[-1, :] = 0.0

    # -------------------------------------------------
    # v diffusion
    # -------------------------------------------------
    lap_v = (
        (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
        +
        (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    )
    v_star[1:-1, 1:-1] += nu * dt * lap_v

    # keep v boundary conditions
    v_star[:, 0] = 0.0
    v_star[:, -1] = 0.0
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0

    return u_star, v_star

# [CORE]
def step_diffusion_projection_mac(u, v, dx, dy, dt, nu, U_lid=1.0):
    # enforce BC on current field
    u, v = apply_velocity_bc_mac(u.copy(), v.copy(), U_lid=U_lid)

    # diffusion predictor
    u_star, v_star = compute_diffusion_predictor_mac(
        u, v, dx, dy, dt, nu, U_lid=U_lid
    )

    # enforce BC again after predictor
    u_star, v_star = apply_velocity_bc_mac(u_star, v_star, U_lid=U_lid)

    # divergence of predictor
    div_star = compute_divergence_mac(u_star, v_star, dx, dy)

    # Poisson solve
    rhs = div_star / dt
    p = solve_poisson_jacobi_center_neumann(
    rhs, u_star, v_star, dx, dy, dt,
    max_iter=20000, tol=1e-10
    )
    # projection
    u_new, v_new = project_velocity_mac(u_star, v_star, p, dx, dy, dt)

    # enforce BC again after projection
    u_new, v_new = apply_velocity_bc_mac(u_new, v_new, U_lid=U_lid)

    div_new = compute_divergence_mac(u_new, v_new, dx, dy)

    return {
        "u": u,
        "v": v,
        "u_star": u_star,
        "v_star": v_star,
        "p": p,
        "u_new": u_new,
        "v_new": v_new,
        "div_star": div_star,
        "div_new": div_new,
    }

# [CORE] （之後想細分升級成 [RUNNER]）
def run_diffusion_projection_mac(Nx, Ny, dx, dy, nsteps=20, dt=1e-3, nu=0.1, U_lid=1.0):
    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))

    history = []

    for step in range(nsteps):
        results = step_diffusion_projection_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

        u = results["u_new"]
        v = results["v_new"]

        history.append({
            "step": step + 1,
            "u": u.copy(),
            "v": v.copy(),
            "u_star": results["u_star"].copy(),
            "v_star": results["v_star"].copy(),
            "p": results["p"].copy(),
            "div_star": results["div_star"].copy(),
            "div_new": results["div_new"].copy(),
        })

    return history

# Advection 區域
# [CORE]
def compute_advection_mac(u, v, dx, dy):
    """
    Minimal central-difference advection terms on MAC grid.

    Returns
    -------
    adv_u : same shape as u
    adv_v : same shape as v
    """
    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    # ---- interpolate v to u-locations ----
    # shape: (Nx-1, Ny)
    v_on_u = 0.25 * (
        v[:-1, :-1] + v[1:, :-1] +
        v[:-1, 1:]  + v[1:, 1:]
    )

    # ---- interpolate u to v-locations ----
    # shape: (Nx, Ny-1)
    u_on_v = 0.25 * (
        u[:-1, :-1] + u[1:, :-1] +
        u[:-1, 1:]  + u[1:, 1:]
    )

    # -------------------------
    # u-equation advection
    # adv_u = u du/dx + v du/dy
    # only interior in both x and y
    # -------------------------
    du_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)      # shape (Nx-1, Ny-2)
    du_dy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)      # shape (Nx-1, Ny-2)

    adv_u[1:-1, 1:-1] = (
        u[1:-1, 1:-1] * du_dx +
        v_on_u[:, 1:-1] * du_dy
    )

    # -------------------------
    # v-equation advection
    # adv_v = u dv/dx + v dv/dy
    # only interior in both x and y
    # -------------------------
    dv_dx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)      # shape (Nx-2, Ny-1)
    dv_dy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)      # shape (Nx-2, Ny-1)

    adv_v[1:-1, 1:-1] = (
        u_on_v[1:-1, :] * dv_dx +
        v[1:-1, 1:-1] * dv_dy
    )

    return adv_u, adv_v

# [CORE]
def compute_ns_predictor_mac(u, v, dx, dy, dt, nu, U_lid=1.0):
    """
    Minimal Navier-Stokes predictor:
        u* = u + dt * ( - advection + nu * laplacian )
    """
    # 先做 diffusion 那部分
    u_star, v_star = compute_diffusion_predictor_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

    # 再扣掉 advection
    adv_u, adv_v = compute_advection_mac(u, v, dx, dy)

    u_star[1:-1, 1:-1] -= dt * adv_u[1:-1, 1:-1]
    v_star[1:-1, 1:-1] -= dt * adv_v[1:-1, 1:-1]

    return u_star, v_star

# [CORE]
def step_ns_projection_mac(u, v, dx, dy, dt, nu, U_lid=1.0):
    # enforce BC on current state
    u, v = apply_velocity_bc_mac(u.copy(), v.copy(), U_lid=U_lid)

    # predictor: advection + diffusion
    u_star, v_star = compute_ns_predictor_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

    # enforce BC again
    u_star, v_star = apply_velocity_bc_mac(u_star, v_star, U_lid=U_lid)

    # divergence of predictor
    div_star = compute_divergence_mac(u_star, v_star, dx, dy)

    # projection (use simple solver for now)
    rhs = div_star / dt

    # 不要放這印 每step就狂噴 改成存入dict和其他東西一起印
    rhs_max_overall = np.max(np.abs(rhs))
    rhs_max_interior = np.max(np.abs(rhs[1:-1, 1:-1]))
    #print("max abs rhs overall =", np.max(np.abs(rhs)))
    #print("max abs rhs interior =", np.max(np.abs(rhs[1:-1, 1:-1])))

    p = solve_poisson_jacobi_center_simple(
        rhs, dx, dy,
        max_iter=20000, tol=1e-10,
        verbose=False
    )

    u_new, v_new = project_velocity_mac(u_star, v_star, p, dx, dy, dt)
    u_new, v_new = apply_velocity_bc_mac(u_new, v_new, U_lid=U_lid)

    div_new = compute_divergence_mac(u_new, v_new, dx, dy)

    return {
        "u_star": u_star,
        "v_star": v_star,
        "p": p,
        "u_new": u_new,
        "v_new": v_new,
        "div_star": div_star,
        "div_new": div_new,
        "rhs_max_overall": rhs_max_overall,
        "rhs_max_interior": rhs_max_interior,
    }

# [CORE]
def run_ns_projection_mac(Nx, Ny, dx, dy, nsteps=50, dt=1e-3, nu=0.1, U_lid=1.0):
    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))

    history = []

    for step in range(nsteps):
        results = step_ns_projection_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

        u = results["u_new"]
        v = results["v_new"]

        history.append({
            "step": step + 1,
            "u": u.copy(),
            "v": v.copy(),
            "u_star": results["u_star"].copy(),
            "v_star": results["v_star"].copy(),
            "p": results["p"].copy(),
            "div_star": results["div_star"].copy(),
            "div_new": results["div_new"].copy(),
            "rhs_max_overall": results["rhs_max_overall"],  #這兩個 scalar 不需要 .copy()，因為是數字。
            "rhs_max_interior": results["rhs_max_interior"],
        })

        if (step + 1) % 20 == 0:
            print(
                f"step {step+1:3d} | "
                f"max|rhs| overall = {results['rhs_max_overall']:.3e} | "
                f"max|rhs| interior = {results['rhs_max_interior']:.3e} | "
                f"max|div_star| = {np.max(np.abs(results['div_star'])):.3e} | "
                f"max|div_new| = {np.max(np.abs(results['div_new'])):.3e}"
            )

    return history
