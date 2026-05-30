"""
Core numerical routines for a 2D incompressible Navier-Stokes solver
using a finite-difference projection method on a staggered MAC grid.

The main solver path uses the ghost-cell MAC formulation, where velocity
boundary conditions are imposed through ghost values and incompressibility
is enforced by solving a pressure Poisson equation followed by projection.

Some non-ghost routines are retained as legacy/experimental versions used
during development and operator-consistency testing.
"""

import numpy as np

# ============================================================
# Main ghost-cell MAC projection solver
# ============================================================
def setup_mac_grid_ghost(Nx, Ny, Lx=1.0, Ly=1.0):
    dx = Lx / Nx
    dy = Ly / Ny

    # variables
    p = np.zeros((Nx, Ny))              # cell center
    u = np.zeros((Nx + 1, Ny + 2))      # vertical u faces + bottom/top ghost
    v = np.zeros((Nx + 2 , Ny + 1))      # horizontal v faces + left/right ghost

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

def face_to_center_velocity_ghost(u, v):
    """
    Convert ghost-cell MAC velocities to cell-center velocities.
    u shape: (Nx+1, Ny+2)
        physical u: u[:, 1:-1]
    v shape: (Nx+2, Ny+1)
        physical v: v[1:-1, :]
    """

    u_phys = u[:, 1:-1]
    v_phys = v[1:-1, :]

    u_c = 0.5 * (u_phys[:-1, :] + u_phys[1:, :])
    v_c = 0.5 * (v_phys[:, :-1] + v_phys[:, 1:])

    return u_c, v_c

def compute_divergence_mac_ghost(u, v, dx, dy):
    """
    Compute divergence on pressure cells for ghost-cell MAC layout.
    u shape: (Nx+1, Ny+2)
        physical u faces: u[:, 1:-1]
    v shape: (Nx+2, Ny+1)
        physical v faces: v[1:-1, :]
    output div shape: (Nx, Ny)
    """

    div = (
        (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
        +
        (v[1:-1, 1:] - v[1:-1, :-1]) / dy
    )
    return div

def compute_pressure_gradient_mac_ghost(p, dx, dy):
    """
    Pressure gradient for ghost-cell MAC layout.
    p shape: (Nx, Ny)
    returns:
        dpdx_u shape: (Nx+1, Ny+2)
        dpdy_v shape: (Nx+2, Ny+1)

    Only physical interior faces are filled.
    Ghost/boundary-adjacent unused entries stay zero.
    """
    Nx, Ny = p.shape

    dpdx_u = np.zeros((Nx + 1, Ny + 2))
    dpdy_v = np.zeros((Nx + 2, Ny + 1))

    # u physical faces, excluding left/right boundary faces
    # p difference across vertical faces
    dpdx_u[1:Nx, 1:-1] = (p[1:, :] - p[:-1, :]) / dx

    # v physical faces, excluding bottom/top boundary faces
    # p difference across horizontal faces
    dpdy_v[1:-1, 1:Ny] = (p[:, 1:] - p[:, :-1]) / dy

    return dpdx_u, dpdy_v

def compute_laplacian_center(p, dx, dy):
    Nx, Ny = p.shape
    lap = np.zeros_like(p)

    lap[1:-1, 1:-1] = (
        (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx**2
        +
        (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dy**2
    )

    return lap

def solve_poisson_sor_mac(
    rhs, dx, dy, omega=1.7, max_iter=1000, tol=1e-6, verbose=False, return_residual_history=False
):
    Nx, Ny = rhs.shape

    rhs = rhs - np.mean(rhs)

    p = np.zeros((Nx, Ny))

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2

    residual_history = []

    for it in range(max_iter):
        p_old = p.copy()

        for i in range(Nx):
            for j in range(Ny):

                neighbor_sum = 0.0
                diag = 0.0

                if i > 0:
                    neighbor_sum += p[i - 1, j] * inv_dx2
                    diag += inv_dx2
                if i < Nx - 1:
                    neighbor_sum += p[i + 1, j] * inv_dx2
                    diag += inv_dx2
                if j > 0:
                    neighbor_sum += p[i, j - 1] * inv_dy2
                    diag += inv_dy2
                if j < Ny - 1:
                    neighbor_sum += p[i, j + 1] * inv_dy2
                    diag += inv_dy2

                p_new = (neighbor_sum - rhs[i, j]) / diag

                # SOR update
                p[i, j] = p[i, j] + omega * (p_new - p[i, j])

        p -= np.mean(p)

        err = np.max(np.abs(p - p_old))
        residual_history.append(err)

        if err < tol:
            if verbose:
                print(f"SOR converged at iter {it}, err={err:.3e}")
            break
            
    if return_residual_history:
        return p, residual_history
    
    return p

def project_velocity_mac_ghost(u_star, v_star, p, dx, dy, dt):
    """
    Projection step for ghost-cell MAC layout.

    u_star shape: (Nx+1, Ny+2)
    v_star shape: (Nx+2, Ny+1)
    p shape:      (Nx, Ny)

    Projection only corrects physical interior faces.
    Ghost cells and true boundary-normal faces are not modified.
    """

    dpdx_u, dpdy_v = compute_pressure_gradient_mac_ghost(p, dx, dy)

    u_new = u_star.copy()
    v_new = v_star.copy()

    # u physical interior faces only:
    # exclude left/right boundary faces and bottom/top ghost
    u_new[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * dpdx_u[1:-1, 1:-1]

    # v physical interior faces only:
    # exclude left/right ghost and bottom/top boundary faces
    v_new[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * dpdy_v[1:-1, 1:-1]

    return u_new, v_new

def apply_velocity_bc_mac_ghost(u, v, U_lid=1.0):
    """
    Apply ghost-cell velocity boundary conditions on a MAC grid.

    u shape: (Nx+1, Ny+2), with physical values in u[:, 1:-1].
    v shape: (Nx+2, Ny+1), with physical values in v[1:-1, :].

    Wall-normal velocities are set to zero. Tangential no-slip and
    moving-lid conditions are imposed through ghost-cell reflection.
    """

    # u velocity: left/right walls enforce no penetration.
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # Bottom wall: (u_ghost + u_first) / 2 = 0.
    u[:, 0] = -u[:, 1]

    # Top lid: (u_top_physical + u_top_ghost) / 2 = U_lid.
    u[:, -1] = 2.0 * U_lid - u[:, -2]

    # Side-wall conditions take precedence at corners.
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # v velocity: bottom/top walls enforce v = 0 (no penetration).
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    # Left/right walls: enforce tangential no-slip via ghost reflection.
    v[0, :] = -v[1, :]
    v[-1, :] = -v[-2, :]

    # Bottom/top wall conditions take precedence at corners.
    v[:, 0] = 0.0
    v[:, -1] = 0.0

    return u, v

def compute_diffusion_predictor_mac_ghost(u, v, dx, dy, dt, nu, U_lid=1.0):
    u = u.copy()
    v = v.copy()

    u, v = apply_velocity_bc_mac_ghost(u, v, U_lid=U_lid)

    u_star = u.copy()
    v_star = v.copy()

    # u physical interior faces:
    # x interior faces: 1:-1
    # y physical rows: 1:-1
    lap_u = (
        (u[2:, 1:-1] - 2.0*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        +
        (u[1:-1, 2:] - 2.0*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )

    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + nu * dt * lap_u

    # v physical interior faces:
    # x physical rows: 1:-1
    # y interior faces: 1:-1
    lap_v = (
        (v[2:, 1:-1] - 2.0*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
        +
        (v[1:-1, 2:] - 2.0*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    )

    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + nu * dt * lap_v

    u_star, v_star = apply_velocity_bc_mac_ghost(u_star, v_star, U_lid=U_lid)

    return u_star, v_star

def compute_ns_predictor_mac_ghost(u, v, dx, dy, dt, nu, U_lid=1.0):
    """
    Full NS predictor for ghost-cell MAC grid.

    Includes:
    - advection
    - diffusion

    Returns:
        u_star, v_star
    """

    u = u.copy()
    v = v.copy()

    u, v = apply_velocity_bc_mac_ghost(u, v, U_lid=U_lid)

    u_star = u.copy()
    v_star = v.copy()

    # =================================================
    # u equation
    # =================================================
    u_c = 0.5 * (u[1:-1, 1:-1] + u[2:, 1:-1])

    v_on_u = 0.25 * (
        v[1:-2, :-1] + v[2:-1, :-1]
        + v[1:-2, 1:] + v[2:-1, 1:]
    )

    du_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
    du_dy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)

    adv_u = u_c * du_dx + v_on_u * du_dy

    lap_u = (
        (u[2:, 1:-1] - 2.0*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        +
        (u[1:-1, 2:] - 2.0*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )

    u_star[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        - dt * adv_u
        + nu * dt * lap_u
    )

    # =================================================
    # v equation
    # =================================================
    v_c = 0.5 * (v[1:-1, 1:-1] + v[1:-1, 2:])

    u_on_v = 0.25 * (
        u[:-1, 1:-2] + u[:-1, 2:-1]
        + u[1:, 1:-2] + u[1:, 2:-1]
    )

    dv_dx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
    dv_dy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)

    adv_v = u_on_v * dv_dx + v_c * dv_dy

    lap_v = (
        (v[2:, 1:-1] - 2.0*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
        +
        (v[1:-1, 2:] - 2.0*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    )

    v_star[1:-1, 1:-1] = (
        v[1:-1, 1:-1]
        - dt * adv_v
        + nu * dt * lap_v
    )

    u_star, v_star = apply_velocity_bc_mac_ghost(
        u_star, v_star, U_lid=U_lid
    )

    return u_star, v_star

def step_ns_projection_mac_ghost(u, v, dx, dy, dt, nu, U_lid=1.0):
    # 1. Apply ghost-cell BC to the current state.
    u, v = apply_velocity_bc_mac_ghost(u.copy(), v.copy(), U_lid=U_lid)

    # 2. Compute the predictor velocity field.
    u_star, v_star = compute_ns_predictor_mac_ghost(
        u, v, dx, dy, dt, nu, U_lid=U_lid
    )

    # 3. Compute predictor divergence.
    div_star = compute_divergence_mac_ghost(u_star, v_star, dx, dy)

    # 4. Construct the pressure Poisson RHS.
    rhs = div_star / dt
    rhs = rhs - np.mean(rhs)

    rhs_max_overall = np.max(np.abs(rhs))
    rhs_max_interior = np.max(np.abs(rhs[1:-1, 1:-1]))

    # 5. Solve the pressure Poisson equation.
    p, poisson_residual_history  = solve_poisson_sor_mac(
        rhs, dx, dy,
        omega=1.7,
        max_iter=1000,
        tol=1e-6,
        verbose=False,
        return_residual_history=True,
    )

    # 6. Projection step.
    u_proj, v_proj = project_velocity_mac_ghost(u_star, v_star, p, dx, dy, dt)

    # 7. Re-apply ghost-cell BC after projection.
    u_new, v_new = apply_velocity_bc_mac_ghost(u_proj, v_proj, U_lid=U_lid)

    # 8. Diagnostics.
    div_proj = compute_divergence_mac_ghost(u_proj, v_proj, dx, dy)
    div_new = compute_divergence_mac_ghost(u_new, v_new, dx, dy)

    umax = np.max(np.abs(u_new[:, 1:-1]))
    vmax = np.max(np.abs(v_new[1:-1, :]))
    cfl = umax * dt / dx + vmax * dt / dy

    return {
        "u_star": u_star,
        "v_star": v_star,
        "p": p,
        "u_new": u_new,
        "v_new": v_new,
        "div_star": div_star,
        "div_proj": div_proj,
        "div_new": div_new,
        "rhs_max_overall": rhs_max_overall,
        "rhs_max_interior": rhs_max_interior,
        "cfl": cfl,
        "umax": umax,
        "vmax": vmax,
        "poisson_residual_history": poisson_residual_history,
        "poisson_iterations": len(poisson_residual_history),
    }

def run_ns_projection_mac_ghost(
    Nx, Ny, dx, dy,
    nsteps=50,
    dt=1e-3,
    nu=0.1,
    U_lid=1.0,
    steady_tol=None,
    min_steps=50,
    print_every=50,
):
    u = np.zeros((Nx + 1, Ny + 2))
    v = np.zeros((Nx + 2, Ny + 1))

    u, v = apply_velocity_bc_mac_ghost(u, v, U_lid=U_lid)

    history = []

    for step in range(nsteps):
        u_old = u.copy()
        v_old = v.copy()

        results = step_ns_projection_mac_ghost(
            u, v, dx, dy, dt, nu, U_lid=U_lid
        )

        u = results["u_new"]
        v = results["v_new"]

        velocity_change = max(
            np.max(np.abs(u[:, 1:-1] - u_old[:, 1:-1])),
            np.max(np.abs(v[1:-1, :] - v_old[1:-1, :])),
        )

        history.append({
            "step": step + 1,
            "u": u.copy(),
            "v": v.copy(),
            "u_star": results["u_star"].copy(),
            "v_star": results["v_star"].copy(),
            "p": results["p"].copy(),
            "div_star": results["div_star"].copy(),
            "div_proj": results["div_proj"].copy(),
            "div_new": results["div_new"].copy(),
            "velocity_change": velocity_change,
            "rhs_max_overall": results["rhs_max_overall"],
            "rhs_max_interior": results["rhs_max_interior"],
            "cfl": results["cfl"],
            "umax": results["umax"],
            "vmax": results["vmax"],
            "poisson_residual_history": results["poisson_residual_history"],
            "poisson_iterations": results["poisson_iterations"],
        })

        if steady_tol is not None and (step + 1) >= min_steps:
            if velocity_change < steady_tol:
                print(
                    f"Converged at step {step+1}, "
                    f"vel_change = {velocity_change:.3e}"
                )
                break

        if (step + 1) == 1 or (step + 1) % print_every == 0 or (step + 1) == nsteps:
            print(
                f"step {step+1:4d} | "
                f"CFL = {results['cfl']:.3e} | "
                f"vel_change = {velocity_change:.3e} | "
                f"max|rhs| overall = {results['rhs_max_overall']:.3e} | "
                f"max|rhs| interior = {results['rhs_max_interior']:.3e} | "
                f"max|div_star| = {np.max(np.abs(results['div_star'])):.3e} | "
                f"max|div_proj| = {np.max(np.abs(results['div_proj'])):.3e} | "
                f"max|div_new| = {np.max(np.abs(results['div_new'])):.3e}"
            )

    if len(history) == nsteps:
        print("WARNING: reached max nsteps before steady convergence")
    
    return history


# ============================================================
# Legacy non-ghost routines
# ============================================================
# These routines are retained for development history and
# boundary-condition comparison tests. They are not used by the
# final cavity benchmark solver.

def setup_mac_grid(Nx, Ny, Lx=1.0, Ly=1.0):
    dx = Lx / Nx
    dy = Ly / Ny

    # variables
    p = np.zeros((Nx, Ny))          # cell center
    u = np.zeros((Nx + 1, Ny))      # vertical u faces 
    v = np.zeros((Nx, Ny + 1))      # horizontal v faces 

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

def get_cell_center_coordinates(Nx, Ny, Lx=1.0, Ly=1.0):
    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    Xp, Yp = np.meshgrid(x, y, indexing="ij")
    return Xp, Yp

def face_to_center_velocity(u, v):
    u_c = 0.5 * (u[:-1, :] + u[1:, :])
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])
    return u_c, v_c

def compute_divergence_mac(u, v, dx, dy):
    div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy
    return div

def compute_pressure_gradient_mac(p, dx, dy):
    Nx, Ny = p.shape

    dpdx_u = np.zeros((Nx + 1, Ny))
    dpdy_v = np.zeros((Nx, Ny + 1))

    # interior u-faces
    dpdx_u[1:Nx, :] = (p[1:, :] - p[:-1, :]) / dx

    # interior v-faces
    dpdy_v[:, 1:Ny] = (p[:, 1:] - p[:, :-1]) / dy

    return dpdx_u, dpdy_v

def solve_poisson_jacobi_center_simple(rhs, dx, dy, max_iter=20000, tol=1e-10, verbose=True):
    """
    Solve the cell-centered Poisson equation using Jacobi iteration
    with homogeneous Dirichlet pressure boundary conditions (p = 0 on boundary cells).
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
            if verbose: 
                print(f"Poisson converged at iteration {it}, error = {err:.3e}")
            break
    else:
        print(f"Poisson NOT converged, final error = {err:.3e}")

    return p

def solve_poisson_jacobi_center_neumann(rhs, u_star, v_star, dx, dy, dt, max_iter=20000, tol=1e-10, verbose = True):
    Nx, Ny = rhs.shape
    p = np.zeros((Nx, Ny))
    pn = np.zeros_like(p)

    coef = 2.0 * (dx**2 + dy**2)

    for it in range(max_iter):
        pn[:] = p[:]

        # Update pressure boundary values using the predictor velocity field.
        # Applying the boundary condition to pn first improves Jacobi consistency.
        pn = apply_pressure_bc_neumann_from_predictor(pn, u_star, v_star, dx, dy, dt)

        p[1:-1, 1:-1] = (
            (dy**2) * (pn[2:, 1:-1] + pn[:-2, 1:-1]) +
            (dx**2) * (pn[1:-1, 2:] + pn[1:-1, :-2]) -
            (dx**2) * (dy**2) * rhs[1:-1, 1:-1]
        ) / coef

        # Reapply pressure boundary conditions after the Jacobi update.
        p = apply_pressure_bc_neumann_from_predictor(p, u_star, v_star, dx, dy, dt)

        # Fix the pressure gauge to remove the null-space ambiguity.
        p[0, 0] = 0.0

        err = np.max(np.abs(p - pn))
        if err < tol:
            if verbose: 
                print(f"Poisson converged at iteration {it}, error = {err:.3e}")
            break
    else:
        print(f"Poisson NOT converged, final error = {err:.3e}")

    return p

def solve_poisson_jacobi_mac_consistent(rhs, dx, dy, max_iter=1000, tol=1e-6, verbose=False):
    """
    Solve L p = rhs where L = div(grad(p)) using the SAME MAC
    divergence and gradient convention as project_velocity_mac().

    Boundary meaning:
    - pressure normal gradient at domain boundary is zero
    - boundary faces are not corrected by pressure
    - pressure is fixed by removing mean(p)
    """

    Nx, Ny = rhs.shape

    # Neumann compatibility condition
    rhs = rhs - np.mean(rhs)

    p = np.zeros((Nx, Ny))
    pn = np.zeros_like(p)

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2

    for it in range(max_iter):
        pn[:] = p[:]

        # loop version first: slower, but much clearer and less bug-prone
        for i in range(Nx):
            for j in range(Ny):
                neighbor_sum = 0.0
                diag = 0.0

                # left neighbor
                if i > 0:
                    neighbor_sum += pn[i - 1, j] * inv_dx2
                    diag += inv_dx2

                # right neighbor
                if i < Nx - 1:
                    neighbor_sum += pn[i + 1, j] * inv_dx2
                    diag += inv_dx2

                # bottom neighbor
                if j > 0:
                    neighbor_sum += pn[i, j - 1] * inv_dy2
                    diag += inv_dy2

                # top neighbor
                if j < Ny - 1:
                    neighbor_sum += pn[i, j + 1] * inv_dy2
                    diag += inv_dy2

                p[i, j] = (neighbor_sum - rhs[i, j]) / diag

        # pressure gauge fixing: pressure itself is arbitrary up to a constant
        p -= np.mean(p)

        err = np.max(np.abs(p - pn))

        if err < tol:
            if verbose:
                print(f"Poisson converged at iteration {it}, error = {err:.3e}")
            break
    else:
        if verbose:
            print(f"Poisson NOT converged, final error = {err:.3e}")

    return p

def solve_poisson_jacobi_mac_consistent_vectorized(
    rhs, dx, dy, max_iter=1000, tol=1e-6, verbose=False
):
    """
    Vectorized Jacobi solver for Lp = rhs,
    where L = div(grad(p)) using the same MAC convention.

    Boundary behavior:
    - zero normal pressure gradient
    - boundary cells have fewer neighbors
    """

    Nx, Ny = rhs.shape

    rhs = rhs - np.mean(rhs)

    p = np.zeros((Nx, Ny))

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2

    for it in range(max_iter):
        pn = p.copy()

        neighbor_sum = np.zeros_like(p)
        diag = np.zeros_like(p)

        # left neighbor
        neighbor_sum[1:, :] += pn[:-1, :] * inv_dx2
        diag[1:, :] += inv_dx2

        # right neighbor
        neighbor_sum[:-1, :] += pn[1:, :] * inv_dx2
        diag[:-1, :] += inv_dx2

        # bottom neighbor
        neighbor_sum[:, 1:] += pn[:, :-1] * inv_dy2
        diag[:, 1:] += inv_dy2

        # top neighbor
        neighbor_sum[:, :-1] += pn[:, 1:] * inv_dy2
        diag[:, :-1] += inv_dy2

        p = (neighbor_sum - rhs) / diag

        # pressure gauge
        p -= np.mean(p)

        err = np.max(np.abs(p - pn))

        if err < tol:
            if verbose:
                print(f"Poisson converged at iteration {it}, error = {err:.3e}")
            break
    else:
        if verbose:
            print(f"Poisson NOT converged, final error = {err:.3e}")

    return p

def project_velocity_mac(u_star, v_star, p, dx, dy, dt):
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)

    u_new = u_star.copy()
    v_new = v_star.copy()

    # correct interior faces only
    u_new[1:-1, :] = u_star[1:-1, :] - dt * dpdx_u[1:-1, :]
    v_new[:, 1:-1] = v_star[:, 1:-1] - dt * dpdy_v[:, 1:-1]

    return u_new, v_new

def apply_velocity_bc_mac(u, v, U_lid=1.0):
    """
    Projection-compatible MAC velocity boundary conditions.
    Only wall-normal velocity components are enforced directly.
    Tangential wall effects enter through the predictor/diffusion stencil.
    """    
    # normal velocity at left/right walls
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # normal velocity at bottom/top walls
    v[:, 0] = 0.0
    v[:,-1] = 0.0

    return u, v

def apply_velocity_bc_mac_ns(u, v, U_lid=1.0):
    """
    Classical lid-driven cavity boundary conditions.

    The top lid velocity is directly imposed on the stored velocity field,
    with u = U_lid prescribed at the top boundary.

    This approach produces immediate lid motion and is consistent with
    simple cavity-flow formulations that treat boundary values as part
    of the state array.
    """
    # u
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = U_lid

    # keep side-wall corners zero
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # v
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0

    return u, v

def apply_pressure_bc_neumann_from_predictor(p, u_star, v_star, dx, dy, dt):
    """
    Apply explicit Neumann pressure BC using predictor velocities.
    Used only in early non-ghost projection experiments.
    """
    Nx, Ny = p.shape

    # Left wall (x = 0)
    # dp/dx = u_star / dt
    for j in range(Ny):
        g = u_star[0, j] / dt
        p[0, j] = p[1, j] - g * dx

    # Right wall (x = L)
    for j in range(Ny):
        g = u_star[-1, j] / dt
        p[-1, j] = p[-2, j] + g * dx

    # Bottom wall (y = 0)
    for i in range(Nx):
        g = v_star[i, 0] / dt
        p[i, 0] = p[i, 1] - g * dy

    # Top wall (y = L)
    for i in range(Nx):
        g = v_star[i, -1] / dt
        p[i, -1] = p[i, -2] + g * dy

    return p

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

def compute_ns_predictor_mac(u, v, dx, dy, dt, nu, U_lid=1.0):
    """
    Compute the Navier-Stokes predictor step:
        u* = u + dt * ( - advection + nu * laplacian )
    """
    # First apply the diffusion contribution.
    u_star, v_star = compute_diffusion_predictor_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

    # Then subtract the advection contribution.
    adv_u, adv_v = compute_advection_mac(u, v, dx, dy)

    u_star[1:-1, 1:-1] -= dt * adv_u[1:-1, 1:-1]
    v_star[1:-1, 1:-1] -= dt * adv_v[1:-1, 1:-1]

    return u_star, v_star

def step_ns_projection_mac(u, v, dx, dy, dt, nu, U_lid=1.0, bc_mode="A"):
    """
    Legacy non-ghost projection step for testing boundary-condition variants.

    bc_mode:
        "A": no velocity BC after projection
        "B": re-apply full stored velocity BC after projection
        "C": re-apply normal-only velocity BC after projection
    """
    # 1. enforce BC on the current state.
    u, v = apply_velocity_bc_mac(u.copy(), v.copy(), U_lid=U_lid)

    # 2. Predictor step: advection + diffusion.
    u_star, v_star = compute_ns_predictor_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

    # 3. Re-apply BC after the predictor step.
    u_star, v_star = apply_velocity_bc_mac(u_star, v_star, U_lid=U_lid)

    # 4. Compute divergence of the predictor field.
    div_star = compute_divergence_mac(u_star, v_star, dx, dy)

    # 5. Construct the pressure Poisson RHS.
    rhs = div_star / dt
    rhs = rhs - np.mean(rhs)

    rhs_max_overall = np.max(np.abs(rhs))
    rhs_max_interior = np.max(np.abs(rhs[1:-1, 1:-1]))

    # 6. Solve the pressure Poisson equation.
    p = solve_poisson_sor_mac(
    rhs, dx, dy, omega=1.7, max_iter=1000, tol=1e-6, verbose=False
    )

    # 7. Projection step.
    u_proj, v_proj = project_velocity_mac(u_star, v_star, p, dx, dy, dt)

    # 8. Diagnostic: divergence after projection.
    div_proj = compute_divergence_mac(u_proj, v_proj, dx, dy)

    # 9. Apply post-projection BC treatment.
    if bc_mode == "A":
        u_new, v_new = u_proj, v_proj

    elif bc_mode == "B":
        u_new, v_new = apply_velocity_bc_mac_ns(u_proj, v_proj, U_lid=U_lid)

    elif bc_mode == "C":
        u_new, v_new = apply_velocity_bc_mac(u_proj, v_proj, U_lid=U_lid)

    else:
        raise ValueError(f"Unknown bc_mode: {bc_mode}")

    div_new = compute_divergence_mac(u_new, v_new, dx, dy)

    # 10. Compute CFL condition.
    umax = np.max(np.abs(u_new))
    vmax = np.max(np.abs(v_new))
    cfl = umax * dt / dx + vmax * dt / dy

    return {
        "u_star": u_star,
        "v_star": v_star,
        "p": p,
        "u_new": u_new,
        "v_new": v_new,
        "div_star": div_star,
        "div_proj": div_proj,
        "div_new": div_new,
        "rhs_max_overall": rhs_max_overall,
        "rhs_max_interior": rhs_max_interior,
        "cfl": cfl,
        "umax": umax,
        "vmax": vmax,
    }

def run_ns_projection_mac(
    Nx, Ny, dx, dy,
    nsteps=50,
    dt=1e-3,
    nu=0.1,
    U_lid=1.0,
    steady_tol=None,
    min_steps=50,
    print_every=50,
    bc_mode="A",
):
    """
    Legacy non-ghost Navier-Stokes projection runner.

    Repeatedly advances the non-ghost projection step and records
    diagnostic fields for boundary-condition comparison tests.
    """
    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))

    history = []

    for step in range(nsteps):
        u_old = u.copy()
        
        v_old = v.copy()
        results = step_ns_projection_mac(u, v, dx, dy, dt, nu, U_lid=U_lid, bc_mode=bc_mode)

        u = results["u_new"]
        v = results["v_new"]

        velocity_change = max(
            np.max(np.abs(u - u_old)),
            np.max(np.abs(v - v_old))
        )

        if steady_tol is not None and (step + 1) >= min_steps:
            if velocity_change < steady_tol:
                print(f"Converged at step {step+1}, vel_change = {velocity_change:.3e}")
                break

        history.append({
            "step": step + 1,
            "u": u.copy(),
            "v": v.copy(),
            "u_star": results["u_star"].copy(),
            "v_star": results["v_star"].copy(),
            "p": results["p"].copy(),
            "div_star": results["div_star"].copy(),
            "div_proj": results["div_proj"].copy(),
            "div_new": results["div_new"].copy(),
            "velocity_change": velocity_change,
            "rhs_max_overall": results["rhs_max_overall"],  
            "rhs_max_interior": results["rhs_max_interior"],
            "cfl": results["cfl"],
            "umax": results["umax"],
            "vmax": results["vmax"],
        })

        if (step + 1) == 1 or (step + 1) % print_every == 0 or (step + 1) == nsteps:
            print(
                f"step {step+1:4d} | "
                f"CFL = {results['cfl']:.3e} | "
                f"vel_change = {velocity_change:.3e} | "
                f"max|rhs| overall = {results['rhs_max_overall']:.3e} | "
                f"max|rhs| interior = {results['rhs_max_interior']:.3e} | "
                f"max|div_star| = {np.max(np.abs(results['div_star'])):.3e} | "
                f"max|div_proj| = {np.max(np.abs(results['div_proj'])):.3e} | "
                f"max|div_new| = {np.max(np.abs(results['div_new'])):.3e}"
            )

    return history