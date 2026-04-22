import numpy as np
from CORE import (
    compute_divergence_mac,
    compute_pressure_gradient_mac,
    compute_laplacian_center,
    solve_poisson_jacobi_center_simple,
    project_velocity_mac,
    step_diffusion_projection_mac,
)

# [TEST]
def test_divergence(Nx, Ny, dx, dy):
    print("\n=== Divergence Tests ===")

    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))

    # Test A: zero field
    div = compute_divergence_mac(u, v, dx, dy)
    print("\nTest A: zero field")
    print("max abs =", np.max(np.abs(div)))

    # Test B: constant u
    u[:, :] = 1.0
    v[:, :] = 0.0
    div = compute_divergence_mac(u, v, dx, dy)
    print("\nTest B: constant u")
    print("max abs =", np.max(np.abs(div)))

    # Test C: u = x
    x_u = np.arange(Nx + 1) * dx
    u = np.repeat(x_u[:, None], Ny, axis=1)
    v = np.zeros((Nx, Ny + 1))

    div = compute_divergence_mac(u, v, dx, dy)
    print("\nTest C: u = x")
    print("min =", div.min(), "max =", div.max())

# [TEST]
def test_gradient(Nx, Ny, dx, dy):
    print("\n=== Gradient Tests ===")

    # Test A: constant p
    p = np.ones((Nx, Ny))
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)

    print("\nTest A: constant p")
    print("max dpdx =", np.max(np.abs(dpdx_u)))
    print("max dpdy =", np.max(np.abs(dpdy_v)))

    # Test B: p = x
    x_p = (np.arange(Nx) + 0.5) * dx
    p = np.repeat(x_p[:, None], Ny, axis=1)

    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)

    print("\nTest B: p = x")
    print("dpdx min =", dpdx_u[1:Nx, :].min(),
          "max =", dpdx_u[1:Nx, :].max())
    print("dpdy max abs =", np.max(np.abs(dpdy_v[:, 1:Ny])))

    # Test C: p = y
    y_p = (np.arange(Ny) + 0.5) * dy
    p = np.repeat(y_p[None, :], Nx, axis=0)

    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)

    print("\nTest C: p = y")
    print("dpdx max abs =", np.max(np.abs(dpdx_u[1:Nx, :])))
    print("dpdy min =", dpdy_v[:, 1:Ny].min(),
          "max =", dpdy_v[:, 1:Ny].max())

# [TEST]
def test_div_grad_closure(Nx, Ny, dx, dy):
    print("\n=== div(grad p) vs Laplacian Tests ===")

    x_p = (np.arange(Nx) + 0.5) * dx
    y_p = (np.arange(Ny) + 0.5) * dy
    Xp, Yp = np.meshgrid(x_p, y_p, indexing="ij")

    # Test A
    p = np.ones((Nx, Ny))
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)
    div_grad_A = compute_divergence_mac(dpdx_u, dpdy_v, dx, dy)
    lap_A = compute_laplacian_center(p, dx, dy)
    err_A = div_grad_A[1:-1, 1:-1] - lap_A[1:-1, 1:-1]

    print("\nTest A: p = constant")
    print("max abs div_grad interior =", np.max(np.abs(div_grad_A[1:-1, 1:-1])))
    print("max abs lap interior      =", np.max(np.abs(lap_A[1:-1, 1:-1])))
    print("max abs error interior    =", np.max(np.abs(err_A)))

    # Test B
    p = Xp.copy()
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)
    div_grad_B = compute_divergence_mac(dpdx_u, dpdy_v, dx, dy)
    lap_B = compute_laplacian_center(p, dx, dy)
    err_B = div_grad_B[1:-1, 1:-1] - lap_B[1:-1, 1:-1]

    print("\nTest B: p = x")
    print("max abs div_grad interior =", np.max(np.abs(div_grad_B[1:-1, 1:-1])))
    print("max abs lap interior      =", np.max(np.abs(lap_B[1:-1, 1:-1])))
    print("max abs error interior    =", np.max(np.abs(err_B)))

    # Test C
    p = Xp**2 + Yp**2
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)
    div_grad_C = compute_divergence_mac(dpdx_u, dpdy_v, dx, dy)
    lap_C = compute_laplacian_center(p, dx, dy)
    err_C = div_grad_C[1:-1, 1:-1] - lap_C[1:-1, 1:-1]

    print("\nTest C: p = x^2 + y^2")
    print("div_grad interior min =", div_grad_C[1:-1, 1:-1].min(),
          "max =", div_grad_C[1:-1, 1:-1].max())
    print("lap interior min      =", lap_C[1:-1, 1:-1].min(),
          "max =", lap_C[1:-1, 1:-1].max())
    print("max abs error interior =", np.max(np.abs(err_C)))

    return {
        "Xp": Xp,
        "Yp": Yp,
        "div_grad_C": div_grad_C,
        "lap_C": lap_C,
    }

# [TEST]
def test_projection_mac(Nx, Ny, dx, dy, dt=1.0):
    print("\n=== Projection Test (MAC) ===")

    x_u = np.arange(Nx + 1) * dx
    u_star = np.repeat(x_u[:, None], Ny, axis=1)
    v_star = np.zeros((Nx, Ny + 1))

    div_star = compute_divergence_mac(u_star, v_star, dx, dy)

    print("\nBefore projection:")
    print("div_star min =", div_star.min(), "max =", div_star.max())
    print("max abs div_star =", np.max(np.abs(div_star)))

    rhs = div_star / dt
    p = solve_poisson_jacobi_center_simple(rhs, dx, dy, max_iter=20000, tol=1e-10)

    u_new, v_new = project_velocity_mac(u_star, v_star, p, dx, dy, dt)
    div_new = compute_divergence_mac(u_new, v_new, dx, dy)

    print("\nAfter projection:")
    print("div_new min =", div_new.min(), "max =", div_new.max())
    print("max abs div_new =", np.max(np.abs(div_new)))

    print("\nInterior after projection:")
    print("div_new interior min =", div_new[1:-1, 1:-1].min(),
          "max =", div_new[1:-1, 1:-1].max())
    print("max abs div_new interior =", np.max(np.abs(div_new[1:-1, 1:-1])))

    return {
        "u_star": u_star,
        "v_star": v_star,
        "div_star": div_star,
        "p": p,
        "u_new": u_new,
        "v_new": v_new,
        "div_new": div_new,
    }


# [TEST]
def test_diffusion_projection_mac(Nx, Ny, dx, dy, dt=1e-3, nu=0.1, U_lid=1.0):
    print("\n=== Diffusion + Projection Test (MAC) ===")

    u = np.zeros((Nx + 1, Ny))
    v = np.zeros((Nx, Ny + 1))

    results = step_diffusion_projection_mac(u, v, dx, dy, dt, nu, U_lid=U_lid)

    print("Before projection:")
    print("max abs div_star =", np.max(np.abs(results["div_star"])))

    print("\nAfter projection:")
    print("max abs div_new =", np.max(np.abs(results["div_new"])))
    print("max abs div_new interior =",
          np.max(np.abs(results["div_new"][1:-1, 1:-1])))
    print("max abs div_star interior =", np.max(np.abs(results["div_star"][1:-1, 1:-1])))
    print("max abs div_star top row   =", np.max(np.abs(results["div_star"][:, -1])))
    print("max abs div_star bottom row=", np.max(np.abs(results["div_star"][:, 0])))

    return results