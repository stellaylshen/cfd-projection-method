from core import setup_mac_grid, face_to_center_velocity, run_ns_projection_mac
from diag import get_ghia_re100_data, compute_ghia_errors
from plots import plot_grid_convergence

def run_case(N, nsteps=300, dt=5e-3, Re=100.0):
    dx, dy, _, _, _, Xp, Yp, _, _, _, _ = setup_mac_grid(N, N)
    nu = 1.0 / Re

    history = run_ns_projection_mac(
        N, N, dx, dy,
        nsteps=nsteps,
        dt=dt,
        nu=nu,
        U_lid=1.0,
    )

    final = history[-1]
    u_c, v_c = face_to_center_velocity(final["u"], final["v"])

    ghia = get_ghia_re100_data()
    errors = compute_ghia_errors(u_c, v_c, Xp, Yp, ghia)

    return errors

if __name__ == "__main__":
    grid_list = [21, 31, 41, 51]

    u_errors = []
    v_errors = []

    for N in grid_list:
        print(f"\nRunning N = {N} ...")
        errors = run_case(N)

        print(f"u_L2 = {errors['u_L2']:.4f}, v_L2 = {errors['v_L2']:.4f}")

        u_errors.append(errors["u_L2"])
        v_errors.append(errors["v_L2"])

    plot_grid_convergence(grid_list, u_errors, v_errors)

    