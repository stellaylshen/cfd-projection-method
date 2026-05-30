"""
Main driver script for the final Re=100 lid-driven cavity simulation.

Runs the ghost-cell MAC projection solver, generates benchmark
comparisons and projection diagnostics, and reproduces the figures
used in the report.
"""

import numpy as np

from core import (
    setup_mac_grid_ghost,
    run_ns_projection_mac_ghost,
    face_to_center_velocity_ghost,
)

from plots import (
    plot_final_velocity,
    plot_streamlines,
    animate_velocity_field,
    plot_projection_diagnostics_clean,
    plot_ghia_comparison_combined,
    plot_divergence_history,
    plot_poisson_residual_history,
)

from diag import (
    get_ghia_re100_data,
    compute_ghia_errors,
)

# -------------------------------------------------
# Simulation parameters
# Final report results use a 61x61 MAC pressure grid.
# Smaller grids such as 21x21 are useful only for quick debugging.
# -------------------------------------------------
Nx = 61
Ny = 61

Re = 100.0
nu = 1.0 / Re

dt = 5e-3
nsteps = 5000

# -------------------------------------------------
# Setup grid
# -------------------------------------------------
dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid_ghost(Nx, Ny)

# -------------------------------------------------
# Run simulation
# -------------------------------------------------
history = run_ns_projection_mac_ghost(
    Nx, Ny,
    dx, dy,
    nsteps=nsteps,
    dt=dt,
    nu=nu,
    U_lid=1.0,
    steady_tol=1e-6,
    min_steps=100,
    print_every=500,
)

final = history[-1]

# -------------------------------------------------
# Convert to cell-center velocity
# -------------------------------------------------
u_c, v_c = face_to_center_velocity_ghost(
    final["u"],
    final["v"],
)
# -------------------------------------------------
# Ghia benchmark comparison
# -------------------------------------------------
ghia = get_ghia_re100_data()

errors = compute_ghia_errors(u_c, v_c, Xp, Yp, ghia)

print("\n=== Ghia Benchmark ===")
print(f"u_L2 = {errors['u_L2']:.4f}")
print(f"v_L2 = {errors['v_L2']:.4f}")
print(f"u_max = {errors['u_max']:.4f}")
print(f"v_max = {errors['v_max']:.4f}")

# -------------------------------------------------
# Diagnostics
# -------------------------------------------------
print("\n=== Final Diagnostics ===")

print(
    "max |div_new| =",
    np.max(np.abs(final["div_new"]))
)

print(
    "final velocity_change =",
    final["velocity_change"]
)

print(
    "final umax =",
    final["umax"]
)

print(
    "final vmax =",
    final["vmax"]
)

u_min = np.min(u_c)
v_min = np.min(v_c)

print("final u_min =", u_min)
print("final v_min =", v_min)
# -------------------------------------------------
# Plot
# -------------------------------------------------
plot_final_velocity(
    Xp,
    Yp,
    u_c,
    v_c,
)

plot_ghia_comparison_combined(
    Xp,
    Yp,
    u_c,
    v_c,
    ghia,
    errors,
    Re,
)

plot_streamlines(Xp, Yp, u_c, v_c, Re)

plot_divergence_history(history)
# -------------------------------------------------
# Projection diagnostics
# Use an intermediate frame to visualize divergence removal.
# The final steady-state field has much weaker divergence, making
# the projection effect less visible in the diagnostic plot.
# -------------------------------------------------
diagnostic_step = 2000

diag_frame = min(
    history,
    key=lambda frame: abs(frame["step"] - diagnostic_step)
)

plot_poisson_residual_history(diag_frame)

print("Poisson diagnostic step =", diag_frame["step"])
print("SOR iterations =", diag_frame["poisson_iterations"])
print("initial SOR residual =", diag_frame["poisson_residual_history"][0])
print("final SOR residual =", diag_frame["poisson_residual_history"][-1])

diag_projection = plot_projection_diagnostics_clean(
    Xp,
    Yp,
    diag_frame["div_star"],
    diag_frame["p"],
    diag_frame["div_new"],
    Re,
)

anim = animate_velocity_field(
    history,
    Xp,
    Yp,
    face_to_center_velocity_ghost,
    skip=50,
)
anim.save("velocity_evolution.gif", writer="pillow", fps=10)
speed = np.sqrt(u_c**2 + v_c**2)
print("final speed max =", np.max(speed))
print("final umax =", final["umax"])
print("final vmax =", final["vmax"])
print("final step =", final["step"])

