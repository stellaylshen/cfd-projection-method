"""
Visualization utilities for the 2D lid-driven cavity solver.

This module contains plotting functions for:
1. projection-method diagnostics,
2. velocity and pressure field visualization,
3. benchmark comparison with Ghia et al. (1982),
4. grid-refinement and convergence analysis,
5. transient flow animation.

Most functions assume a staggered MAC grid, with:
- pressure stored at cell centers,
- velocity components stored on cell faces,
- interpolated cell-centered velocities used for visualization.

Some legacy/debug plotting functions are intentionally preserved
to document the solver development and debugging process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from core import compute_pressure_gradient_mac

# directory for saved figures
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Main plotting utilities
# ============================================================
def plot_projection_dashboard(
    Xp, Yp, Xu, Yu, Xv, Yv,
    u_star, v_star,
    u_new, v_new,
    p,
    div_star, div_new,
    dx, dy,
    Nx, Ny, Lx=1.0, Ly=1.0
):
    """
    Plot a full projection-method diagnostic dashboard.

    The dashboard includes the MAC-grid layout, predictor velocity,
    projected velocity, pressure-gradient correction, divergence before
    projection, divergence after projection, and numerical divergence
    summaries.

    This plot is intended for checking whether one MAC-grid projection
    step behaves consistently.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # -------------------------
    # common grid
    # -------------------------
    x_edges = np.linspace(0.0, Lx, Nx + 1)
    y_edges = np.linspace(0.0, Ly, Ny + 1)

    def draw_base(ax):
        for x in x_edges:
            ax.axvline(x, color='gray', lw=1, alpha=0.25)
        for y in y_edges:
            ax.axhline(y, color='gray', lw=1, alpha=0.25)
        ax.set_xlim(-0.02, Lx + 0.02)
        ax.set_ylim(-0.02, Ly + 0.02)
        ax.set_aspect('equal')

    # -------------------------
    # interpolate helper
    # -------------------------
    def face_to_center(u, v):
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        return uc, vc

    # -------------------------
    # safe quiver helper
    # -------------------------
    def safe_quiver(ax, X, Y, U, V, title, scale=None):
        draw_base(ax)
        mag = np.sqrt(U**2 + V**2)
        if np.max(mag) < 1e-14:
            ax.text(
                0.5, 0.5, "≈ 0 everywhere",
                ha="center", va="center",
                transform=ax.transAxes
            )
        else:
            if scale is None:
                ax.quiver(X, Y, U, V)
            else:
                ax.quiver(X, Y, U, V, scale=scale)
        ax.set_title(title)

    # =========================
    # (1) MAC layout
    # =========================
    ax = axes[0]
    draw_base(ax)
    ax.scatter(Xp, Yp, c='black', s=30, label='p')
    ax.scatter(Xu, Yu, c='red', s=15, marker='s', label='u')
    ax.scatter(Xv, Yv, c='blue', s=15, marker='^', label='v')
    ax.set_title("MAC layout")
    ax.legend(fontsize=8, loc='upper right')

    # =========================
    # (2) u_star
    # =========================
    u_star_c, v_star_c = face_to_center(u_star, v_star)
    safe_quiver(axes[1], Xp, Yp, u_star_c, v_star_c, "u_star")

    # =========================
    # (3) velocity field (same as u_new, emphasized as actual flow)
    # =========================
    u_new_c, v_new_c = face_to_center(u_new, v_new)
    safe_quiver(axes[2], Xp, Yp, u_new_c, v_new_c, "velocity field")

    # =========================
    # (4) correction = -grad(p)
    # =========================
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)
    dpdx_c = 0.5 * (dpdx_u[:-1, :] + dpdx_u[1:, :])
    dpdy_c = 0.5 * (dpdy_v[:, :-1] + dpdy_v[:, 1:])

    corr_u = -dpdx_c
    corr_v = -dpdy_c
    safe_quiver(axes[3], Xp, Yp, corr_u, corr_v, "correction = -grad(p)")

    # =========================
    # (5) u_new
    # =========================
    safe_quiver(axes[4], Xp, Yp, u_new_c, v_new_c, "u_new")

    # =========================
    # shared color scale
    # =========================
    vmax = max(np.max(np.abs(div_star)), np.max(np.abs(div_new)))
    if vmax < 1e-14:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, 21)

    # =========================
    # (6) div_star
    # =========================
    ax = axes[5]
    draw_base(ax)
    cf1 = ax.contourf(Xp, Yp, div_star, levels=levels)
    ax.set_title("div_star")
    fig.colorbar(cf1, ax=ax)

    # =========================
    # (7) div_new
    # =========================
    ax = axes[6]
    draw_base(ax)
    cf2 = ax.contourf(Xp, Yp, div_new, levels=levels)
    ax.set_title("div_new")
    fig.colorbar(cf2, ax=ax)

    # =========================
    # (8) projection diagnostics
    # =========================
    ax = axes[7]
    ax.axis("off")
    ax.text(
        0.5, 0.75,
        "projection diagnostics",
        ha="center", va="center",
        fontsize=14,
        transform=ax.transAxes
    )
    ax.text(
        0.5, 0.48,
        f"max|div_star| = {np.max(np.abs(div_star)):.3e}",
        ha="center", va="center",
        fontsize=11,
        transform=ax.transAxes
    )
    ax.text(
        0.5, 0.34,
        f"max|div_new|  = {np.max(np.abs(div_new)):.3e}",
        ha="center", va="center",
        fontsize=11,
        transform=ax.transAxes
    )
    ax.text(
        0.5, 0.20,
        f"max|div_new interior| = {np.max(np.abs(div_new[1:-1, 1:-1])):.3e}",
        ha="center", va="center",
        fontsize=11,
        transform=ax.transAxes
    )

    plt.tight_layout()
    plt.show()

def plot_streamlines(Xp, Yp, u_c, v_c, Re):
    """
    Plot the steady cavity-flow structure using velocity magnitude and
    streamlines.

    The velocity magnitude is shown as a filled contour plot, while the
    cell-centered velocity field is used to draw streamlines.

    Used as the main qualitative flow-visualization figure.
    """
    speed = np.sqrt(u_c**2 + v_c**2)

    plt.figure(figsize=(6, 5))
    cf = plt.contourf(Xp, Yp, speed, levels=30)
    plt.streamplot(
        Xp.T, Yp.T,
        u_c.T, v_c.T,
        density=1.4,
        linewidth=0.9,
        arrowsize=0.8,
        color="k",
    )
    plt.colorbar(cf, label="velocity magnitude")
    plt.title(f"Lid-driven cavity flow at Re={Re:.0f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "streamlines_re100.png", dpi=300)
    plt.show()

def animate_velocity_field(history, Xp, Yp, face_to_center_func, skip=20):
    """
    Animate the transient evolution of the velocity field.

    Each frame shows the cell-centered velocity magnitude together with
    velocity vectors. A fixed color scale is used across the animation
    so that changes over time remain visually comparable.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # downsample arrows
    stride = max(1, Xp.shape[0] // 20)

    def get_frame_data(frame):
        u_c, v_c = face_to_center_func(frame["u"], frame["v"])
        speed = np.sqrt(u_c**2 + v_c**2)
        return u_c, v_c, speed

    # -------------------------------------------------
    # fixed color scale over the whole animation
    # -------------------------------------------------
    global_speed_max = 0.0
    for frame in history:
        _, _, speed = get_frame_data(frame)
        global_speed_max = max(global_speed_max, np.max(speed))

    levels = np.linspace(0.0, global_speed_max, 30)

    # initial frame
    u_c, v_c, speed = get_frame_data(history[0])

    contour = ax.contourf(Xp, Yp, speed, levels=30)
    q = ax.quiver(
        Xp[::stride, ::stride],
        Yp[::stride, ::stride],
        u_c[::stride, ::stride],
        v_c[::stride, ::stride],
        scale=12,
        width=0.003,
    )

    fig.colorbar(contour, ax=ax, label="velocity magnitude")
    ax.set_aspect("equal")

    def update(frame_idx):
        ax.clear()

        frame = history[frame_idx]
        u_c, v_c, speed = get_frame_data(frame)

        ax.contourf(Xp, Yp, speed, levels=levels)
        ax.quiver(
            Xp[::stride, ::stride],
            Yp[::stride, ::stride],
            u_c[::stride, ::stride],
            v_c[::stride, ::stride],
            scale=12,
            width=0.003,
        )

        ax.set_title(f"Velocity magnitude and field | step {frame['step']}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    frames = range(0, len(history), skip)

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=100,
        blit=False,
    )

    plt.show()
    return anim

def plot_projection_diagnostics_clean(Xp, Yp, div_star, p, div_new, Re=100):
    """
    Plot a clean divergence-reduction diagnostic for the projection step.

    Boundary cells are masked to emphasize the interior projection result.
    The figure compares divergence before and after projection and reports
    the reduction factor using maximum absolute divergence.

    Used as the report-ready projection diagnostic figure.
    """
    # mask boundary cells for cleaner interior visualization
    div_star_plot = div_star.copy()
    div_new_plot = div_new.copy()

    div_star_plot[[0, -1], :] = np.nan
    div_star_plot[:, [0, -1]] = np.nan
    div_new_plot[[0, -1], :] = np.nan
    div_new_plot[:, [0, -1]] = np.nan

    # use a fixed clipped scale for visualization
    div_max = 0.1
    div_levels = np.linspace(-div_max, div_max, 31)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # -------------------------
    # before projection
    # -------------------------
    cf0 = axes[0].contourf(
        Xp, Yp, div_star_plot,
        levels=div_levels,
        extend="both"
    )
    axes[0].set_title(r"$\nabla \cdot u^*$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    fig.colorbar(cf0, ax=axes[0], fraction=0.046, pad=0.04)

    # -------------------------
    # after projection
    # -------------------------
    cf1 = axes[1].contourf(
        Xp, Yp, div_new_plot,
        levels=div_levels,
        extend="both"
    )
    axes[1].set_title(r"$\nabla \cdot u^{n+1}$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.04)

    # -------------------------
    # quantitative reduction
    # -------------------------
    max_div_star = np.nanmax(np.abs(div_star_plot))
    max_div_new = np.nanmax(np.abs(div_new_plot))
    reduction = max_div_star / max_div_new if max_div_new > 0 else np.inf

    axes[2].bar(
        [r"$\nabla \cdot u^*$", r"$\nabla \cdot u^{n+1}$"],
        [max_div_star, max_div_new]
    )
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-7, 1)
    axes[2].set_ylabel("max absolute divergence")
    axes[2].set_title("divergence reduction")
    axes[2].grid(True, axis="y", alpha=0.3)

    axes[2].text(
        0.5,
        0.82,
        f"reduction ≈ {reduction:.1f}×",
        ha="center",
        va="center",
        transform=axes[2].transAxes
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "projection_diagnostics_re100.png", dpi=300)
    plt.show()

    return {
    "max_div_star": max_div_star,
    "max_div_new": max_div_new,
    "reduction_factor": reduction
    }

def plot_ghia_comparison_combined(Xp, Yp, u_c, v_c, ghia, errors, Re=100):
    """
    Compare simulated centerline velocity profiles with the Ghia et al.
    (1982) lid-driven cavity benchmark.

    The plot shows:
    - u velocity along the vertical centerline,
    - v velocity along the horizontal centerline,
    - RMSE values for both profiles.

    Used as the main benchmark-validation figure for Re=100.
    """
    Nx, Ny = u_c.shape
    i_mid = Nx // 2
    j_mid = Ny // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -------------------------------------------------
    # u velocity along vertical centerline x = 0.5
    # -------------------------------------------------
    axes[0].plot(
        u_c[i_mid, :],
        Yp[i_mid, :],
        "o-",
        label="simulation",
        markersize=4,
    )
    axes[0].plot(
        ghia["u"],
        ghia["y"],
        "s",
        label="Ghia et al. (1982)",
        markersize=4,
    )

    axes[0].set_xlabel(r"$u$ at $x=0.5$")
    axes[0].set_ylabel("y")
    axes[0].set_title("Vertical centerline velocity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[0].text(
        0.05,
        0.12,
        f"RMSE = {errors['u_L2']:.4f}",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # -------------------------------------------------
    # v velocity along horizontal centerline y = 0.5
    # -------------------------------------------------
    axes[1].plot(
        Xp[:, j_mid],
        v_c[:, j_mid],
        "o-",
        label="simulation",
        markersize=4,
    )
    axes[1].plot(
        ghia["x"],
        ghia["v"],
        "s",
        label="Ghia et al. (1982)",
        markersize=4,
    )

    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$v$ at $y=0.5$")
    axes[1].set_title("Horizontal centerline velocity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    axes[1].axhline(0, color="gray", lw=0.8, alpha=0.5)

    axes[1].text(
        0.05,
        0.12,
        f"RMSE = {errors['v_L2']:.4f}",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ghia_centerline_comparison_re100.png", dpi=300)
    plt.show()

def plot_grid_convergence_loglog(grid_list, u_errors, v_errors):
    """
    Plot grid-refinement errors on a log-log scale.

    The function compares the centerline RMSE for u and v over multiple
    grid resolutions. It is used to visualize whether the numerical
    solution improves under mesh refinement.
    """
    h = 1.0 / np.array(grid_list)

    plt.figure(figsize=(6, 5))

    plt.loglog(
        h,
        u_errors,
        "o-",
        label=r"$u$ RMSE",
        markersize=5,
    )

    plt.loglog(
        h,
        v_errors,
        "s-",
        label=r"$v$ RMSE",
        markersize=5,
    )

    # refinement goes to the right
    plt.gca().invert_xaxis()
    plt.xlabel(r"Grid spacing $h$")
    plt.ylabel("RMSE")
    plt.title("Grid refinement study")
    plt.grid(True, which="major", alpha=0.3)
    plt.legend(fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG_DIR / "grid_refinement_re100.png", dpi=300)

    plt.show()



# ============================================================
# Legacy/debug plotting utilities
# ============================================================
# These functions are retained for development history,
# intermediate projection diagnostics, and MAC-grid debugging.
# They are not required by the final benchmark/report pipeline.

def plot_mac_layout(Xp, Yp, Xu, Yu, Xv, Yv, Nx, Ny, Lx=1.0, Ly=1.0):
    plt.figure(figsize=(7, 7))

    x_edges = np.linspace(0.0, Lx, Nx + 1)
    y_edges = np.linspace(0.0, Ly, Ny + 1)

    for x in x_edges:
        plt.axvline(x, color='gray', lw=1, alpha=0.6)
    for y in y_edges:
        plt.axhline(y, color='gray', lw=1, alpha=0.6)

    plt.scatter(Xp, Yp, c='black', s=70, label='p center', zorder=3)
    plt.scatter(Xu, Yu, c='red', s=25, marker='s', label='u-face center', zorder=3)
    plt.scatter(Xv, Yv, c='blue', s=25, marker='^', label='v-face center', zorder=3)

    plt.xlim(-0.02, Lx + 0.02)
    plt.ylim(-0.02, Ly + 0.02)
    plt.gca().set_aspect('equal')
    plt.title("MAC Grid Layout")
    plt.legend(loc='upper right')
    plt.show()

def plot_divergence_field(div, Xp, Yp):
    plt.figure(figsize=(6,5))
    plt.contourf(Xp, Yp, div, levels=20)
    plt.colorbar()
    plt.title("Divergence (center)")
    plt.gca().set_aspect('equal')
    plt.show()

def plot_mac_overview(Xp, Yp, Xu, Yu, Xv, Yv, div_star, div_new, Nx, Ny, Lx=1.0, Ly=1.0):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x_edges = np.linspace(0.0, Lx, Nx + 1)
    y_edges = np.linspace(0.0, Ly, Ny + 1)

    def draw_base(ax):
        for x in x_edges:
            ax.axvline(x, color='gray', lw=1, alpha=0.35)
        for y in y_edges:
            ax.axhline(y, color='gray', lw=1, alpha=0.35)
        ax.set_xlim(-0.02, Lx + 0.02)
        ax.set_ylim(-0.02, Ly + 0.02)
        ax.set_aspect('equal')

    vmax = max(np.max(np.abs(div_star)), np.max(np.abs(div_new)))
    levels = np.linspace(-vmax, vmax, 21)    

    # --------------------------------
    # (1) MAC layout
    # --------------------------------
    ax = axes[0]
    draw_base(ax)
    ax.scatter(Xp, Yp, c='black', s=45, label='p')
    ax.scatter(Xu, Yu, c='red', s=22, marker='s', label='u')
    ax.scatter(Xv, Yv, c='blue', s=22, marker='^', label='v')
    ax.set_title("MAC layout")
    ax.legend(loc='upper right', fontsize=10)

    # --------------------------------
    # (2) div_star
    # --------------------------------
    ax = axes[1]
    draw_base(ax)
    cf1 = ax.contourf(Xp, Yp, div_star, levels=levels)
    ax.set_title("div_star (before projection)")
    fig.colorbar(cf1, ax=ax)

    # --------------------------------
    # (3) div_new
    # --------------------------------
    ax = axes[2]
    draw_base(ax)
    cf2 = ax.contourf(Xp, Yp, div_new, levels=levels)
    ax.set_title("div_new (after projection)")
    fig.colorbar(cf2, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_full_overview(Xp, Yp, Xu, Yu, Xv, Yv,
                      div_star, div_new,
                      Nx, Ny, Lx=1.0, Ly=1.0):

    """
    Visualize the MAC-grid arrangement together with divergence fields
    before and after projection.

    This function provides:
    - the staggered locations of pressure and velocity variables,
    - an overlay view of the MAC grid,
    - divergence of the predictor velocity field (div_star),
    - divergence after projection (div_new).

    Parameters
    ----------
    Xp, Yp : ndarray
        Cell-centered coordinates for pressure.

    Xu, Yu : ndarray
        Face-centered coordinates for horizontal velocity u.

    Xv, Yv : ndarray
        Face-centered coordinates for vertical velocity v.

    div_star : ndarray
        Divergence field before projection.

    div_new : ndarray
        Divergence field after projection.

    Nx, Ny : int
        Number of pressure cells in x and y directions.

    Lx, Ly : float, optional
        Physical domain size.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    gs_left = gs[:, 0].subgridspec(2, 2)

    axes_left = [
        fig.add_subplot(gs_left[0, 0]),  # p
        fig.add_subplot(gs_left[0, 1]),  # u
        fig.add_subplot(gs_left[1, 0]),  # v
        fig.add_subplot(gs_left[1, 1])   # overlay
    ]
    ax_div_star = fig.add_subplot(gs[0, 1])
    ax_div_new  = fig.add_subplot(gs[1, 1])

    x_edges = np.linspace(0.0, Lx, Nx + 1)
    y_edges = np.linspace(0.0, Ly, Ny + 1)

    def draw_base(ax):
        for x in x_edges:
            ax.axvline(x, color='gray', lw=1, alpha=0.3)
        for y in y_edges:
            ax.axhline(y, color='gray', lw=1, alpha=0.3)
        ax.set_xlim(-0.02, Lx + 0.02)
        ax.set_ylim(-0.02, Ly + 0.02)
        ax.set_aspect('equal')

    # p
    ax = axes_left[0]
    draw_base(ax)
    ax.scatter(Xp, Yp, c='black', s=40)
    ax.set_title("p-grid")

    # u
    ax = axes_left[1]
    draw_base(ax)
    ax.scatter(Xu, Yu, c='red', s=20, marker='s')
    ax.set_title("u-grid")

    # v
    ax = axes_left[2]
    draw_base(ax)
    ax.scatter(Xv, Yv, c='blue', s=20, marker='^')
    ax.set_title("v-grid")

    # overlay
    ax = axes_left[3]
    draw_base(ax)
    ax.scatter(Xp, Yp, c='black', s=40, label='p')
    ax.scatter(Xu, Yu, c='red', s=20, marker='s', label='u')
    ax.scatter(Xv, Yv, c='blue', s=20, marker='^', label='v')
    ax.set_title("overlay")
    ax.legend(fontsize=8)

    vmax = max(np.max(np.abs(div_star)), np.max(np.abs(div_new)))
    levels = np.linspace(-vmax, vmax, 21)

    draw_base(ax_div_star)
    cf1 = ax_div_star.contourf(Xp, Yp, div_star, levels=levels)
    ax_div_star.set_title("div_star")

    draw_base(ax_div_new)
    cf2 = ax_div_new.contourf(Xp, Yp, div_new, levels=levels)
    ax_div_new.set_title("div_new")

    fig.colorbar(cf1, ax=ax_div_star)
    fig.colorbar(cf2, ax=ax_div_new)

    plt.tight_layout()
    plt.show()

def plot_projection_debug(
    Xp, Yp, Xu, Yu, Xv, Yv,
    u_star, v_star,
    u_new, v_new,
    p,
    div_star, div_new,
    dx, dy,
    Nx, Ny, Lx=1.0, Ly=1.0
):
    """
    Plot a compact legacy diagnostic overview of one projection step.

    Shows the MAC layout, predictor velocity, pressure-gradient field,
    projected velocity, and divergence fields before and after projection.

    Retained for development history and projection-method debugging.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # -------------------------
    # common grid
    # -------------------------
    x_edges = np.linspace(0.0, Lx, Nx + 1)
    y_edges = np.linspace(0.0, Ly, Ny + 1)

    def draw_base(ax):
        for x in x_edges:
            ax.axvline(x, color='gray', lw=1, alpha=0.25)
        for y in y_edges:
            ax.axhline(y, color='gray', lw=1, alpha=0.25)
        ax.set_xlim(-0.02, Lx + 0.02)
        ax.set_ylim(-0.02, Ly + 0.02)
        ax.set_aspect('equal')

    # -------------------------
    # interpolate helper
    # -------------------------
    def face_to_center(u, v):
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        return uc, vc

    # =========================
    # (1) MAC layout
    # =========================
    ax = axes[0, 0]
    draw_base(ax)
    ax.scatter(Xp, Yp, c='black', s=30, label='p')
    ax.scatter(Xu, Yu, c='red', s=15, marker='s', label='u')
    ax.scatter(Xv, Yv, c='blue', s=15, marker='^', label='v')
    ax.set_title("MAC layout")
    ax.legend(fontsize=8)

    # =========================
    # (2) u_star
    # =========================
    ax = axes[0, 1]
    draw_base(ax)
    uc, vc = face_to_center(u_star, v_star)
    ax.quiver(Xp, Yp, uc, vc)
    ax.set_title("u_star")

    # =========================
    # (3) grad(p)
    # =========================
    ax = axes[0, 2]
    draw_base(ax)
    dpdx_u, dpdy_v = compute_pressure_gradient_mac(p, dx, dy)
    dpdx_c = 0.5 * (dpdx_u[:-1, :] + dpdx_u[1:, :])
    dpdy_c = 0.5 * (dpdy_v[:, :-1] + dpdy_v[:, 1:])
    ax.quiver(Xp, Yp, dpdx_c, dpdy_c)
    ax.set_title("grad(p)")

    # =========================
    # (4) u_new
    # =========================
    ax = axes[1, 0]
    draw_base(ax)
    uc, vc = face_to_center(u_new, v_new)
    ax.quiver(Xp, Yp, uc, vc)
    ax.set_title("u_new")

    # =========================
    # color scale (shared)
    # =========================
    vmax = max(np.max(np.abs(div_star)), np.max(np.abs(div_new)))
    levels = np.linspace(-vmax, vmax, 21)

    # =========================
    # (5) div_star
    # =========================
    ax = axes[1, 1]
    draw_base(ax)
    cf1 = ax.contourf(Xp, Yp, div_star, levels=levels)
    ax.set_title("div_star")

    # =========================
    # (6) div_new
    # =========================
    ax = axes[1, 2]
    draw_base(ax)
    cf2 = ax.contourf(Xp, Yp, div_new, levels=levels)
    ax.set_title("div_new")

    fig.colorbar(cf1, ax=axes[1, 1])
    fig.colorbar(cf2, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()

def plot_velocity_correction(u_star, v_star, u_new, v_new, Xp, Yp):
    # interpolate face velocities to centers
    u_star_c = 0.5 * (u_star[:-1, :] + u_star[1:, :])
    v_star_c = 0.5 * (v_star[:, :-1] + v_star[:, 1:])

    u_new_c = 0.5 * (u_new[:-1, :] + u_new[1:, :])
    v_new_c = 0.5 * (v_new[:, :-1] + v_new[:, 1:])

    du = u_new_c - u_star_c
    dv = v_new_c - v_star_c

    plt.figure(figsize=(6, 5))
    plt.quiver(Xp, Yp, du, dv)
    plt.title("velocity correction = u_new - u_star")
    plt.gca().set_aspect('equal')
    plt.show()

def plot_divergence_interior_only(div, Xp, Yp):
    div_interior = np.full_like(div, np.nan)
    div_interior[1:-1, 1:-1] = div[1:-1, 1:-1]

    plt.figure(figsize=(6, 5))
    plt.contourf(Xp, Yp, div_interior, levels=20)
    plt.colorbar()
    plt.title("div_new (interior only)")
    plt.gca().set_aspect('equal')
    plt.show()

def plot_vector_decomposition(u_star, v_star, u_new, v_new, Xp, Yp):
    # face → center
    def to_center(u, v):
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        return uc, vc

    u_star_c, v_star_c = to_center(u_star, v_star)
    u_new_c,  v_new_c  = to_center(u_new,  v_new)

    du = u_new_c - u_star_c
    dv = v_new_c - v_star_c

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------------------------
    # (1) u_star
    # -------------------------
    ax = axes[0]
    ax.quiver(Xp, Yp, u_star_c, v_star_c)
    ax.set_title("u_star")
    ax.set_aspect('equal')

    # -------------------------
    # (2) correction
    # -------------------------
    ax = axes[1]
    ax.quiver(Xp, Yp, du, dv)
    ax.set_title("correction = u_new - u_star")
    ax.set_aspect('equal')

    # -------------------------
    # (3) u_new
    # -------------------------
    ax = axes[2]
    ax.quiver(Xp, Yp, u_new_c, v_new_c)
    ax.set_title("u_new")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def plot_final_velocity(Xp, Yp, u_c, v_c):
    plt.figure(figsize=(6, 5))
    plt.quiver(Xp, Yp, u_c, v_c)
    plt.title("Final velocity field")
    plt.gca().set_aspect("equal")
    plt.show()

def plot_divergence(Xp, Yp, div, title="Final divergence"):
    plt.figure(figsize=(6, 5))
    plt.contourf(Xp, Yp, div, levels=20)
    plt.colorbar(label=title)
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.show()

def plot_centerline_u(Y, u_sim, ghia, Re=100):
    plt.figure(figsize=(5, 4))
    plt.plot(u_sim, Y, marker="o", label="simulation")
    plt.plot(ghia["u"], ghia["y"], "s", label="Ghia et al. 1982")
    plt.xlabel("u velocity at x = 0.5")
    plt.ylabel("y")
    plt.title(f"Centerline u profile, Re={Re:.0f}")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_centerline_v(X, v_sim, ghia, Re=100):
    plt.figure(figsize=(5, 4))
    plt.plot(X, v_sim, marker="o", label="simulation")
    plt.plot(ghia["x"], ghia["v"], "s", label="Ghia et al. 1982")
    plt.xlabel("x")
    plt.ylabel("v velocity at y = 0.5")
    plt.title(f"Centerline v profile, Re={Re:.0f}")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_grid_convergence(grid_list, u_errors, v_errors):
    plt.figure(figsize=(5, 4))
    plt.plot(grid_list, u_errors, "o-", label="u centerline L2 error")
    plt.plot(grid_list, v_errors, "s-", label="v centerline L2 error")
    plt.xlabel("Grid size (N)")
    plt.ylabel("L2 error")
    plt.title("Grid refinement study (Re=100)")
    plt.grid(True)
    plt.legend()
    plt.show()