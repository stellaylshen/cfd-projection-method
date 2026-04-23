# Plots for staggered_grid.py
import numpy as np
import matplotlib.pyplot as plt
from core import compute_pressure_gradient_mac
# =========================================================
# Plots
# =========================================================
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

# 可看到 p, u, v grid 分開
def plot_full_overview(Xp, Yp, Xu, Yu, Xv, Yv,
                      div_star, div_new,
                      Nx, Ny, Lx=1.0, Ly=1.0):

    fig = plt.figure(figsize=(16, 10))

    # ===== 外層 grid：2 rows × 2 cols =====
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    # ===== 左邊再切成 2x2 =====
    gs_left = gs[:, 0].subgridspec(2, 2)

    axes_left = [
        fig.add_subplot(gs_left[0, 0]),  # p
        fig.add_subplot(gs_left[0, 1]),  # u
        fig.add_subplot(gs_left[1, 0]),  # v
        fig.add_subplot(gs_left[1, 1])   # overlay
    ]

    # ===== 右邊 =====
    ax_div_star = fig.add_subplot(gs[0, 1])
    ax_div_new  = fig.add_subplot(gs[1, 1])

    # ===== 共用 grid =====
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

    # ===== 左邊 4 張 =====

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

    # ===== 右邊 divergence =====

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

# [ MAC layout ]   [ u_star ]     [ grad(p) ]
# [ u_new      ]   [ div_star ]   [ div_new ]
def plot_projection_debug(
    Xp, Yp, Xu, Yu, Xv, Yv,
    u_star, v_star,
    u_new, v_new,
    p,
    div_star, div_new,
    dx, dy,
    Nx, Ny, Lx=1.0, Ly=1.0
):
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
    import matplotlib.pyplot as plt
    import numpy as np

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

# [ MAC layout ]   [ u_star ]        [ velocity field ]   [ correction = -grad(p) ]
# [ u_new      ]   [ div_star ]      [ div_new        ]   [ debug slot            ]

def plot_projection_dashboard(
    Xp, Yp, Xu, Yu, Xv, Yv,
    u_star, v_star,
    u_new, v_new,
    p,
    div_star, div_new,
    dx, dy,
    Nx, Ny, Lx=1.0, Ly=1.0
):
    import numpy as np
    import matplotlib.pyplot as plt

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
    # (8) debug slot
    # =========================
    ax = axes[7]
    ax.axis("off")
    ax.text(
        0.5, 0.75,
        "debug slot",
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



