import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle


# =========================================================
# 0. Small 2D toy model setup
# =========================================================

# 物理意義：等一下所有的速度、pressure、divergence，都是定義在這些格點上。
nx, ny = 7, 7
Lx, Ly = 1.0, 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing = "ij")

# 定義時間前進速度，以及 diffusion 和 pressure correction 的強度。
dt = 0.01              # 每一步前進多少時間
nu = 0.10              # 黏滯係數: 越大代表 diffusion 越強，速度會更快被抹平。
nt = 100                 # 跑 3~5 步，這裡選 4
poisson_iter = 80      # 固定迭代 80 次 (故意用最簡單 Jacobi，不追求效率)

# =========================================================
# 1. Helper functions
# =========================================================
# 把 BC 換成真正的 lid-driven cavity
U_lid = 1.0

def apply_velocity_bc(u, v):
    """
    Lid-driven cavity BC

    left wall   : u = 0, v = 0
    right wall  : u = 0, v = 0
    bottom wall : u = 0, v = 0
    top lid     : u = U_lid, v = 0
    """

    # left wall
    u[0, :] = 0.0
    v[0, :] = 0.0

    # right wall
    u[-1, :] = 0.0
    v[-1, :] = 0.0

    # bottom wall
    u[:, 0] = 0.0
    v[:, 0] = 0.0

    # top lid
    u[:, -1] = U_lid
    v[:, -1] = 0.0

    return u, v

# Laplacian 在 diffusion 裡的角色，就是量測「某格跟周圍相比有多凸、多野」。
# 如果某格比鄰居高很多，Laplacian 常會驅動它往外抹平
# 如果某格比鄰居低很多，也會被周圍補回來
# diffusion: 速度場自己在做模糊化 / 抹平化
# 這一步在 pipeline 裡只服務 diffusion predictor。
def compute_laplacian(phi, dx, dy):
    """
    2D Laplacian:
    ∇²phi = d²phi/dx² + d²phi/dy²
    """
    lap = np.zeros_like(phi)

    lap[1:-1, 1:-1] = (
        (phi[2:, 1:-1] - 2.0 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx**2
        + (phi[1:-1, 2:] - 2.0 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy**2
    )

    return lap

# divergence: 哪裡在漏水 (衡量一個小區塊附近，流體是不是在「淨流出」。)
# divergence > 0：像這格在往外吐水
# divergence < 0：像這格在吸水
# divergence = 0：流進跟流出平衡，沒有憑空生滅
# 這一步在 pipeline 裡的位置：
# predictor 後算一次，得到 div_star (修之前漏多少)
# projection 後再算一次，得到 div_new (修之前漏多少)
def compute_divergence(u, v, dx, dy):
    """
    div(u, v) = du/dx + dv/dy
    """
    div = np.zeros_like(u)

    div[1:-1, 1:-1] = (
        (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
        + (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
    )

    return div

# pressure gradient: 一個修正向量場，把速度重新調整
# 如果某處壓力高、隔壁壓力低，就會有一個由高往低的 gradient。
# 這一步在 pipeline 裡的位置：pressure Poisson 解完之後，立刻接 projection。
def compute_pressure_gradient(p, dx, dy):
    """
    ∇p = (dp/dx, dp/dy)
    """
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)

    dpdx[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * dx)
    dpdy[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * dy)

    return dpdx, dpdy

# Jacobi iteration method
def solve_poisson_jacobi(b, dx, dy, n_iter = 80, return_history = False):
    """
    Solve:
        ∇²p = b
    with simple Jacobi iteration.

    We use Neumann-like BC:
        dp/dn = 0 on boundaries
    and pin one point:
        p[0,0] = 0
    so the pressure has a reference value.
    """
    p = np.zeros_like(b)
    residual_history = []

    for _ in range(n_iter):
        pn = p.copy()

        p[1:-1, 1:-1] = (
            (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy**2
            + (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx**2
            - b[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2.0 * (dx**2 + dy**2))

        # Neumann BC: zero normal derivative
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        # Fix one reference point
        p[0, 0] = 0.0

        if return_history:
            r = compute_poisson_residual(p, b, dx, dy)
            residual_history.append(np.max(np.abs(r[1:-1, 1:-1])))

    if return_history:
        return p, residual_history
    return p

# Gauss-Seidel iteration method
# 掃格點時，更新到哪裡就立刻寫回 p，本輪後面格點可以直接吃到這些新值。
def solve_poisson_gs_until_converged(b, dx, dy, tol = 1e-4, max_iter = 5000, return_history = False):
    p = np.zeros_like(b)
    residual_history = []

    coef = 2.0 * (dx**2 + dy**2)

    for it in range(max_iter):
        for i in range(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                p[i, j] = (
                    (p[i+1, j] + p[i-1, j]) * dy**2
                    + (p[i, j+1] + p[i, j-1]) * dx**2
                    - b[i, j] * dx**2 * dy**2
                ) / coef

        # pressure BC
        p[0, :]  = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0]  = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, 0] = 0.0

        r = compute_poisson_residual(p, b, dx, dy)
        res_max = np.max(np.abs(r[1:-1, 1:-1]))
        residual_history.append(res_max)

        if res_max < tol:
            break

    if return_history:
        return p, residual_history, it + 1
    return p

# SOR iteration method (Successive over-relaxation)
# a variant of the Gauss–Seidel method
def solve_poisson_sor(b, dx, dy, n_iter = 80, omega = 1.5, return_history = False):
    p = np.zeros_like(b)
    residual_history = []

    coef = 2.0 * (dx**2 + dy**2)

    for _ in range(n_iter):
        for i in range(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                p_gs = (
                    (p[i+1, j] + p[i-1, j]) * dy**2
                    + (p[i, j+1] + p[i, j-1]) * dx**2
                    - b[i, j] * dx**2 * dy**2
                ) / coef

                # 1 < ω < 2 是 over-relaxation ; ω = 1 是 GS
                p[i, j] = (1.0 - omega) * p[i, j] + omega * p_gs

        # BC
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, 0] = 0.0

        if return_history:
            r = compute_poisson_residual(p, b, dx, dy)
            residual_history.append(np.max(np.abs(r[1:-1, 1:-1])))

    if return_history:
        return p, residual_history
    return p


def compute_poisson_residual(p, b, dx, dy):
    """
    residual r = b - Laplacian(p)
    """
    lap_p = np.zeros_like(p)

    lap_p[1:-1, 1:-1] = (
        (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx**2
        + (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dy**2
    )

    r = np.zeros_like(p)
    r[1:-1, 1:-1] = b[1:-1, 1:-1] - lap_p[1:-1, 1:-1]

    return r

def compute_advection(u, v, dx, dy):
    """
    Compute advection terms for 2D velocity field:

    adv_u = u * du/dx + v * du/dy
    adv_v = u * dv/dx + v * dv/dy

    return:
        adv_u, adv_v
    """
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dvdx = np.zeros_like(v)
    dvdy = np.zeros_like(v)

    # central difference
    dudx[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
    dudy[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)

    dvdx[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
    dvdy[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    adv_u[1:-1, 1:-1] = (
        u[1:-1, 1:-1] * dudx[1:-1, 1:-1]
        + v[1:-1, 1:-1] * dudy[1:-1, 1:-1]
    )

    adv_v[1:-1, 1:-1] = (
        u[1:-1, 1:-1] * dvdx[1:-1, 1:-1]
        + v[1:-1, 1:-1] * dvdy[1:-1, 1:-1]
    )

    return adv_u, adv_v





# =========================================================
# 2. Visualization helpers
# =========================================================
def annotate_scalar(ax, field, fmt="{:.2f}", fontsize=8):
    """
    Put a number on every cell for scalar fields.
    """
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            ax.text(
                j, i, fmt.format(field[i, j]),
                ha="center", va="center",
                color="black", fontsize=fontsize
            )


def annotate_uv(ax, u, v, fontsize=6):
    """
    Put (u, v) in every cell.
    7x7 is small enough, so this is still readable.
    """
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            txt = f"{u[i, j]:.2f}\n{v[i, j]:.2f}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="darkred", fontsize=fontsize
            )


def plot_timestep(step, u, v, div_star, p, div_new):
    """
    Show one timestep with 4 panels:
    1) velocity field (quiver + annotated u,v)
    2) div_star
    3) pressure
    4) div_new
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # -----------------------------------------------------
    # Panel 1: velocity field
    # -----------------------------------------------------
    # background just for reference
    speed = np.sqrt(u**2 + v**2)
    im0 = axes[0].imshow(speed.T, origin="lower", cmap="Blues")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # quiver needs x-horizontal, y-vertical display order
    axes[0].quiver(
        np.arange(ny), np.arange(nx),
        u.T, v.T,
        angles="xy", scale_units="xy", scale=1.0, color="k"
    )

    annotate_uv(axes[0], u, v, fontsize=6)

    axes[0].set_title(f"step {step}: velocity (u,v)")
    axes[0].set_xticks(range(ny))
    axes[0].set_yticks(range(nx))
    axes[0].invert_yaxis()

    # -----------------------------------------------------
    # Panel 2: div_star
    # -----------------------------------------------------
    im1 = axes[1].imshow(div_star.T, origin="lower", cmap="coolwarm")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    annotate_scalar(axes[1], div_star, fmt="{:.2f}", fontsize=8)
    axes[1].set_title(f"step {step}: div_star")
    axes[1].set_xticks(range(ny))
    axes[1].set_yticks(range(nx))
    axes[1].invert_yaxis()

    # -----------------------------------------------------
    # Panel 3: pressure
    # -----------------------------------------------------
    im2 = axes[2].imshow(p.T, origin="lower", cmap="viridis")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    annotate_scalar(axes[2], p, fmt="{:.2f}", fontsize=8)
    axes[2].set_title(f"step {step}: pressure p")
    axes[2].set_xticks(range(ny))
    axes[2].set_yticks(range(nx))
    axes[2].invert_yaxis()

    # -----------------------------------------------------
    # Panel 4: div_new
    # -----------------------------------------------------
    im3 = axes[3].imshow(div_new.T, origin="lower", cmap="coolwarm")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    annotate_scalar(axes[3], div_new, fmt="{:.2f}", fontsize=8)
    axes[3].set_title(f"step {step}: div_new")
    axes[3].set_xticks(range(ny))
    axes[3].set_yticks(range(nx))
    axes[3].invert_yaxis()

    plt.tight_layout()


def plot_all_timesteps(history):
    """
    history: list of dict
        each dict contains:
        {
            'step': int,
            'u': ...,
            'v': ...,
            'div_star': ...,
            'p': ...,
            'div_new': ...
        }

    output:
        rows = timesteps
        cols = [velocity, div_star, p, div_new]
    """
    nsteps = len(history)
    fig, axes = plt.subplots(nsteps, 4, figsize=(22, 5 * nsteps))

    # 如果只有 1 個 timestep，axes 不會是 2D，這裡強制變成 2D
    if nsteps == 1:
        axes = axes[np.newaxis, :]

    for row, data in enumerate(history):
        step = data["step"]
        u = data["u"]
        v = data["v"]
        div_star = data["div_star"]
        p = data["p"]
        div_new = data["div_new"]

        # -------------------------------
        # Col 0: velocity
        # -------------------------------
        ax = axes[row, 0]
        speed = np.sqrt(u**2 + v**2)
        im0 = ax.imshow(speed.T, origin="lower", cmap="Blues")
        fig.colorbar(im0, ax=ax, fraction=0.046)

        ax.quiver(
            np.arange(ny), np.arange(nx),
            u.T, v.T,
            angles="xy", scale_units="xy", scale=1.0, color="k"
        )
        annotate_uv(ax, u, v, fontsize=6)
        ax.set_title(f"step {step}: velocity (u,v)")
        ax.set_xticks(range(ny))
        ax.set_yticks(range(nx))
        ax.invert_yaxis()

        # -------------------------------
        # Col 1: div_star
        # -------------------------------
        ax = axes[row, 1]
        im1 = ax.imshow(div_star.T, origin="lower", cmap="coolwarm")
        fig.colorbar(im1, ax=ax, fraction=0.046)
        annotate_scalar(ax, div_star, fmt="{:.2f}", fontsize=8)
        ax.set_title(f"step {step}: div_star")
        ax.set_xticks(range(ny))
        ax.set_yticks(range(nx))
        ax.invert_yaxis()

        # -------------------------------
        # Col 2: pressure
        # -------------------------------
        ax = axes[row, 2]
        im2 = ax.imshow(p.T, origin="lower", cmap="viridis")
        fig.colorbar(im2, ax=ax, fraction=0.046)
        annotate_scalar(ax, p, fmt="{:.2f}", fontsize=8)
        ax.set_title(f"step {step}: pressure p")
        ax.set_xticks(range(ny))
        ax.set_yticks(range(nx))
        ax.invert_yaxis()

        # -------------------------------
        # Col 3: div_new
        # -------------------------------
        ax = axes[row, 3]
        im3 = ax.imshow(div_new.T, origin="lower", cmap="coolwarm")
        fig.colorbar(im3, ax=ax, fraction=0.046)
        annotate_scalar(ax, div_new, fmt="{:.2f}", fontsize=8)
        ax.set_title(f"step {step}: div_new")
        ax.set_xticks(range(ny))
        ax.set_yticks(range(nx))
        ax.invert_yaxis()

    plt.tight_layout()
    
def annotate_scalar(ax, field, fmt="{:.2f}", fontsize=8):
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            ax.text(
                j, i, fmt.format(field[i, j]),
                ha="center", va="center",
                color="black", fontsize=fontsize
            )

def annotate_uv(ax, u, v, fontsize=6):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            txt = f"{u[i, j]:.2f}\n{v[i, j]:.2f}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="darkred", fontsize=fontsize
            )

def slider_view(history):
    """
    用 slider 拖 timestep，更新同一組 4 張圖：
    1) velocity
    2) div_star
    3) pressure
    4) div_new
    """

    nsteps = len(history)
    if nsteps == 0:
        print("history is empty.")
        return

    # --------------------------------------------------
    # 建立 figure 和 4 個 subplot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    plt.subplots_adjust(bottom=0.20)  # 留空間給 slider

    # --------------------------------------------------
    # 先畫 step 1
    # --------------------------------------------------
    data0 = history[0]
    step = data0["step"]
    u = data0["u"]
    v = data0["v"]
    div_star = data0["div_star"]
    p = data0["p"]
    div_new = data0["div_new"]

    # ===== Panel 1: velocity =====
    speed = np.sqrt(u**2 + v**2)
    im0 = axes[0].imshow(speed.T, origin="lower", cmap="Blues")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046)

    q = axes[0].quiver(
        np.arange(u.shape[1]), np.arange(u.shape[0]),
        u.T, v.T,
        angles="xy", scale_units="xy", scale=1.0, color="k"
    )

    axes[0].set_title(f"step {step}: velocity (u,v)", pad=12)
    axes[0].set_xticks(range(u.shape[1]))
    axes[0].set_yticks(range(u.shape[0]))

    # 先存文字物件，之後更新時要刪掉重畫
    vel_texts = []
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            t = axes[0].text(
                j, i, f"{u[i,j]:.2f}\n{v[i,j]:.2f}",
                ha="center", va="center",
                color="darkred", fontsize=6
            )
            vel_texts.append(t)

    # ===== Panel 2: div_star =====
    im1 = axes[1].imshow(div_star.T, origin="lower", cmap="coolwarm")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"step {step}: div_star", pad=12)
    axes[1].set_xticks(range(div_star.shape[1]))
    axes[1].set_yticks(range(div_star.shape[0]))

    div_star_texts = []
    for i in range(div_star.shape[0]):
        for j in range(div_star.shape[1]):
            t = axes[1].text(
                j, i, f"{div_star[i,j]:.2f}",
                ha="center", va="center",
                color="black", fontsize=8
            )
            div_star_texts.append(t)

    # ===== Panel 3: pressure =====
    im2 = axes[2].imshow(p.T, origin="lower", cmap="viridis")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].set_title(f"step {step}: pressure p", pad=12)
    axes[2].set_xticks(range(p.shape[1]))
    axes[2].set_yticks(range(p.shape[0]))

    p_texts = []
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            t = axes[2].text(
                j, i, f"{p[i,j]:.2f}",
                ha="center", va="center",
                color="black", fontsize=8
            )
            p_texts.append(t)

    # ===== Panel 4: div_new =====
    im3 = axes[3].imshow(div_new.T, origin="lower", cmap="coolwarm")
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046)
    axes[3].set_title(f"step {step}: div_new", pad=12)
    axes[3].set_xticks(range(div_new.shape[1]))
    axes[3].set_yticks(range(div_new.shape[0]))

    div_new_texts = []
    for i in range(div_new.shape[0]):
        for j in range(div_new.shape[1]):
            t = axes[3].text(
                j, i, f"{div_new[i,j]:.2f}",
                ha="center", va="center",
                color="black", fontsize=8
            )
            div_new_texts.append(t)

    # --------------------------------------------------
    # 建 slider
    # --------------------------------------------------
    ax_slider = plt.axes([0.20, 0.06, 0.60, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="timestep",
        valmin=1,
        valmax=nsteps,
        valinit=1,
        valstep=1

    )
    # 在slider 上直接顯示 1 2 3 4 刻度（tick labels）
    slider.ax.set_xticks(np.arange(1, nsteps+1))
    slider.ax.set_xticklabels([str(i) for i in range(1, nsteps+1)])
    slider.ax.tick_params(labelsize=10)

    # --------------------------------------------------
    # update function: 當 slider 改變時更新圖
    # --------------------------------------------------
    def update(val):
        nonlocal vel_texts, div_star_texts, p_texts, div_new_texts, q

        idx = int(slider.val) - 1
        data = history[idx]

        step = data["step"]
        u = data["u"]
        v = data["v"]
        div_star = data["div_star"]
        p = data["p"]
        div_new = data["div_new"]

        # ------------------------------
        # 更新 Panel 1: velocity
        # ------------------------------
        speed = np.sqrt(u**2 + v**2)
        im0.set_data(speed.T)
        im0.set_clim(vmin=np.min(speed), vmax=np.max(speed) + 1e-12)

        q.set_UVC(u.T, v.T)

        for t in vel_texts:
            t.remove()
        vel_texts = []

        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                t = axes[0].text(
                    j, i, f"{u[i,j]:.2f}\n{v[i,j]:.2f}",
                    ha="center", va="center",
                    color="darkred", fontsize=6
                )
                vel_texts.append(t)

        axes[0].set_title(f"step {step}: velocity (u,v)", pad=12)

        # ------------------------------
        # 更新 Panel 2: div_star
        # ------------------------------
        im1.set_data(div_star.T)
        im1.set_clim(vmin=np.min(div_star), vmax=np.max(div_star) + 1e-12)

        for t in div_star_texts:
            t.remove()
        div_star_texts = []

        for i in range(div_star.shape[0]):
            for j in range(div_star.shape[1]):
                t = axes[1].text(
                    j, i, f"{div_star[i,j]:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=8
                )
                div_star_texts.append(t)

        axes[1].set_title(f"step {step}: div_star", pad=12)

        # ------------------------------
        # 更新 Panel 3: pressure
        # ------------------------------
        im2.set_data(p.T)
        im2.set_clim(vmin=np.min(p), vmax=np.max(p) + 1e-12)

        for t in p_texts:
            t.remove()
        p_texts = []

        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                t = axes[2].text(
                    j, i, f"{p[i,j]:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=8
                )
                p_texts.append(t)

        axes[2].set_title(f"step {step}: pressure p", pad=12)

        # ------------------------------
        # 更新 Panel 4: div_new
        # ------------------------------
        im3.set_data(div_new.T)
        im3.set_clim(vmin=np.min(div_new), vmax=np.max(div_new) + 1e-12)

        for t in div_new_texts:
            t.remove()
        div_new_texts = []

        for i in range(div_new.shape[0]):
            for j in range(div_new.shape[1]):
                t = axes[3].text(
                    j, i, f"{div_new[i,j]:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=8
                )
                div_new_texts.append(t)

        axes[3].set_title(f"step {step}: div_new", pad=12)

        fig.canvas.draw_idle()

    slider.on_changed(update)

def slider_view_6panel(history):

    nsteps = len(history)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(bottom=0.20)

    def draw(step_idx):

        data = history[step_idx]

        u_n = data["u_n"]
        v_n = data["v_n"]

        u_star = data["u_star"]
        v_star = data["v_star"]

        div_star = data["div_star"]
        p = data["p"]

        u_new = data["u_new"]
        v_new = data["v_new"]

        div_new = data["div_new"]

        fields = [
            ("u_n", u_n, v_n),
            ("u_star", u_star, v_star),
            ("div_star", div_star, None),
            ("pressure", p, None),
            ("u_new", u_new, v_new),
            ("div_new", div_new, None)
        ]

        for ax in axes.flat:
            ax.clear()

        for i, (name, f, f2) in enumerate(fields):

            ax = axes.flat[i]

            if f2 is None:
                im = ax.imshow(f.T, origin="lower", cmap="coolwarm")
                for ii in range(f.shape[0]):
                    for jj in range(f.shape[1]):
                        ax.text(jj, ii, f"{f[ii,jj]:.2f}",
                                ha="center", va="center", fontsize=8)

            else:
                speed = np.sqrt(f**2 + f2**2)
                im = ax.imshow(speed.T, origin="lower", cmap="Blues")

                ax.quiver(
                    np.arange(f.shape[1]), np.arange(f.shape[0]),
                    f.T, f2.T,
                    angles="xy", scale_units="xy", scale=1
                )

                for ii in range(f.shape[0]):
                    for jj in range(f.shape[1]):
                        ax.text(jj, ii, f"{f[ii,jj]:.2f}",
                                ha="center", va="center", fontsize=6)

            ax.set_title(name)
            ax.set_xticks(range(f.shape[1]))
            ax.set_yticks(range(f.shape[0]))

        fig.suptitle(f"STEP {data['step']}", fontsize=16)

    # 初始畫面
    draw(0)

    # slider
    from matplotlib.widgets import Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])

    slider = Slider(ax_slider, "timestep", 1, nsteps, valinit=1, valstep=1)

    def update(val):
        idx = int(slider.val) - 1
        draw(idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    #plt.show()

def compute_global_ranges(history):
    """
    掃過整個 history，為不同欄位找全域固定色階
    """

    p_min = min(np.min(data["p"]) for data in history)
    p_max = max(np.max(data["p"]) for data in history)

    div_star_min = min(np.min(data["div_star"]) for data in history)
    div_star_max = max(np.max(data["div_star"]) for data in history)

    div_new_min = min(np.min(data["div_new"]) for data in history)
    div_new_max = max(np.max(data["div_new"]) for data in history)

    corr_mag_max = 0.0
    vel_mag_max = 0.0
    corr_vec_abs_max = 0.0

    for data in history:
        du_corr = data["u_new"] - data["u_star"]
        dv_corr = data["v_new"] - data["v_star"]
        corr_mag = np.sqrt(du_corr**2 + dv_corr**2)

        u_n = data["u_n"]
        v_n = data["v_n"]
        u_star = data["u_star"]
        v_star = data["v_star"]
        u_new = data["u_new"]
        v_new = data["v_new"]

        vel_mag_max = max(
            vel_mag_max,
            np.max(np.sqrt(u_n**2 + v_n**2)),
            np.max(np.sqrt(u_star**2 + v_star**2)),
            np.max(np.sqrt(u_new**2 + v_new**2))
        )

        corr_mag_max = max(corr_mag_max, np.max(corr_mag))
        corr_vec_abs_max = max(
            corr_vec_abs_max,
            np.max(np.abs(du_corr)),
            np.max(np.abs(dv_corr))
        )

    return {
        "p": (p_min, p_max),
        "div_star": (div_star_min, div_star_max),
        "div_new": (div_new_min, div_new_max),
        "corr_mag": (0.0, corr_mag_max),
        "vel_mag": (0.0, vel_mag_max),
        "corr_vec_abs": (-corr_vec_abs_max, corr_vec_abs_max),
    }

def slider_view_8panel(history, dx, dy, fixed_scale=True):
    """
    2x4 panel slider viewer

    Top row:
        u_n | u_star | correction | u_new

    Bottom row:
        div_star | pressure | div_new | corr_mag

    Extra:
    - highlight max-|div_star| cell with red box
    - draw pressure gradient arrow on pressure panel
    - optionally use fixed global color scales across all timesteps
    """

    if len(history) == 0:
        print("history is empty.")
        return

    nsteps = len(history)

    if fixed_scale:
        ranges = compute_global_ranges(history)
    else:
        ranges = None

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    plt.subplots_adjust(bottom=0.18, top=0.90, wspace=0.35, hspace=0.35)

    ax_slider = plt.axes([0.20, 0.06, 0.60, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="timestep",
        valmin=1,
        valmax=nsteps,
        valinit=1,
        valstep=1
    )

    def draw_velocity_panel(ax, u, v, title, vmin=None, vmax=None):
        speed = np.sqrt(u**2 + v**2)
        im = ax.imshow(speed.T, origin="lower", cmap="Blues", vmin=vmin, vmax=vmax)

        ax.quiver(
            np.arange(u.shape[1]),
            np.arange(u.shape[0]),
            u.T, v.T,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="k"
        )

        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                txt = f"{u[i,j]:.2f}\n{v[i,j]:.2f}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=6, color="darkred"
                )

        ax.set_title(title, pad=12)
        ax.set_xticks(range(u.shape[1]))
        ax.set_yticks(range(u.shape[0]))
        return im

    def draw_scalar_panel(ax, field, title, cmap="coolwarm", fontsize=8, vmin=None, vmax=None):
        im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                ax.text(
                    j, i, f"{field[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=fontsize, color="black"
                )

        ax.set_title(title, pad=12)
        ax.set_xticks(range(field.shape[1]))
        ax.set_yticks(range(field.shape[0]))
        return im

    def draw(step_idx):
        data = history[step_idx]
        step = data["step"]

        u_n = data["u_n"]
        v_n = data["v_n"]

        u_star = data["u_star"]
        v_star = data["v_star"]

        div_star = data["div_star"]
        p = data["p"]

        u_new = data["u_new"]
        v_new = data["v_new"]

        div_new = data["div_new"]

        du_corr = u_new - u_star
        dv_corr = v_new - v_star
        corr_mag = np.sqrt(du_corr**2 + dv_corr**2)

        dpdx = np.zeros_like(p)
        dpdy = np.zeros_like(p)
        dpdx[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * dx)
        dpdy[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * dy)

        abs_div_core = np.abs(div_star[1:-1, 1:-1])
        i_local, j_local = np.unravel_index(np.argmax(abs_div_core), abs_div_core.shape)
        i_max = i_local + 1
        j_max = j_local + 1

        for ax in axes.flat:
            ax.clear()

        if fixed_scale:
            vel_vmin, vel_vmax = ranges["vel_mag"]
            p_vmin, p_vmax = ranges["p"]
            ds_vmin, ds_vmax = ranges["div_star"]
            dn_vmin, dn_vmax = ranges["div_new"]
            cm_vmin, cm_vmax = ranges["corr_mag"]
            corr_vmin, corr_vmax = ranges["corr_vec_abs"]
        else:
            vel_vmin, vel_vmax = None, None
            p_vmin, p_vmax = None, None
            ds_vmin, ds_vmax = None, None
            dn_vmin, dn_vmax = None, None
            cm_vmin, cm_vmax = None, None
            corr_vmin, corr_vmax = None, None

        # top row
        draw_velocity_panel(axes[0, 0], u_n, v_n, "u_n", vmin=vel_vmin, vmax=vel_vmax)
        draw_velocity_panel(axes[0, 1], u_star, v_star, "u_star", vmin=vel_vmin, vmax=vel_vmax)
        draw_velocity_panel(axes[0, 2], du_corr, dv_corr, "correction = u_new - u_star", vmin=cm_vmin, vmax=cm_vmax)
        draw_velocity_panel(axes[0, 3], u_new, v_new, "u_new", vmin=vel_vmin, vmax=vel_vmax)

        # bottom row
        draw_scalar_panel(axes[1, 0], div_star, "div_star", cmap="coolwarm", fontsize=8, vmin=ds_vmin, vmax=ds_vmax)
        draw_scalar_panel(axes[1, 1], p, "pressure p", cmap="viridis", fontsize=8, vmin=p_vmin, vmax=p_vmax)
        draw_scalar_panel(axes[1, 2], div_new, "div_new", cmap="coolwarm", fontsize=8, vmin=dn_vmin, vmax=dn_vmax)
        draw_scalar_panel(axes[1, 3], corr_mag, "corr_mag", cmap="magma", fontsize=8, vmin=cm_vmin, vmax=cm_vmax)

        # red boxes
        for ax in [axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3], axes[0, 2]]:
            ax.add_patch(
                Rectangle(
                    (j_max - 0.5, i_max - 0.5),
                    1.0, 1.0,
                    fill=False,
                    edgecolor="red",
                    linewidth=2
                )
            )

        # ------------------------------------------
        # keep scale information, but cap max arrow length
        # ------------------------------------------
        gx = dpdy[i_max, j_max]   # plot-x component
        gy = dpdx[i_max, j_max]   # plot-y component

        gnorm = np.sqrt(gx**2 + gy**2) + 1e-12

        base_scale = 0.20      # 原本想用的比例
        max_display_len = 1.0  # 最長不要超過大約一格

        scale_arrow = min(base_scale, max_display_len / gnorm)

        # grad(p)
        axes[1, 1].arrow(
            j_max, i_max,
            scale_arrow * gx,
            scale_arrow * gy,
            head_width=0.18,
            head_length=0.18,
            fc="black", ec="black",
            linewidth=2
        )

        # -grad(p) = correction direction
        axes[1, 1].arrow(
            j_max, i_max,
            -scale_arrow * gx,
            -scale_arrow * gy,
            head_width=0.18,
            head_length=0.18,
            fc="cyan", ec="cyan",
            linewidth=2
        )

        # correction arrow on correction panel
        axes[0, 2].arrow(
            j_max, i_max,
            du_corr[i_max, j_max],
            dv_corr[i_max, j_max],
            head_width=0.18,
            head_length=0.18,
            fc="lime", ec="lime",
            linewidth=2
        )

        scale_mode = "fixed global scales" if fixed_scale else "auto scales"
        fig.suptitle(
            f"STEP {step}   |   red box = max |div_star| cell   |   black = grad(p), cyan = -grad(p)   |   {scale_mode}",
            fontsize=14
        )

        fig.canvas.draw_idle()

    def update(val):
        idx = int(slider.val) - 1
        draw(idx)

    slider.on_changed(update)
    draw(0)
    plt.show()

def slider_view_ij_xy(history):
    """
    同一個 timestep 顯示：
    1) u_star 的 ij view
    2) u_star 的 xy view
    3) div_star
    4) pressure

    目的：
    專門比較 array storage vs physical display
    """

    if len(history) == 0:
        print("history is empty.")
        return

    nsteps = len(history)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plt.subplots_adjust(bottom=0.16, top=0.90, wspace=0.28, hspace=0.30)

    ax_slider = plt.axes([0.20, 0.06, 0.60, 0.04])
    slider = Slider(
        ax=ax_slider,
        label="timestep",
        valmin=1,
        valmax=nsteps,
        valinit=1,
        valstep=1
    )

    def draw(step_idx):
        data = history[step_idx]

        step = data["step"]
        u_star = data["u_star"]
        v_star = data["v_star"]
        div_star = data["div_star"]
        p = data["p"]

        # 用 speed 當成 u_star 的背景
        speed_star = np.sqrt(u_star**2 + v_star**2)

        # 找 max |div_star| 的 interior cell
        abs_div_core = np.abs(div_star[1:-1, 1:-1])
        i_local, j_local = np.unravel_index(np.argmax(abs_div_core), abs_div_core.shape)
        i_max = i_local + 1
        j_max = j_local + 1

        for ax in axes.flat:
            ax.clear()

        # ==================================================
        # (1) ij view : matrix storage
        # ==================================================
        ax = axes[0, 0]
        ax.imshow(speed_star, cmap="Blues")
        ax.set_title("u_star : ij view (matrix storage)", pad=12)
        ax.set_xticks(range(speed_star.shape[1]))
        ax.set_yticks(range(speed_star.shape[0]))

        # 每格標 (i,j) 和 speed
        for i in range(speed_star.shape[0]):
            for j in range(speed_star.shape[1]):
                txt = f"({i},{j})\n{speed_star[i,j]:.2f}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=7, color="darkred"
                )

        # 紅框：同一個 max-div cell
        ax.add_patch(
            Rectangle(
                (j_max - 0.5, i_max - 0.5),
                1.0, 1.0,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

        # ==================================================
        # (2) xy view : physical space
        # ==================================================
        ax = axes[0, 1]
        ax.imshow(speed_star.T, origin="lower", cmap="Blues")

        ax.quiver(
            np.arange(u_star.shape[1]),
            np.arange(u_star.shape[0]),
            u_star.T, v_star.T,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="k"
        )

        ax.set_title("u_star : xy view (physical space)", pad=12)
        ax.set_xticks(range(speed_star.shape[1]))
        ax.set_yticks(range(speed_star.shape[0]))

        # 每格也標 (i,j) 和 speed
        # 注意：這裡標的還是原始 data 的 index，不是 plot 的 row/col
        for i in range(speed_star.shape[0]):
            for j in range(speed_star.shape[1]):
                txt = f"({i},{j})\n{speed_star[i,j]:.2f}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=7, color="darkred"
                )

        ax.add_patch(
            Rectangle(
                (j_max - 0.5, i_max - 0.5),
                1.0, 1.0,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

        # ==================================================
        # (3) div_star
        # ==================================================
        ax = axes[1, 0]
        ax.imshow(div_star.T, origin="lower", cmap="coolwarm")
        ax.set_title("div_star", pad=12)
        ax.set_xticks(range(div_star.shape[1]))
        ax.set_yticks(range(div_star.shape[0]))

        for i in range(div_star.shape[0]):
            for j in range(div_star.shape[1]):
                ax.text(
                    j, i, f"{div_star[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=8, color="black"
                )

        ax.add_patch(
            Rectangle(
                (j_max - 0.5, i_max - 0.5),
                1.0, 1.0,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

        # ==================================================
        # (4) pressure
        # ==================================================
        ax = axes[1, 1]
        ax.imshow(p.T, origin="lower", cmap="viridis")
        ax.set_title("pressure p", pad=12)
        ax.set_xticks(range(p.shape[1]))
        ax.set_yticks(range(p.shape[0]))

        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(
                    j, i, f"{p[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=8, color="black"
                )

        ax.add_patch(
            Rectangle(
                (j_max - 0.5, i_max - 0.5),
                1.0, 1.0,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

        fig.suptitle(
            f"STEP {step}   |   red box = max |div_star| cell",
            fontsize=16
        )

        fig.canvas.draw_idle()

    def update(val):
        idx = int(slider.val) - 1
        draw(idx)

    slider.on_changed(update)
    draw(0)
    #plt.show()

# =========================================================
# 3. Initial condition
# =========================================================
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# 初始時整個 cavity 靜止，只有上蓋邊界會被設成 moving lid
u, v = apply_velocity_bc(u, v)

print("Initial u:")
print(np.round(u, 2))
print("\nInitial v:")
print(np.round(v, 2))

# =========================================================
# 4. Time stepping: diffusion predictor + projection
# =========================================================
history = []

saved_b = None
saved_div_star = None
saved_u_star = None
saved_v_star = None

for n in range(1, nt + 1):

    # -----------------------------------------------------
    # Step 1. diffusion predictor
    # u* = u^n + dt * nu * ∇²u^n
    # v* = v^n + dt * nu * ∇²v^n
    # -----------------------------------------------------
    lap_u = compute_laplacian(u, dx, dy)
    lap_v = compute_laplacian(v, dx, dy)

    # 加入 advection
    #adv_u, adv_v = compute_advection(u, v, dx, dy)
    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    u_star = u.copy()
    v_star = v.copy()

    u_star[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + dt * (
            -adv_u[1:-1, 1:-1]
            + nu * lap_u[1:-1, 1:-1]
        )
    )

    v_star[1:-1, 1:-1] = (
        v[1:-1, 1:-1]
        + dt * (
            -adv_v[1:-1, 1:-1]
            + nu * lap_v[1:-1, 1:-1]
        )
    )

    u_star, v_star = apply_velocity_bc(u_star, v_star)

    # -----------------------------------------------------
    # Step 2. divergence of predictor
    # -----------------------------------------------------
    div_star = compute_divergence(u_star, v_star, dx, dy)
    b = div_star / dt

    # 單個 timestep snapshot
    # 存 RHS（b）一定要在 solve Poisson 之前那一刻
    if n == 10:
        saved_b = b.copy()
        saved_div_star = div_star.copy()
        saved_u_star = u_star.copy()
        saved_v_star = v_star.copy()

    # -----------------------------------------------------
    # Step 3. solve pressure Poisson (with Jacobi, GS, or SOR)
    #     ∇²p = (1/dt) div_star
    # -----------------------------------------------------
    #b = div_star / dt
    p, poisson_res_hist, poisson_iters_used = solve_poisson_gs_until_converged(
    b, dx, dy,
    tol = 1e-4,
    max_iter = 5000,
    return_history = True
    )
    # -----------------------------------------------------
    # Step 4. pressure gradient
    # -----------------------------------------------------
    dpdx, dpdy = compute_pressure_gradient(p, dx, dy)

    # -----------------------------------------------------
    # Step 5. projection
    #     u^{n+1} = u* - dt dp/dx
    #     v^{n+1} = v* - dt dp/dy
    # -----------------------------------------------------
    u_new = u_star.copy()
    v_new = v_star.copy()

    u_new[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * dpdx[1:-1, 1:-1]
    v_new[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * dpdy[1:-1, 1:-1]

    u_new, v_new = apply_velocity_bc(u_new, v_new)

    # -----------------------------------------------------
    # Step 6. check divergence after projection
    # -----------------------------------------------------
    div_new = compute_divergence(u_new, v_new, dx, dy)

    res_final = poisson_res_hist[-1]

    max_div_star = np.max(np.abs(div_star[1:-1, 1:-1]))
    max_div_new  = np.max(np.abs(div_new[1:-1, 1:-1]))

    print(f"\n================ timestep {n} ================")
    print("max|div_star| =", np.max(np.abs(div_star[1:-1, 1:-1])))
    print("max|div_new|  =", np.max(np.abs(div_new[1:-1, 1:-1])))
    print("poisson iterations used =", poisson_iters_used)
    print("poisson final residual =", res_final)
    print("div reduction ratio =", max_div_new / (max_div_star + 1e-12))

    # 先存起來最後再畫
    # 加入 u_n → u_star → u_new
    history.append({
        "step": n,

        "u_n": u.copy(),
        "v_n": v.copy(),

        "adv_u": adv_u.copy(),
        "adv_v": adv_v.copy(),

        "u_star": u_star.copy(),
        "v_star": v_star.copy(),
        "div_star": div_star.copy(),

        "p": p.copy(),

        "u_new": u_new.copy(),
        "v_new": v_new.copy(),

        "div_new": div_new.copy(),

        "poisson_iters_used": poisson_iters_used,

        "poisson_res_hist": poisson_res_hist.copy(),
        "poisson_res_final": poisson_res_hist[-1]
    })

    # March in time
    u = u_new.copy()
    v = v_new.copy()

# 選擇圖片顯示方式
#slider_view_ij_xy(history)
slider_view_8panel(history, dx, dy)

print("\n===== Test 3: GS n_iter sweep on fixed RHS =====")

if saved_b is None:
    print("ERROR: saved_b is None (did you forget to save at n == 10?)")
else:
    saved_max_div_star = np.max(np.abs(saved_div_star[1:-1, 1:-1]))
    print("saved max|div_star| =", saved_max_div_star)

    test_iters = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

    for n_iter in test_iters:
        # ---------------------------------------------
        # Solve Poisson with fixed number of GS sweeps
        # ---------------------------------------------
        p_gs, hist_gs, _ = solve_poisson_gs_until_converged(
            saved_b, dx, dy,
            tol = 1e-4,
            max_iter = 5000,
            return_history = True
        )
        # ---------------------------------------------
        # Pressure gradient
        # ---------------------------------------------
        dpdx_gs, dpdy_gs = compute_pressure_gradient(p_gs, dx, dy)

        # ---------------------------------------------
        # Projection using the SAME saved u_star, v_star
        # ---------------------------------------------
        u_new_gs = saved_u_star.copy()
        v_new_gs = saved_v_star.copy()

        u_new_gs[1:-1, 1:-1] = (
            saved_u_star[1:-1, 1:-1]
            - dt * dpdx_gs[1:-1, 1:-1]
        )

        v_new_gs[1:-1, 1:-1] = (
            saved_v_star[1:-1, 1:-1]
            - dt * dpdy_gs[1:-1, 1:-1]
        )

        u_new_gs, v_new_gs = apply_velocity_bc(u_new_gs, v_new_gs)

        # ---------------------------------------------
        # Divergence after projection
        # ---------------------------------------------
        div_new_gs = compute_divergence(u_new_gs, v_new_gs, dx, dy)

        max_div_new_gs = np.max(np.abs(div_new_gs[1:-1, 1:-1]))
        final_res_gs = hist_gs[-1]
        ratio_gs = max_div_new_gs / (saved_max_div_star + 1e-12)

        # ---------------------------------------------
        # Print
        # ---------------------------------------------
        print(
            f"n_iter = {n_iter:4d} | "
            f"final residual = {final_res_gs:.6e} | "
            f"max|div_new| = {max_div_new_gs:.6e} | "
            f"div reduction ratio = {ratio_gs:.6f}"
        )
