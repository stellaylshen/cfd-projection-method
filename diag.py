# Diagnostic functions for staggered_grid.py
import numpy as np

# [DIAG]
def debug_one_cell(u, v, i, j):
    print(f"\n=== Cell ({i},{j}) flux ===")

    u_left  = u[i, j]
    u_right = u[i+1, j]
    v_bot   = v[i, j]
    v_top   = v[i, j+1]

    print(f"u_left  = {u_left}")
    print(f"u_right = {u_right}")
    print(f"v_bot   = {v_bot}")
    print(f"v_top   = {v_top}")

    print("\nFlux contribution:")
    print(f"(u_right - u_left) = {u_right - u_left}")
    print(f"(v_top   - v_bot ) = {v_top - v_bot}")