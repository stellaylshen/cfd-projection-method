# 2D Incompressible Navier–Stokes Solver (Projection Method)

## Overview

This project implements a 2D incompressible Navier–Stokes solver from scratch using a finite-difference projection method.

The primary goal is not only to reproduce the classical lid-driven cavity benchmark, but also to investigate the numerical structure of incompressible flow solvers, particularly pressure–velocity coupling, incompressibility enforcement, and staggered-grid discretization.

The implementation focuses on understanding the numerical behavior of the solver rather than treating CFD as a black-box simulation pipeline.

## Features

- Finite-difference discretization (2D)
- Projection method for incompressibility enforcement
- Pressure Poisson equation solver
- Staggered (MAC-like) grid formulation
- Divergence diagnostics and consistency checks
- Lid-driven cavity benchmark validation
- Jacobi and SOR iterative solvers

## Current Status

- Projection pipeline implemented
- Divergence reduction verified
- MAC staggered-grid formulation implemented
- Velocity-field and divergence visualization completed
- Preliminary benchmark comparison against cavity-flow reference data
- Ongoing refinement of:
  - boundary-condition consistency
  - pressure projection
  - solver stability

## Numerical Challenges Explored

This repository documents several intermediate implementation and debugging stages, including:

- Checkerboard pressure artifacts in collocated grids
- Pressure–velocity decoupling
- Divergence persistence after projection
- Neumann pressure gauge fixing
- Jacobi vs. SOR convergence behavior
- Boundary-condition consistency on staggered grids

## Repository Structure

```text
run_projection_solver.py
Main driver used to reproduce the final Re=100 cavity simulation and report figures.

grid_convergence_study.py
Runs grid-refinement cases and generates convergence data.

test_operator_consistency.py
Standalone consistency checks for divergence, gradient, projection, and ghost-cell operators.

experiments/
Exploratory scripts used during development, including collocated-grid pathology and archived solver variants.
```

## Notes
This repository reflects an evolving numerical implementation rather than a finalized CFD solver.

The commit history documents the development process, including debugging stages, numerical experiments, and design decisions made throughout the implementation.

## Future Work
- Improve Poisson solver efficiency (GS / SOR / multigrid)
- Refine boundary-condition treatment
- Improve convergence behavior
- Add higher-order advection schemes
- Extend to passive scalar transport

## Technologies
- Python
- NumPy
- Matplotlib