# 2D Incompressible Navier–Stokes Solver (Projection Method)

## Overview

This project implements a 2D incompressible Navier–Stokes solver from scratch using the projection method.
The goal is not only to reproduce a classical lid-driven cavity flow, but to understand the numerical and structural aspects of pressure–velocity coupling.

## Key Components

* Finite difference discretization (2D)
* Projection method for incompressibility enforcement
* Pressure Poisson equation solver
* Staggered (MAC-like) grid formulation
* Diagnostics for divergence and consistency checks

## Motivation

While lid-driven cavity is a classical benchmark problem, this implementation focuses on:

* Understanding the difference between collocated and staggered grids
* Investigating divergence control and numerical consistency
* Debugging pressure–velocity coupling from first principles

## Current Status

* Projection pipeline implemented
* Divergence reduction verified
* Visualization of velocity field and divergence
* Ongoing refinement of boundary treatment and solver stability

## Structure

* `CORE.py` – core numerical routines
* `MAIN.py` – main execution script
* `DIAG.py` – diagnostics and validation
* `PLOTS.py` – visualization

## Notes

This repository reflects an evolving implementation rather than a finalized solver.
The commit history documents the development process, including debugging and design decisions.

## Future Work

* Improve Poisson solver efficiency (GS / SOR)
* Refine boundary conditions
* Extend to passive scalar transport
