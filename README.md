# Projectile Motion ODE Simulation

This project simulates projectile motion with quadratic air resistance using numerical methods for ordinary differential equations (ODEs).

The model treats the projectile as a spherical object moving in two dimensions. Its motion is affected by gravity and air resistance. Since drag depends on the current speed of the object, the acceleration changes throughout the flight. Because of this, the trajectory is calculated step by step using numerical ODE solvers.

## Project goals

The project has two main goals:

1. Compare how different spherical objects behave under the same launch conditions.
2. Compare how accurately Euler, RK2, and RK4 solve the same projectile motion problem.

## Numerical methods used

The project uses three numerical ODE methods:

- Euler method
- RK2 midpoint method
- RK4 method

A fine-step RK4 solution is used as the reference solution for the error analysis.

## Physical model

The projectile state is represented as:

```text
[x, y, vx, vy]
```

where:

```text
- `x` is the horizontal position,
- `y` is the height,
- `vx` is the horizontal velocity,
- `vy` is the vertical velocity.
```

The system of equations is:

```text
x'  = vx
y'  = vy
vx' = -k * speed * vx
vy' = -g - k * speed * vy
```

where:

```text
speed = sqrt(vx^2 + vy^2)
```

The drag parameter is:

```text
k = (rho * Cd * A) / (2m)
```

For a spherical object:

```text
A = pi * r^2
```

## Project structure

```text
projectile-motion-ode-simulation/
├── notebooks/
│   └── 01.ipynb
├── src/
│   ├── projectile.py
│   └── solvers.py
├── requirements.txt
└── README.md
```

## What the notebook does

The notebook compares the motion of a tennis ball, baseball, and steel ball using the same launch conditions. It also compares Euler, RK2, and RK4 using different step sizes.

The results include trajectory plots, speed plots, range comparison, numerical error tables, and error plots.

## Main results

The tennis ball is affected most strongly by air resistance and has the shortest range. The steel ball is affected least and travels the farthest.

Euler produces the largest numerical errors. RK2 and RK4 are much closer to the fine-step RK4 reference solution, especially as the step size becomes smaller.

## Requirements

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
