# this part solves the model step by step, uses the rates of change to actually move the projectile forward in time
# it expects a function of this form:
# y' = f(t, y)
# in our project:
# y = [x, y, vx, vy] and f(t, y) = projectile_with_drag(t, y, k)

# so the solver repeatedly does:
# current time and state
# → calculate derivatives using projectile_with_drag()
# → estimate next state
# → repeat until the projectile lands

import numpy as np


def euler_solver(f, t0, t_end, y0, h, stop_condition=None):
    """
    Solve y' = f(t, y) using the Euler method.

    Optional stop_condition:
        A function stop_condition(t, y) returning True when simulation should stop.

    Example of passing down the arguments:

    f = projectile_with_drag
    t0 = 0
    t_end = 10
    y0 = [0, 2, vx0, vy0]
    h = 0.01 or another step size
    stop_condition = has_hit_ground
    """
    n_steps = int((t_end - t0) / h)             # maximum number of steps.
    t_values = [t0]                             # list of time values.

    y0 = np.array(y0, dtype=float)              # convert the initial state into a NumPy array
    #if y0.ndim == 0:
    #   y0 = np.array([y0], dtype=float)

    y_values = [y0]                             # stores the projectile states.

    n_function_evals = 0                        # counts how many times the solver calls the ODE function f (calculate x', y', vx', vy' which includes recalculating speed and drag.)

    t = t0
    y = y0.copy()                               # this creates working variables, copy to not modify original initial states

    # now euler loop itself
    
    for _ in range(n_steps):                    # solver simply repeats up to n_steps times.
        slope = np.array(f(t, y), dtype=float)  # calls thte ODE function, so projectile_with_drag(t, y, k) which returns derivatives [x', y', vx', vy'].
        y_next = y + h * slope                  # euler formula, calculatin next step
        t_next = t + h

        n_function_evals += 1                   # function called so counter goes up

        t_values.append(t_next)                 
        y_values.append(y_next.copy())          # these store the new time and new state

        t = t_next
        y = y_next                              # this moves the simulation forward, the next iteration now starts from the new time and new state.

        if stop_condition is not None and stop_condition(t, y): #check condition whether to stop
            break

    return {
        "method": "Euler",
        "t": np.array(t_values),
        "y": np.array(y_values),
        "step_size": h,
        "n_steps": len(t_values) - 1,
        "n_function_evals": n_function_evals,
    }


def rk2_solver(f, t0, t_end, y0, h, stop_condition=None):
    """
    Solve y' = f(t, y) using the RK2 midpoint method.

    Example of passing down the arguments:

    f = projectile_with_drag
    t0 = 0
    t_end = 10
    y0 = [0, 2, vx0, vy0]
    h = 0.01 or another step size
    stop_condition = has_hit_ground
    """
    n_steps = int((t_end - t0) / h)             # maximum number of steps.
    t_values = [t0]                             # list of time values.

    y0 = np.array(y0, dtype=float)              # convert the initial state into a NumPy array
    #if y0.ndim == 0:
    #   y0 = np.array([y0], dtype=float)

    y_values = [y0]                             # stores the projectile states.

    n_function_evals = 0                        # counts how many times the solver calls the ODE function f (calculate x', y', vx', vy' which includes recalculating speed and drag.)

    t = t0
    y = y0.copy()                               # this creates working variables, copy to not modify original initial states

    # now rk2 loop itself

    for _ in range(n_steps):
        k1 = np.array(f(t, y), dtype=float)     # calculates the first slope at the current point (current projectile state)
        k2 = np.array(f(t + h / 2, y + h * k1 / 2), dtype=float) # midpoint slope, halfway through the time step.

        y_next = y + h * k2                     # rk2 formula, updates the state using the midpoint slope
        t_next = t + h

        n_function_evals += 2                   # RK2 calls the ODE function twice per step

        t_values.append(t_next)                 
        y_values.append(y_next.copy())          # these store the new time and new state

        t = t_next
        y = y_next                              # this moves the simulation forward, the next iteration now starts from the new time and new state.

        if stop_condition is not None and stop_condition(t, y): #check condition whether to stop
            break

    return {
        "method": "RK2",
        "t": np.array(t_values),
        "y": np.array(y_values),
        "step_size": h,
        "n_steps": len(t_values) - 1,
        "n_function_evals": n_function_evals,
    }


def rk4_solver(f, t0, t_end, y0, h, stop_condition=None):
    """
    Solve y' = f(t, y) using the classical RK4 method.

    Example of passing down the arguments:

    f = projectile_with_drag
    t0 = 0
    t_end = 10
    y0 = [0, 2, vx0, vy0]
    h = 0.01 or another step size
    stop_condition = has_hit_ground
    """
    n_steps = int((t_end - t0) / h)             # maximum number of steps.
    t_values = [t0]                             # list of time values.

    y0 = np.array(y0, dtype=float)              # convert the initial state into a NumPy array
    #if y0.ndim == 0:
    #   y0 = np.array([y0], dtype=float)

    y_values = [y0]                             # stores the projectile states.

    n_function_evals = 0                        # counts how many times the solver calls the ODE function f (calculate x', y', vx', vy' which includes recalculating speed and drag.)

    t = t0
    y = y0.copy()                               # this creates working variables, copy to not modify original initial states

    # now rk4 loop itself

    for _ in range(n_steps):
        k1 = np.array(f(t, y), dtype=float)     # calculates the first slope at the current point (current projectile state)
        k2 = np.array(f(t + h / 2, y + h * k1 / 2), dtype=float) # midpoint slope, halfway through the time step.
        k3 = np.array(f(t + h / 2, y + h * k2 / 2), dtype=float) # another midpoint slope using k2
        k4 = np.array(f(t + h, y + h * k3), dtype=float) # slope at the end of the step.

        y_next = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4) # rk4 formula, combines all four slopes and scales by h / 6
        t_next = t + h

        n_function_evals += 4 # rk4 calls function 4 times

        t_values.append(t_next)                 
        y_values.append(y_next.copy())          # these store the new time and new state

        t = t_next
        y = y_next                              # this moves the simulation forward, the next iteration now starts from the new time and new state.

        if stop_condition is not None and stop_condition(t, y): #check condition whether to stop
            break

    return {
        "method": "RK4",
        "t": np.array(t_values),
        "y": np.array(y_values),
        "step_size": h,
        "n_steps": len(t_values) - 1,
        "n_function_evals": n_function_evals,
    }