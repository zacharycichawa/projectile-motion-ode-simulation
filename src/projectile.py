# this part defines the physical model part of the project

import numpy as np


GRAVITY = 9.81
AIR_DENSITY = 1.225 # sea level


OBJECTS = {
    "tennis_ball": {
        "name": "Tennis Ball",
        "mass": 0.057,          # kg,
        "radius": 0.0335,       # m
        "drag_coefficient": 0.55
    },
    "baseball": {
        "name": "Baseball",
        "mass": 0.145,          # kg
        "radius": 0.0366,       # m
        "drag_coefficient": 0.35
    },
    "steel_ball": {
        "name": "Steel Ball",
        "mass": 0.5,            # kg
        "radius": 0.025,        # m
        "drag_coefficient": 0.47
    }
}


def cross_sectional_area(radius):
    """
    Cross-sectional area of a sphere.
    """
    return np.pi * radius ** 2


def drag_parameter(mass, radius, drag_coefficient, air_density=AIR_DENSITY):
    """
    Compute the drag parameter:

        k = (rho * Cd * A) / (2m)

    where:
        rho = air density
        Cd = drag coefficient
        A = cross-sectional area
        m = mass

    (F_drag = 0.5 * rho * Cd * A * speed^2, a = F / m)

    """
    area = cross_sectional_area(radius)
    return (air_density * drag_coefficient * area) / (2 * mass)


def create_initial_state(speed, angle_degrees, initial_height=1.5):
    """
    Create the initial state vector:

        [x, y, vx, vy]

    from launch speed, launch angle, and initial height.
    """
    angle_radians = np.deg2rad(angle_degrees) #numpy sin and cos functions use radians, not degrees

    x0 = 0.0
    y0 = initial_height
    vx0 = speed * np.cos(angle_radians)
    vy0 = speed * np.sin(angle_radians)

    return np.array([x0, y0, vx0, vy0], dtype=float)


def projectile_with_drag(t, state, k, gravity=GRAVITY):
    """
    Projectile motion with quadratic air resistance.
    This only calculates the current rates of change, which the solvers receive to estimate next state. 

    State:
        state[0] = x position
        state[1] = y position / height
        state[2] = x velocity
        state[3] = y velocity

    Equations:
        x'  = vx
        y'  = vy
        vx' = -k * speed * vx
        vy' = -g - k * speed * vy
    """
    x = state[0] #unpacks the state vector
    y = state[1]
    vx = state[2]
    vy = state[3]

    speed = np.sqrt(vx ** 2 + vy ** 2)

    dxdt = vx
    dydt = vy
    dvxdt = -k * speed * vx
    dvydt = -gravity - k * speed * vy

    return np.array([dxdt, dydt, dvxdt, dvydt], dtype=float)


def has_hit_ground(t, state):
    """
    Stop condition for projectile motion.
    Returns True when the projectile reaches or goes below ground level.
    """
    return state[1] <= 0.0 # y <= 0


def get_object_config(object_key):
    """
    Return physical parameters for a selected object.
    """
    if object_key not in OBJECTS:
        raise ValueError(f"Unknown object '{object_key}'. Use one of: {list(OBJECTS.keys())}")

    obj = OBJECTS[object_key].copy()

    obj["area"] = cross_sectional_area(obj["radius"])
    obj["k"] = drag_parameter(
        mass=obj["mass"],
        radius=obj["radius"],
        drag_coefficient=obj["drag_coefficient"]
    )

    return obj


def summarize_trajectory(result):
    """
    Calculate physical summary statistics from a projectile simulation.

    If the final point is below ground level, linearly interpolate between
    the last two points to estimate the landing time and range more accurately.
    """
    t = result["t"]
    y = result["y"]
    # these extract the time array and the trajectory array
    #[0.00, 0.01, 0.02, ...]
    #[x, y, vx, vy]

    x_position = y[:, 0] #all rows, column 0 (all horizontal positions.)
    height = y[:, 1]
    vx = y[:, 2]
    vy = y[:, 3]

    max_height = np.max(height)

    # Default values use the final stored point
    landing_range = x_position[-1] #[-1] means the final value.     
    flight_time = t[-1]
    final_vx = vx[-1]
    final_vy = vy[-1]

    # If the last point is below ground, interpolate landing point
    if len(t) >= 2 and height[-1] < 0:
        h1 = height[-2]
        h2 = height[-1]

        # fraction between previous and final point where height becomes zero
        alpha = h1 / (h1 - h2)

        landing_range = x_position[-2] + alpha * (x_position[-1] - x_position[-2])
        flight_time = t[-2] + alpha * (t[-1] - t[-2])
        final_vx = vx[-2] + alpha * (vx[-1] - vx[-2])
        final_vy = vy[-2] + alpha * (vy[-1] - vy[-2])

    final_speed = np.sqrt(final_vx ** 2 + final_vy ** 2)

    return {
        "range": landing_range,
        "max_height": max_height,
        "flight_time": flight_time,
        "final_speed": final_speed,
    }