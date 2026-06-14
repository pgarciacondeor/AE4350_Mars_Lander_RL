import jax
import jax.numpy as jnp

# Environmental Constants
G_MARS = jnp.array([0.0, 0.0, -3.721])  # m/s^2
G0 = 9.80665                            # m/s^2 
H_SCALE = 11100.0                       # m 
RHO_0 = 0.020                           # kg/m^3 -> Mars Fact Sheet NASA

# Vehicle Characteristics 
INITIAL_MASS = 2400.0                   # kg
DRY_MASS = 1500.0                       # kg
CD = 1.05                                # Drag coefficient
AREA = 8.4                              # m^2

# Propulsion System
T_MAX = 2600                          # N (Max thrust per engine)
ISP = 225.0                             # s

# Lander Inertia - rectangular box model
I_MAT = jnp.diag(jnp.array([2000.0, 2000.0, 3000.0])) 
I_INV = jnp.linalg.inv(I_MAT)

# 8 engines facing downwards with 15 degree angle
R = 1.5
SLANT_ANGLE = jnp.radians(15.0)
sin_c = jnp.sin(SLANT_ANGLE)
cos_c = jnp.cos(SLANT_ANGLE)

# Positions of the 8 engines
ENGINE_POSITIONS = jnp.array([
    [ R,  R, 0], [ R,  R, 0],  
    [-R,  R, 0], [-R,  R, 0], 
    [-R, -R, 0], [-R, -R, 0],  
    [ R, -R, 0], [ R, -R, 0]   
])

# Thrust direction unit vectors
ENGINE_DIRS = jnp.array([
    [ sin_c,  sin_c, 1.0], [ sin_c,  sin_c, 1.0], 
    [-sin_c,  sin_c, 1.0], [-sin_c,  sin_c, 1.0],
    [-sin_c, -sin_c, 1.0], [-sin_c, -sin_c, 1.0],
    [ sin_c, -sin_c, 1.0], [ sin_c, -sin_c, 1.0]
])

ENGINE_DIRS = ENGINE_DIRS / jnp.linalg.norm(ENGINE_DIRS, axis=1, keepdims=True)


@jax.jit
def calc_density(altitude):
    """Exponential atmosphere model."""
    h = jnp.maximum(altitude, 0.0)
    return RHO_0 * jnp.exp(-h / H_SCALE)

@jax.jit
def quaternion_rotate(q, v):
    """Rotates a vector v by quaternion q (Body to Inertial frame)."""
    q0, q1, q2, q3 = q
    
    R = jnp.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),   1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1),   1 - 2*(q1**2 + q2**2)]
    ])

    return jnp.dot(R, v)

@jax.jit
def calc_dynamics(state, action, wind_accel=jnp.zeros(3), gravity_scale=1.0, drag_scale=1.0):
    """
    Calculates the derivatives of the state given current actions.
    State: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, mass]
    Action: [u1, ..., u8] (Throttle values 0.0 to 1.0)
    wind_accel: external horizontal wind-gust acceleration [ax, ay, 0] (m/s^2).
    gravity_scale, drag_scale: multipliers for post-training robustness tests.
    Both 1.0 = nominal Mars conditions used during training.
    """
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]

    # Mass depletion (Re-Entry Systems - Fundamentals of Motion)
    thrust_magnitudes = action * T_MAX
    total_thrust_scalar = jnp.sum(thrust_magnitudes)
    mdot = -total_thrust_scalar / (ISP * G0)

    # Calculate forces and torques in body frame
    forces_body = ENGINE_DIRS * thrust_magnitudes[:, None]
    total_thrust_body = jnp.sum(forces_body, axis=0)
    
    torques_body = jnp.cross(ENGINE_POSITIONS, forces_body)
    total_torque_body = jnp.sum(torques_body, axis=0)

    # Transform thrust to Inertial frame
    thrust_inertial = quaternion_rotate(q, total_thrust_body)

    # Drag
    speed = jnp.linalg.norm(vel)
    rho = calc_density(pos[2])
    
    v_hat = jnp.where(speed > 1e-6, vel / speed, jnp.zeros_like(vel))
    drag_inertial = -0.5 * rho * (speed**2) * CD * AREA * v_hat * drag_scale

    # Translational acceleration (gravity + thrust + drag + wind-gust disturbance)
    accel = (thrust_inertial + drag_inertial) / mass + G_MARS * gravity_scale + wind_accel

    # Rotational acceleration
    omega_dot = jnp.dot(I_INV, (total_torque_body - jnp.cross(omega, jnp.dot(I_MAT, omega))))

    # Quaternion derivative
    q0, q1, q2, q3 = q
    wx, wy, wz = omega
    q_dot = 0.5 * jnp.array([
        -q1*wx - q2*wy - q3*wz,
         q0*wx - q3*wy + q2*wz,
         q3*wx + q0*wy - q1*wz,
        -q2*wx + q1*wy + q0*wz
    ])

    state_dot = jnp.concatenate([vel, accel, q_dot, omega_dot, jnp.array([mdot])])

    return state_dot

@jax.jit
def euler_step(state, action, dt, wind_accel=jnp.zeros(3), gravity_scale=1.0, drag_scale=1.0):
    """Simple Euler integration step."""
    state_dot = calc_dynamics(state, action, wind_accel, gravity_scale, drag_scale)
    new_state = state + state_dot * dt
    
    q_new = new_state[6:10]
    q_new = q_new / jnp.linalg.norm(q_new)

    mass_new = jnp.maximum(new_state[13], DRY_MASS)
    
    return jnp.concatenate([new_state[0:6], q_new, new_state[10:13], jnp.array([mass_new])])