import jax
import jax.numpy as jnp
from typing import NamedTuple
from physics import euler_step, INITIAL_MASS, DRY_MASS

# Environment
DT = 0.05                  # Time step
MAX_STEPS = 1000           # Maximum duration 50 s
SAFE_Z_VELOCITY = -5.0     # m/s
SAFE_XY_VELOCITY = 2.0     # m/s

class EnvState(NamedTuple):
    physics_state: jnp.ndarray
    time_step: jnp.int32
    done: jnp.bool_

@jax.jit
def get_terrain_elevation(x, y):
    """
    Creates a procedural 3D terrain mimicking the foothills of Mount Sharp inside Gale Crater.
    The center (0,0) is a mountain peak with a flat pad, surrounded by lower crater features.
    """
    mountain = 500.0 * jnp.exp(-(x**2 + y**2) / 10000.0) 
    ridges = 20.0 * jnp.sin(x / 20.0) * jnp.cos(y / 20.0)
    raw_terrain = mountain + ridges
    
    is_pad = (x**2 + y**2) <= 225.0
    
    return jnp.where(is_pad, 500.0, raw_terrain)

@jax.jit
def reset(key: jax.random.PRNGKey, max_stage: jnp.int32) -> EnvState:
    """Initializes the Mars Lander using Multi-Task Curriculum Learning."""

    keys = jax.random.split(key, 6)
    
    stage = jax.random.randint(keys[5], shape=(), minval=0, maxval=max_stage + 1)
    
    z_min_bounds = jnp.array([550.0, 1000.0, 1500.0, 2000.0])
    z_max_bounds = jnp.array([600.0, 1200.0, 2000.0, 2500.0])
    xy_bounds    = jnp.array([10.0, 100.0, 250.0, 400.0])
    vz_min_bounds= jnp.array([-5.0, -30.0, -60.0, -90.0])
    vz_max_bounds= jnp.array([0.0, -10.0, -30.0, -70.0])
    
    z_min = z_min_bounds[stage]
    z_max = z_max_bounds[stage]
    max_radius = xy_bounds[stage]
    vz_min = vz_min_bounds[stage]
    vz_max = vz_max_bounds[stage]
    
    z = jax.random.uniform(keys[2], shape=(1,), minval=z_min, maxval=z_max)
    
    magnitude_xy = jax.random.uniform(keys[0], shape=(2,), minval=0.0, maxval=max_radius)
    signs = jax.random.choice(keys[1], jnp.array([-1.0, 1.0]), shape=(2,))
    x_y = magnitude_xy * signs
    pos = jnp.concatenate([x_y, z])
    
    target_xy = jnp.array([0.0, 0.0])
    direction_xy = target_xy - x_y
    dist_xy = jnp.linalg.norm(direction_xy)
    unit_direction = jnp.where(dist_xy > 0, direction_xy / dist_xy, jnp.zeros(2))
    
    approach_speed = jnp.where(stage == 0, 0.0, jax.random.uniform(keys[3], shape=(1,), minval=10.0, maxval=30.0))
    vx_vy = unit_direction * approach_speed
    
    vz = jax.random.uniform(keys[4], shape=(1,), minval=vz_min, maxval=vz_max)
    vel = jnp.concatenate([vx_vy, vz])
    
    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega = jnp.array([0.0, 0.0, 0.0])
    mass = jnp.array([INITIAL_MASS])
    
    physics_state = jnp.concatenate([pos, vel, q, omega, mass])
    
    return EnvState(physics_state=physics_state, time_step=jnp.int32(0), done=jnp.bool_(False))

# Reward function
@jax.jit
def calculate_reward(state: jnp.ndarray, action: jnp.ndarray, done: bool):
    """Calculates the dense reward and terminal conditions."""
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]
    
    z = pos[2]
    vz = vel[2]
    
    # Continuous Rewards
    target_xy = jnp.array([0.0, 0.0])
    xy_dist = jnp.linalg.norm(pos[0:2] - target_xy)
    z_dist = jnp.abs(pos[2] - 500.0)
    
    xy_penalty = -0.01 * xy_dist
    z_penalty = -0.002 * z_dist
    distance_penalty = xy_penalty + z_penalty

    velocity_penalty = -0.05 * jnp.linalg.norm(vel)
    upright_penalty = -2.0 * (1.0 - q[0]) 
    spin_penalty = -0.1 * jnp.linalg.norm(omega)
    throttle_penalty = -0.01 * jnp.sum(action)
    
    step_reward = distance_penalty + velocity_penalty + upright_penalty + spin_penalty + throttle_penalty
    
    # Terminal Rewards
    ground_z = get_terrain_elevation(pos[0], pos[1])
    is_grounded = z <= ground_z
    
    safe_impact = (vz >= SAFE_Z_VELOCITY) & (jnp.linalg.norm(vel[0:2]) <= SAFE_XY_VELOCITY)
    upright_impact = q[0] > 0.95
    
    successful_landing = is_grounded & safe_impact & upright_impact
    crash = is_grounded & ~(safe_impact & upright_impact)

    impact_speed = jnp.linalg.norm(vel)
    crash_shaped_penalty = -200.0 - (xy_dist * 2.0) - (impact_speed * 100.0)

    terminal_reward = jnp.where(successful_landing, 5000.0, 0.0)
    terminal_reward = jnp.where(crash, crash_shaped_penalty, terminal_reward)

    flew_away = z > 3000.0
    terminal_reward = jnp.where(flew_away, -10000.0, terminal_reward)
    
    fuel_empty = mass <= DRY_MASS
    step_reward = jnp.where(fuel_empty & ~is_grounded, step_reward - 100.0, step_reward)
    
    total_reward = jnp.where(is_grounded | flew_away, terminal_reward, step_reward)
    
    return total_reward / 1000.0

# Environment step
@jax.jit
def step(env_state: EnvState, action: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advances the simulation by one time step."""
    clipped_action = jnp.clip(action, 0.0, 1.0)
    
    next_physics_state = euler_step(env_state.physics_state, clipped_action, DT)
    
    pos = next_physics_state[0:3]
    z = pos[2]
    mass = next_physics_state[13]
    
    ground_z = get_terrain_elevation(pos[0], pos[1])
    hit_ground = z <= ground_z

    out_of_time = env_state.time_step >= MAX_STEPS
    out_of_fuel = mass <= DRY_MASS
    flew_away = z > 3000.0
    
    done = hit_ground | out_of_time | flew_away | env_state.done
    
    reward = calculate_reward(next_physics_state, clipped_action, done)
    
    next_env_state = EnvState(
        physics_state=next_physics_state,
        time_step=env_state.time_step + 1,
        done=done
    )
    
    return next_env_state, next_physics_state, reward, done