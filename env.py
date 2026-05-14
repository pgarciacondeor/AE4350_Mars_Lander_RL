import jax
import jax.numpy as jnp
from typing import NamedTuple
from physics import euler_step, INITIAL_MASS, DRY_MASS

# Environment
DT = 0.05                  # Time step
MAX_STEPS = 1000           # Maximum duration 50 s
SAFE_Z_VELOCITY = -2.5     # m/s
SAFE_XY_VELOCITY = 1.0     # m/s

class EnvState(NamedTuple):
    physics_state: jnp.ndarray
    time_step: jnp.int32
    done: jnp.bool_

@jax.jit
def reset(key: jax.random.PRNGKey) -> EnvState:
    """Initializes the Mars Lander at the start of the powered descent."""
    keys = jax.random.split(key, 4)
    
    # Randomize initial altitude and lateral drift
    x_y = jax.random.uniform(keys[0], shape=(2,), minval=-50.0, maxval=50.0)
    z = jax.random.uniform(keys[1], shape=(1,), minval=1500.0, maxval=2000.0)
    pos = jnp.concatenate([x_y, z])
    
    # Randomize initial downward velocity
    vx_vy = jax.random.uniform(keys[2], shape=(2,), minval=-5.0, maxval=5.0)
    vz = jax.random.uniform(keys[3], shape=(1,), minval=-100.0, maxval=-80.0)
    vel = jnp.concatenate([vx_vy, vz])
    
    # Upright orientation and zero angular velocity
    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega = jnp.array([0.0, 0.0, 0.0])
    
    mass = jnp.array([INITIAL_MASS])
    
    physics_state = jnp.concatenate([pos, vel, q, omega, mass])
    
    return EnvState(
        physics_state=physics_state,
        time_step=jnp.int32(0),
        done=jnp.bool_(False)
    )

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

    # Distance from target landing pad
    distance_penalty = -0.01 * jnp.linalg.norm(pos)
    
    # High velocities
    velocity_penalty = -0.05 * jnp.linalg.norm(vel)
    
    # Tilting
    upright_penalty = -1.0 * (1.0 - q[0])
    
    # Spinning
    spin_penalty = -0.1 * jnp.linalg.norm(omega)
    
    # Fuel efficiency
    throttle_penalty = -0.01 * jnp.sum(action)
    
    step_reward = distance_penalty + velocity_penalty + upright_penalty + spin_penalty + throttle_penalty
    
    # Terminal Rewards

    is_grounded = z <= 0.0
    
    # Landing logic - grounded, slow, and upright
    safe_impact = (vz >= SAFE_Z_VELOCITY) & (jnp.linalg.norm(vel[0:2]) <= SAFE_XY_VELOCITY)
    upright_impact = q[0] > 0.95
    
    successful_landing = is_grounded & safe_impact & upright_impact
    crash = is_grounded & ~(safe_impact & upright_impact)

    terminal_reward = jnp.where(successful_landing, 1000.0, 0.0)
    terminal_reward = jnp.where(crash, -1000.0, terminal_reward)
    
    # Out of fuel penalty
    fuel_empty = mass <= DRY_MASS
    terminal_reward = jnp.where(fuel_empty & ~is_grounded, -500.0, terminal_reward)
    
    total_reward = jnp.where(is_grounded | fuel_empty, terminal_reward, step_reward)
    
    return total_reward

# Environment step
@jax.jit
def step(env_state: EnvState, action: jnp.ndarray) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advances the simulation by one time step."""
    
    clipped_action = jnp.clip(action, 0.0, 1.0)
    
    next_physics_state = euler_step(env_state.physics_state, clipped_action, DT)
    
    z = next_physics_state[2]
    mass = next_physics_state[13]
    
    hit_ground = z <= 0.0
    out_of_time = env_state.time_step >= MAX_STEPS
    out_of_fuel = mass <= DRY_MASS
    flew_away = z > 3000.0
    
    done = hit_ground | out_of_time | out_of_fuel | flew_away | env_state.done
    
    reward = calculate_reward(next_physics_state, clipped_action, done)
    
    next_env_state = EnvState(
        physics_state=next_physics_state,
        time_step=env_state.time_step + 1,
        done=done
    )
    
    return next_env_state, next_physics_state, reward, done