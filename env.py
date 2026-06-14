import os
import jax
import jax.numpy as jnp
from typing import NamedTuple
from physics import euler_step, INITIAL_MASS, DRY_MASS

# ablation knobs
ENVELOPE_COEF = float(os.environ.get("MARS_ENVELOPE_COEF", 0.1))   # braking-envelope penalty
LATERAL_COEF = float(os.environ.get("MARS_LATERAL_COEF", 3.0))     # over-pad lateral-velocity shaping
GUST_SCALE = float(os.environ.get("MARS_GUST_SCALE", 1.0))         # wind-gust strength multiplier
A_BRAKE = 3.5   # max net deceleration

# Environment
DT = 0.05                  # Time step
MAX_STEPS_BY_STAGE = jnp.array([600, 900, 1200, 1500])   # = 30 / 45 / 60 / 75 s
MAX_STEPS = 1500           # upper bound on episode
SAFE_Z_VELOCITY = -2.0     # m/s
SAFE_XY_VELOCITY = 2.0     # m/s
MAX_ALTITUDE = 3000.0      # m, above this counts as "flew away"
MAX_HORIZ = 600.0          # m, drifting past this counts as out of bounds
PAD_RADIUS = 30.0          # m, radius of the flat circular landing pad

class EnvState(NamedTuple):
    physics_state: jnp.ndarray
    time_step: jnp.int32
    done: jnp.bool_
    gust: jnp.ndarray            # [base_x, base_y, amp_x, amp_y, omega, phase]
    stage: jnp.int32             # curriculum stage this episode was sampled from

# per-stage wind-gust strength
GUST_BASE_MAX = jnp.array([0.0, 0.4, 0.8, 1.2]) * GUST_SCALE   # constant component
GUST_AMP_MAX  = jnp.array([0.0, 0.4, 0.8, 1.2]) * GUST_SCALE   # oscillating amplitude

# generate terrain
@jax.jit
def get_terrain_elevation(x, y):
    mountain = 500.0 * jnp.exp(-(x**2 + y**2) / 120000.0)
    ridges = 8.0 * jnp.sin(x / 60.0) * jnp.cos(y / 60.0)
    raw_terrain = mountain + ridges

    is_pad = (x**2 + y**2) <= PAD_RADIUS**2

    return jnp.where(is_pad, 500.0, raw_terrain)

# reset an episode
@jax.jit
def reset(key: jax.random.PRNGKey, max_stage: jnp.int32, fixed_stage: jnp.int32 = -1, stage_weights=None) -> EnvState:
    keys = jax.random.split(key, 12)

    # sample a stage from adaptive weights, else uniform over [0, max_stage]
    all_stages = jnp.arange(4)
    if stage_weights is None:
        weights = jnp.where(all_stages <= max_stage, 1.0, 0.0)
    else:
        weights = stage_weights
    weights = weights / jnp.sum(weights)
    sampled_stage = jax.random.choice(keys[5], all_stages, p=weights)
    stage = jnp.where(fixed_stage >= 0, jnp.int32(fixed_stage), sampled_stage)
    
    z_min_bounds = jnp.array([520.0, 800.0,  1500.0, 2000.0])
    z_max_bounds = jnp.array([560.0, 1200.0, 2000.0, 2500.0])
    vz_min_bounds= jnp.array([-8.0,  -20.0,  -60.0,  -85.0])
    vz_max_bounds= jnp.array([-3.0,  -5.0,   -30.0,  -75.0])
    xy_bounds    = jnp.array([3.0,   100.0,  250.0,   400.0])
    
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

    # Sample this episode's wind-gust profile, scaled by stage
    base_max = GUST_BASE_MAX[stage]
    amp_max  = GUST_AMP_MAX[stage]
    base_x = jax.random.uniform(keys[6],  shape=(), minval=-base_max, maxval=base_max)
    base_y = jax.random.uniform(keys[7],  shape=(), minval=-base_max, maxval=base_max)
    amp_x  = jax.random.uniform(keys[8],  shape=(), minval=0.0, maxval=amp_max)
    amp_y  = jax.random.uniform(keys[9],  shape=(), minval=0.0, maxval=amp_max)
    omega  = jax.random.uniform(keys[10], shape=(), minval=0.2, maxval=1.0)
    phase  = jax.random.uniform(keys[11], shape=(), minval=0.0, maxval=2.0 * jnp.pi)
    gust = jnp.array([base_x, base_y, amp_x, amp_y, omega, phase])

    return EnvState(physics_state=physics_state, time_step=jnp.int32(0), done=jnp.bool_(False), gust=gust, stage=jnp.int32(stage))


# horizontal wind-gust acceleration at current time
@jax.jit
def current_wind(env_state: EnvState):

    g = env_state.gust
    t = env_state.time_step.astype(jnp.float32) * DT
    wx = g[0] + g[2] * jnp.sin(g[4] * t + g[5])
    wy = g[1] + g[3] * jnp.sin(g[4] * t + g[5] + jnp.pi / 2.0)

    return jnp.stack([wx, wy])


# network input
@jax.jit
def observe(env_state: EnvState):
    return jnp.concatenate([env_state.physics_state, current_wind(env_state)])

# shaping potential
@jax.jit
def _potential(state):

    pos = state[0:3]
    vel = state[3:6]

    altitude_above_pad = jnp.maximum(pos[2] - 500.0, 0.0)
    horiz_dist = jnp.linalg.norm(pos[0:2])
    speed = jnp.linalg.norm(vel)

    # null lateral velocity once over the pad (determined by distance). altitude makes descending raise the penalty
    v_xy = jnp.linalg.norm(vel[0:2])
    near_pad = jnp.exp(-horiz_dist / 30.0)

    return -(1.0 * altitude_above_pad + 1.0 * horiz_dist + 2.0 * speed + LATERAL_COEF * near_pad * v_xy)


# Reward function
@jax.jit
def calculate_reward(prev_state, state, action, newly_grounded, newly_terminated):
    pos = state[0:3]
    vel = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    mass = state[13]

    z = pos[2]
    vz = vel[2]

    shaping = _potential(state) - _potential(prev_state)

    # penalise descending faster than we could still brake before the pad
    altitude_above_pad = jnp.maximum(z - 500.0, 0.0)
    v_safe = jnp.sqrt(2.0 * A_BRAKE * altitude_above_pad)
    overspeed = jnp.clip(-vz - v_safe, 0.0, None)          # descent rate beyond the envelope
    envelope_penalty = -ENVELOPE_COEF * overspeed

    time_penalty     = -0.05
    upright_penalty  = -0.2  * (1.0 - q[0])
    spin_penalty     = -0.05 * jnp.linalg.norm(omega)
    throttle_penalty = -0.002 * jnp.sum(action)

    step_reward = (shaping + envelope_penalty + time_penalty + upright_penalty + spin_penalty + throttle_penalty)

    fuel_empty = mass <= DRY_MASS
    step_reward = jnp.where(fuel_empty & (z > 500.0), step_reward - 1.0, step_reward)

    # terminal rewards
    vz_safe      = vz >= SAFE_Z_VELOCITY
    vxy_safe     = jnp.linalg.norm(vel[0:2]) <= SAFE_XY_VELOCITY
    upright_ok   = q[0] > 0.95

    impact_speed = jnp.linalg.norm(vel)
    xy_dist      = jnp.linalg.norm(pos[0:2])
    on_pad       = xy_dist < PAD_RADIUS

    success      = vz_safe & vxy_safe & upright_ok & on_pad
    crash_penalty = -50.0 - (impact_speed * 2.0) - (xy_dist * 0.1)

    # push toward softer landings
    soft_bonus = 100.0 * jnp.exp(-impact_speed / 1.5)
    ground_terminal = jnp.where(success, 150.0 + soft_bonus, crash_penalty)

    timeout_penalty   = -50.0 - (xy_dist * 0.1) - (impact_speed * 1.0)
    flew_away_penalty = -100.0

    out_of_bounds = (z > MAX_ALTITUDE) | (xy_dist > MAX_HORIZ)
    non_ground_terminal = jnp.where(out_of_bounds, flew_away_penalty, timeout_penalty)

    terminal_reward = jnp.where(newly_grounded, ground_terminal, non_ground_terminal)

    is_terminal = newly_grounded | newly_terminated

    return jnp.where(is_terminal, terminal_reward, step_reward)

# Environment step
@jax.jit
def step(env_state: EnvState, action: jnp.ndarray, gravity_scale=1.0, drag_scale=1.0):
    clipped_action = jnp.clip(action, 0.0, 1.0)

    wind = current_wind(env_state)
    wind_accel = jnp.array([wind[0], wind[1], 0.0])
    next_physics_state = euler_step(env_state.physics_state, clipped_action, DT,
                                    wind_accel, gravity_scale, drag_scale)

    pos  = next_physics_state[0:3]
    z    = pos[2]
    mass = next_physics_state[13]

    horiz_dist = jnp.linalg.norm(pos[0:2])
    ground_z   = get_terrain_elevation(pos[0], pos[1])
    hit_ground = z <= ground_z
    out_of_time = env_state.time_step >= MAX_STEPS_BY_STAGE[env_state.stage]
    out_of_fuel = mass <= DRY_MASS
    flew_away   = (z > MAX_ALTITUDE) | (horiz_dist > MAX_HORIZ)

    was_done = env_state.done

    newly_grounded    = hit_ground   & ~was_done
    newly_terminated  = (out_of_time | out_of_fuel | flew_away) & ~hit_ground & ~was_done

    done = was_done | hit_ground | out_of_time | out_of_fuel | flew_away

    reward = calculate_reward(env_state.physics_state, next_physics_state, clipped_action, newly_grounded, newly_terminated)

    next_env_state = EnvState(physics_state=next_physics_state, time_step=env_state.time_step + 1, done=done, gust=env_state.gust, stage=env_state.stage)

    return next_env_state, next_physics_state, reward, done