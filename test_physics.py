import env, physics, jax
import jax.numpy as jnp

# Add this BEFORE your training loop in main(), run once, then remove it
def sanity_check_physics():
    
    
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng, jnp.int32(0))
    s = state.physics_state
    
    print(f"Initial: z={s[2]:.1f}m  vz={s[5]:.2f}m/s  mass={s[13]:.0f}kg")
    
    # Full throttle on all 8 engines
    full_throttle = jnp.ones(8)
    zero_throttle = jnp.zeros(8)
    
    # What acceleration does full throttle produce?
    sdot_full = physics.calc_dynamics(s, full_throttle)
    sdot_zero = physics.calc_dynamics(s, zero_throttle)
    
    print(f"Full throttle az={sdot_full[5]:.3f} m/s²  (gravity alone: {sdot_zero[5]:.3f} m/s²)")
    print(f"Net deceleration available: {sdot_full[5] - sdot_zero[5]:.3f} m/s²")
    
    # Simulate 200 steps with full throttle
    s_full = s
    s_free = s
    for _ in range(200):
        s_full = physics.euler_step(s_full, full_throttle, env.DT)
        s_free = physics.euler_step(s_free, zero_throttle, env.DT)
    
    print(f"\nAfter 200 steps (10s):")
    print(f"  Full throttle: z={s_full[2]:.1f}m  vz={s_full[5]:.2f}m/s  mass={s_full[13]:.0f}kg")
    print(f"  Free fall:     z={s_free[2]:.1f}m  vz={s_free[5]:.2f}m/s")
    print(f"  Fuel used: {s[13] - s_full[13]:.0f}kg")

sanity_check_physics()

def sanity_check_stage3():
    rng = jax.random.PRNGKey(99)
    # Manually construct worst-case Stage 3 state
    pos = jnp.array([400.0, 0.0, 2000.0])   # max xy offset, min altitude
    vel = jnp.array([-30.0, 0.0, -90.0])    # max lateral + vertical speed
    q   = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega = jnp.zeros(3)
    mass = jnp.array([2400.0])
    s = jnp.concatenate([pos, vel, q, omega, mass])
    
    full_throttle = jnp.ones(8)
    
    for i in range(600):  # 30 seconds
        s = physics.euler_step(s, full_throttle, env.DT)
        if i % 100 == 0:
            print(f"t={i*env.DT:.1f}s  z={s[2]:.1f}m  vz={s[5]:.1f}m/s  vx={s[3]:.1f}m/s  mass={s[13]:.0f}kg")
        if s[2] <= 500.0:
            print(f"Reached pad altitude at t={i*env.DT:.1f}s  vz={s[5]:.1f}m/s  speed={jnp.linalg.norm(s[3:6]):.1f}m/s")
            break

sanity_check_stage3()