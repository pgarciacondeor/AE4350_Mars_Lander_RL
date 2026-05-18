import jax.numpy as jnp
from env import calculate_reward

def create_dummy_state(x, y, z, vx, vy, vz, q0, mass):
    """Helper to build a 14D state vector for testing."""

    pos = jnp.array([x, y, z])
    vel = jnp.array([vx, vy, vz])
    q = jnp.array([q0, 0.0, 0.0, 0.0])
    omega = jnp.array([0.0, 0.0, 0.0])
    m = jnp.array([mass])

    return jnp.concatenate([pos, vel, q, omega, m])

def run_test(name, state, action, done):
    reward = calculate_reward(state, action, done)
    print(f"{name:.<30} {float(reward):>8.3f}")

if __name__ == "__main__":
    print("--- RUNNING REWARD UNIT TESTS ---")
    
    dummy_action = jnp.array([0.0]*8)

    s1 = create_dummy_state(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=-1.0, q0=1.0, mass=1000.0)
    run_test("Perfect Landing", s1, dummy_action, True)

    s2 = create_dummy_state(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=-5.1, q0=1.0, mass=1000.0)
    run_test("Failed Landing (-5.1 m/s)", s2, dummy_action, True)

    s3 = create_dummy_state(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=-120.0, q0=1.0, mass=1000.0)
    run_test("High Speed Crash (120 m/s)", s3, dummy_action, True)

    s4 = create_dummy_state(x=100.0, y=0.0, z=1000.0, vx=15.0, vy=0.0, vz=0.0, q0=1.0, mass=1000.0)
    run_test("Hovering (1000m, off-target)", s4, dummy_action, False)

    s5 = create_dummy_state(x=0.0, y=0.0, z=1000.0, vx=0.0, vy=0.0, vz=-10.0, q0=1.0, mass=490.0)
    run_test("Out of Fuel Mid-Air", s5, dummy_action, False)

    s6 = create_dummy_state(x=0.0, y=0.0, z=3100.0, vx=0.0, vy=0.0, vz=50.0, q0=1.0, mass=1000.0)
    run_test("Escaped to Orbit", s6, dummy_action, True)
    
    print("---------------------------------")