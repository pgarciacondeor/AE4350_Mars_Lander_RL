import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import numpy as np

import env
import agent

# Load trained model
def load_model(filename='model_weights.pkl'):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

# Run deterministic flight
def run_flight(params):
    network = agent.ActorCritic()
    rng = jax.random.PRNGKey(42) 
    
    env_state = env.reset(rng)
    
    history = {
        'x': [], 'y': [], 'z': [],
        'vx': [], 'vy': [], 'vz': [],
        'throttles': []
    }
    
    done = False
    step = 0
    
    print("Simulating final flight...")
    
    while not done and step < env.MAX_STEPS:
        physics_state = env_state.physics_state
        
        history['x'].append(physics_state[0])
        history['y'].append(physics_state[1])
        history['z'].append(physics_state[2])
        history['vx'].append(physics_state[3])
        history['vy'].append(physics_state[4])
        history['vz'].append(physics_state[5])
        
        # Use mean action (no exploration)
        mean_action, _, _ = network.apply({'params': params}, physics_state)
        
        env_state, _, _, done = env.step(env_state, mean_action)
        
        history['throttles'].append(mean_action)
        step += 1

    print(f"Flight complete. Touchdown Z-Velocity: {history['vz'][-1]:.2f} m/s")
    return history

# Plots
def plot_results(history):
    time = np.arange(len(history['z'])) * env.DT
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(history['x'], history['y'], history['z'], label='Trajectory', color='blue', linewidth=2)
    ax1.scatter(0, 0, 0, color='red', marker='x', s=100, label='Target')
    ax1.set_title('3D Powered Descent Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, history['z'], color='green', linewidth=2)
    ax2.set_title('Altitude Profile')
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time, history['vz'], color='orange', linewidth=2)
    ax3.axhline(y=env.SAFE_Z_VELOCITY, color='red', linestyle='--', label='Max Safe Impact')
    ax3.set_title('Vertical Velocity Profile')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 2, 4)
    avg_throttle = np.mean(np.array(history['throttles']), axis=1)
    ax4.plot(time, avg_throttle * 100, color='purple', linewidth=2)
    ax4.set_title('Average MLE Throttle')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('MSL_Flight_Analysis.png', dpi=300)
    print("Dashboard saved to 'MSL_Flight_Analysis.png'.")

# 3D animation
def animate_flight(history, filename='lander_animation.gif'):
    print("Generating 3D animation with Terrain (this takes a minute)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    mid_x = (max(history['x']) + min(history['x'])) * 0.5
    mid_y = (max(history['y']) + min(history['y'])) * 0.5
    max_range = max(
        max(history['x']) - min(history['x']),
        max(history['y']) - min(history['y'])
    ) / 2.0
    
    plot_range = max(max_range, 300.0)
    
    ax.set_xlim(mid_x - plot_range, mid_x + plot_range)
    ax.set_ylim(mid_y - plot_range, mid_y + plot_range)
    ax.set_zlim(0, max(history['z']) + 50)
    
    x_vals = np.linspace(mid_x - plot_range, mid_x + plot_range, 100)
    y_vals = np.linspace(mid_y - plot_range, mid_y + plot_range, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    mountain = 500.0 * np.exp(-(X**2 + Y**2) / 10000.0)
    ridges = 20.0 * np.sin(X / 20.0) * np.cos(Y / 20.0)
    Z = mountain + ridges
    
    ax.plot_surface(X, Y, Z, cmap='copper', alpha=0.6, edgecolor='none')
    
    target_z = 500.0 * np.exp(0) + 20.0 * np.sin(0) * np.cos(0) 
    ax.scatter(0, 0, target_z, color='red', marker='X', s=200, label='Target Pad')
    
    traj_line, = ax.plot([], [], [], lw=2, color='blue', alpha=0.8)
    lander_dot, = ax.plot([], [], [], 'o', color='black', markersize=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Mars Skycrane Descent - Mount Sharp')
    
    def update(frame):
        traj_line.set_data(history['x'][:frame], history['y'][:frame])
        traj_line.set_3d_properties(history['z'][:frame])
        
        lander_dot.set_data([history['x'][frame]], [history['y'][frame]])
        lander_dot.set_3d_properties([history['z'][frame]])
        return traj_line, lander_dot
    
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, len(history['z']), 2), 
        interval=50, blit=False
    )
    ani.save(filename, writer='pillow', fps=20)
    print(f"Animation saved to {filename}")

if __name__ == "__main__":
    params = load_model()
    data = run_flight(params)
    plot_results(data)
    animate_flight(data)