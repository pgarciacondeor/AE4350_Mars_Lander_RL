import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import numpy as np
import csv

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
    ax1.scatter(0, 0, 500.0, color='red', marker='X', s=100, label='Target')
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

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(0, 2500)
    
    x_vals = np.linspace(-400, 400, 150)
    y_vals = np.linspace(-400, 400, 150)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    mountain = 500.0 * np.exp(-(X**2 + Y**2) / 10000.0)
    ridges = 20.0 * np.sin(X / 20.0) * np.cos(Y / 20.0)
    raw_terrain = mountain + ridges
    
    is_pad = (X**2 + Y**2) <= 225.0
    Z = np.where(is_pad, 500.0, raw_terrain)
    
    ax.plot_surface(X, Y, Z, cmap='copper', alpha=0.6, edgecolor='none')
    
    ax.scatter(0, 0, 500.0, color='red', marker='X', s=300, label='Target Pad')
    
    traj_line, = ax.plot([], [], [], lw=2, color='blue', alpha=0.8)
    lander_dot, = ax.plot([], [], [], 'o', color='black', markersize=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('MSL Glideslope Descent - Mount Sharp Mesa')
    
    def update(frame):
        traj_line.set_data(history['x'][:frame], history['y'][:frame])
        traj_line.set_3d_properties(history['z'][:frame])
        
        lander_dot.set_data([history['x'][frame]], [history['y'][frame]])
        lander_dot.set_3d_properties([history['z'][frame]])
        return traj_line, lander_dot
    
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, len(history['z']), 3), 
        interval=40, blit=False
    )
    ani.save(filename, writer='pillow', fps=24)
    print(f"Animation saved to {filename}")

def generate_training_plots():
    updates, rewards, lengths = [], [], []
    misses, impacts = [], []
    p_losses, v_losses = [], []

    try:
        with open('training_log.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                updates.append(int(row['Update']))
                rewards.append(float(row['Avg_Reward']))
                lengths.append(float(row['Avg_Ep_Length']))
                
                # Fetch our new metrics!
                misses.append(float(row.get('Miss_Dist_m', 0.0)))
                impacts.append(float(row.get('Impact_Speed_ms', 0.0)))
                
                p_losses.append(float(row['Policy_Loss']))
                v_losses.append(float(row['Value_Loss']))
    except FileNotFoundError:
        print("Error: 'training_log.csv' not found. Run train.py first!")
        return

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].plot(updates, rewards, color='blue', linewidth=2)
    axs[0, 0].set_title('Episode Return (Average Reward)')
    axs[0, 0].set_xlabel('Updates')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)

    axs[0, 1].plot(updates, lengths, color='green', linewidth=2)
    axs[0, 1].set_title('Average Episode Length')
    axs[0, 1].set_xlabel('Updates')
    axs[0, 1].set_ylabel('Steps Survived')
    axs[0, 1].set_ylim(0, 1000)
    axs[0, 1].grid(True)

    axs[0, 2].plot(updates, v_losses, color='red', linewidth=2)
    axs[0, 2].set_title('Value Loss (Critic)')
    axs[0, 2].set_xlabel('Updates')
    axs[0, 2].set_ylabel('Loss')
    axs[0, 2].set_yscale('log') 
    axs[0, 2].grid(True)

    axs[1, 0].plot(updates, misses, color='purple', linewidth=2)
    axs[1, 0].set_title('Miss Distance (m)')
    axs[1, 0].set_xlabel('Updates')
    axs[1, 0].set_ylabel('Meters off target')
    axs[1, 0].grid(True)

    axs[1, 1].plot(updates, impacts, color='darkorange', linewidth=2)
    axs[1, 1].set_title('Impact Speed (m/s)')
    axs[1, 1].set_xlabel('Updates')
    axs[1, 1].set_ylabel('Velocity at Touchdown')
    axs[1, 1].axhline(y=2.5, color='black', linestyle='--', label='Safe (<2.5 m/s)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(updates, p_losses, color='gray', linewidth=2)
    axs[1, 2].set_title('Policy Loss (Actor)')
    axs[1, 2].set_xlabel('Updates')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('PPO_Training_Dashboard.png', dpi=300)

if __name__ == "__main__":
    params = load_model()
    data = run_flight(params)
    plot_results(data)
    animate_flight(data)
    generate_training_plots()