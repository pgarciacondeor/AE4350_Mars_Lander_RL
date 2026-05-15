import jax
import jax.numpy as jnp
import jax.tree_util
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import time
import pickle
import csv

import env
import agent

# Hyperparameters
NUM_ENVS = 1000             
NUM_STEPS = 200             
TOTAL_UPDATES = 2000         
LEARNING_RATE = 3e-4
GAMMA = 0.99                
GAE_LAMBDA = 0.95           # Smoothing factor for advantage
CLIP_EPSILON = 0.2          # PPO clipping parameter
EPOCHS = 4                  # How many times to reuse rollout data per update
BATCH_SIZE = 256            

# Jax vectorizing
v_reset = jax.vmap(env.reset)
v_step = jax.vmap(env.step, in_axes=(0, 0))
v_sample_action = jax.vmap(agent.sample_action, in_axes=(0, None, 0))
v_calc_log_prob = jax.vmap(agent.calc_log_prob, in_axes=(0, 0, None))

# Generalized Advantage Estimation (GAE)
@jax.jit
def compute_gae(rewards, values, next_value, dones):
    """Calculates Advantages and Returns using jax.lax.scan for insane speed."""

    def body(carry, transition):
        gae, next_val = carry
        r, v, d = transition
        
        delta = r + GAMMA * next_val * (1.0 - d) - v
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - d) * gae
        return (gae, v), gae

    initial_gae = jnp.zeros_like(next_value)

    _, advantages = jax.lax.scan(
        body, 
        (initial_gae, next_value), 
        (rewards, values, dones), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

# Training state initialization
def create_train_state(rng, learning_rate):
    """Initializes the Actor-Critic network and Optax optimizer."""

    network = agent.ActorCritic()
    dummy_obs = jnp.zeros((1, 14))
    params = network.init(rng, dummy_obs)['params']
    
    schedule = optax.linear_schedule(
        init_value=learning_rate, 
        end_value=0.0, 
        transition_steps=TOTAL_UPDATES
    )
    tx = optax.adam(learning_rate=schedule)
    
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

# Single PPO step
@jax.jit
def ppo_update(train_state, states, actions, old_log_probs, advantages, returns):
    """Performs one gradient descent step on a batch of data."""
    def loss_fn(params):
        # We use our custom PPO loss function from agent.py
        total_loss, metrics = agent.ppo_loss_fn(
            params, train_state.apply_fn, states, actions, 
            old_log_probs, advantages, returns, clip_ratio=CLIP_EPSILON
        )
        return total_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(train_state.params)
    
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, metrics

# Training loop
def main():
    rng = jax.random.PRNGKey(42)
    rng, net_rng = jax.random.split(rng)
    
    train_state = create_train_state(net_rng, LEARNING_RATE)
    
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, NUM_ENVS)
    env_states = v_reset(reset_rngs)
    
    print(f"Starting Training: {NUM_ENVS} parallel landers...")
    start_time = time.time()

    with open('training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Update', 'Avg_Reward', 'Avg_Ep_Length', 'Miss_Dist_m', 'Impact_Speed_ms', 'Policy_Loss', 'Value_Loss', 'Time'])

    for update in range(TOTAL_UPDATES):
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_log_probs = []
        batch_dones = []
        
        # Rollout (collecting data)
        for step in range(NUM_STEPS):
            rng, action_rng = jax.random.split(rng)
            
            # Forward pass - get action means and values from network
            mean, log_std, value = train_state.apply_fn({'params': train_state.params}, env_states.physics_state)
            
            # Sample actions with noise for exploration
            action_rngs = jax.random.split(action_rng, NUM_ENVS)
            action = v_sample_action(mean, log_std, action_rngs)
            log_prob = v_calc_log_prob(action, mean, log_std)
            
            # Step all 1000 environments simultaneously
            next_env_states, next_physics_states, reward, done = v_step(env_states, action)
            
            # Store
            batch_states.append(env_states.physics_state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_values.append(jnp.squeeze(value))
            batch_log_probs.append(log_prob)
            batch_dones.append(done)
            
            # Advance
            rng, reset_rng = jax.random.split(rng)
            fresh_env_states = v_reset(jax.random.split(reset_rng, NUM_ENVS))
            
            env_states = jax.tree_util.tree_map(
                lambda next_s, fresh_s: jnp.where(
                    done[:, None] if next_s.ndim > 1 else done, 
                    fresh_s, next_s
                ),
                next_env_states, fresh_env_states
            )

        batch_states = jnp.stack(batch_states)
        batch_actions = jnp.stack(batch_actions)
        batch_rewards = jnp.stack(batch_rewards)
        batch_values = jnp.stack(batch_values)
        batch_log_probs = jnp.stack(batch_log_probs)
        batch_dones = jnp.stack(batch_dones)
        
        # Calculate advantages
        _, _, next_value = train_state.apply_fn({'params': train_state.params}, env_states.physics_state)
        next_value = jnp.squeeze(next_value)
        
        advantages, returns = compute_gae(batch_rewards, batch_values, next_value, batch_dones)
        
        flat_states = batch_states.reshape(-1, 14)
        flat_actions = batch_actions.reshape(-1, 8)
        flat_log_probs = batch_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # Optimize network
        for epoch in range(EPOCHS):
            train_state, (policy_loss, value_loss, entropy_loss) = ppo_update(
                train_state, flat_states, flat_actions, flat_log_probs, flat_advantages, flat_returns
            )

        if update % 10 == 0:
            avg_reward = jnp.mean(jnp.sum(batch_rewards, axis=0))
            
            total_dones = jnp.sum(batch_dones)
            avg_ep_length = (NUM_ENVS * NUM_STEPS) / jnp.maximum(1.0, total_dones) 
            
            scale_factors = jnp.array([
                100.0, 100.0, 1000.0,  # X, Y, Z
                10.0, 10.0, 100.0,     # Vx, Vy, Vz
                1.0, 1.0, 1.0, 1.0,    # Quaternions
                5.0, 5.0, 5.0,         # Angular velocity
                2000.0                 # Mass
            ])
            raw_states = batch_states * scale_factors
            
            terminal_states = raw_states[batch_dones]
            
            if len(terminal_states) > 0:
                miss_distances = jnp.linalg.norm(terminal_states[:, 0:2], axis=-1)
                avg_miss = float(jnp.mean(miss_distances))
                
                impact_speeds = jnp.linalg.norm(terminal_states[:, 3:6], axis=-1)
                avg_impact = float(jnp.mean(impact_speeds))
            else:
                avg_miss = 0.0
                avg_impact = 0.0
            
            elapsed = time.time() - start_time
            
            float_reward = float(avg_reward)
            float_length = float(avg_ep_length)
            float_ploss = float(policy_loss)
            float_vloss = float(value_loss)
            
            print(f"Up: {update:04d} | R: {float_reward:8.1f} | Len: {float_length:5.1f} | Miss: {avg_miss:5.1f}m | Impact: {avg_impact:5.1f}m/s | V-Loss: {float_vloss:8.1f}")
            
            with open('training_log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([update, float_reward, float_length, avg_miss, avg_impact, float_ploss, float_vloss, elapsed])
        
        if update % 100 == 0 and update > 0:
            with open(f'model_weights_checkpoint_{update:04d}.pkl', 'wb') as backup_file:
                pickle.dump(train_state.params, backup_file)

    with open('model_weights.pkl', 'wb') as f:
        pickle.dump(train_state.params, f)
    print("Training complete. Model saved to model_weights.pkl.")

if __name__ == "__main__":
    main()