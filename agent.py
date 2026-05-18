import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

# Architecture
class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network.
    Outputs a Gaussian distribution over the 8 engine throttles (Actor)
    and the expected future reward (Critic).
    """
    action_dim: int = 8

    @nn.compact
    def __call__(self, x):
        
        scale_factors = jnp.array([
            100.0, 100.0, 1000.0,  
            10.0, 10.0, 100.0,    
            1.0, 1.0, 1.0, 1.0,    
            5.0, 5.0, 5.0,         
            2000.0                
        ])
        
        x = x / scale_factors

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        # Critic Head (Value function V(s))
        critic_hidden = nn.Dense(128)(x)
        critic_hidden = nn.relu(critic_hidden)
        value = nn.Dense(1)(critic_hidden)

        # Actor Head (Stochastic Policy pi(a|s))
        actor_hidden = nn.Dense(128)(x)
        actor_hidden = nn.relu(actor_hidden)
        
        # Mean of the action distribution
        mean = nn.Dense(self.action_dim)(actor_hidden)
        mean = nn.sigmoid(mean)
        
        # Log standard deviation
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        
        return mean, log_std, value

# Probability and sampling
@jax.jit
def sample_action(mean, log_std, key):
    """Samples an action from the Gaussian policy."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=mean.shape)
    action = mean + noise * std
    return action

@jax.jit
def calc_log_prob(action, mean, log_std):
    """Calculates the log probability of an action given the distribution."""
    std = jnp.exp(log_std)
    variance = std ** 2
    log_prob = -0.5 * ((action - mean) ** 2) / variance - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return jnp.sum(log_prob, axis=-1)

# Proximal Policy Optimization (PPO) loss function
def ppo_loss_fn(params, apply_fn, states, actions, old_log_probs, advantages, returns, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01):
    """Calculates the PPO clipped surrogate loss, value loss, and entropy bonus."""
    
    ent_coef = 0.01

    # Forward pass
    mean, log_std, values = apply_fn({'params': params}, states)
    values = jnp.squeeze(values)
    
    new_log_probs = calc_log_prob(actions, mean, log_std)
    std = jnp.exp(log_std)
    entropy = log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)
    
    # Policy loss
    ratio = jnp.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # Value loss
    value_loss = jnp.mean((returns - values) ** 2)
    
    # Entropy bonus
    entropy_loss = -jnp.mean(entropy)

    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    
    return total_loss, (policy_loss, value_loss, entropy_loss)