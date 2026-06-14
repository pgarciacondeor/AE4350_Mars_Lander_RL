import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from physics import INITIAL_MASS, DRY_MASS

PAD_Z = 500.0
ACTION_DIM = 8

_SQRT2 = float(np.sqrt(2.0))

# normalise the raw 16-dim state into a deterministic pad-relative observation
def make_obs(state):
    pos = state[..., 0:3]
    vel = state[..., 3:6]
    q = state[..., 6:10]
    omega = state[..., 10:13]
    mass = state[..., 13:14]
    wind = state[..., 14:16]                         

    x = pos[..., 0:1]
    y = pos[..., 1:2]
    z = pos[..., 2:3]
    alt = z - PAD_Z # altitude above the pad

    fuel = (mass - DRY_MASS) / (INITIAL_MASS - DRY_MASS)   

    # scales keep stage-0 signals visible while high stages saturate the first tanh
    obs = jnp.concatenate([x / 100.0, y / 100.0, alt / 150.0, vel / 20.0, q, omega / 2.0, fuel * 2.0 - 1.0, wind / 2.0,], axis=-1)

    return obs


# Gaussian policy over the 8 engine throttles
class Actor(nn.Module):
    action_dim: int = ACTION_DIM

    @nn.compact
    def __call__(self, x):
        
        x = make_obs(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(_SQRT2))(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(_SQRT2))(x)
        x = nn.tanh(x)

        raw_mean = nn.Dense(self.action_dim,
                            kernel_init=nn.initializers.orthogonal(0.01))(x)
        
        # squash mean into the valid throttle range [0, 1]
        mean = 0.5 * (nn.tanh(raw_mean) + 1.0)

        log_std = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))

        # ceiling caps exploration, low floor lets std shrink for precise landings
        log_std = jnp.clip(log_std, -4.0, -0.7)

        return mean, log_std

# state-value function predicting the running-normalised return
class Critic(nn.Module):

    @nn.compact
    def __call__(self, x):

        x = make_obs(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(_SQRT2))(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(_SQRT2))(x)
        x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)

        return jnp.squeeze(value, axis=-1)

# Probability / sampling helpers
@jax.jit
def sample_action(mean, log_std, key):
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=mean.shape)
    return mean + noise * std

@jax.jit
def calc_log_prob(action, mean, log_std):
    std = jnp.exp(log_std)
    variance = std ** 2
    log_prob = -0.5 * ((action - mean) ** 2) / variance - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return jnp.sum(log_prob, axis=-1)

def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))

# PPO losses (separate actor / critic)
def actor_loss_fn(params, apply_fn, states, actions, old_log_probs, advantages, clip_ratio=0.2, ent_coef=0.01):

    mean, log_std = apply_fn({'params': params}, states)
    new_log_probs = calc_log_prob(actions, mean, log_std)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    entropy = gaussian_entropy(log_std)
    entropy_loss = -entropy

    # diagnostics
    approx_kl = jnp.mean(old_log_probs - new_log_probs)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_ratio).astype(jnp.float32))

    total_loss = policy_loss + ent_coef * entropy_loss

    return total_loss, (policy_loss, entropy_loss, approx_kl, clip_frac)


def critic_loss_fn(params, apply_fn, states, target_values):
    values = apply_fn({'params': params}, states)
    value_loss = jnp.mean((target_values - values) ** 2)
    return value_loss, value_loss