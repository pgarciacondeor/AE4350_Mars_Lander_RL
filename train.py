import os
import sys
import csv
import time
import pickle

sys.stdout.reconfigure(line_buffering=True) 

import jax
import jax.numpy as jnp
import jax.tree_util
import optax
import numpy as np
from flax.training.train_state import TrainState

import env
import agent

# Hyperparameters
def _int_env(name, default):
    return int(os.environ.get(name, default))

NUM_ENVS      = _int_env("MARS_NUM_ENVS", 1000)
NUM_STEPS     = _int_env("MARS_NUM_STEPS", 256)
TOTAL_UPDATES = _int_env("MARS_TOTAL_UPDATES", 2000)
START_STAGE   = _int_env("MARS_START_STAGE", 0)
FREEZE_STAGE  = _int_env("MARS_FREEZE_STAGE", 0)   # 1 = disable curriculum advance
SEED          = _int_env("MARS_SEED", 42)          # RNG seed
LOG_EVERY     = _int_env("MARS_LOG_EVERY", 10)

# output paths (configurable so parallel runs don't clash)
LOG_CSV     = os.environ.get("MARS_LOG_CSV", "training_log.csv")
WEIGHTS_DIR = os.environ.get("MARS_WEIGHTS_DIR", "weights")

def _float_env(name, default):
    return float(os.environ.get(name, default))

LEARNING_RATE = 3e-4
GAMMA         = _float_env("MARS_GAMMA", 0.99)   
GAE_LAMBDA    = 0.95
CLIP_EPSILON  = 0.2
EPOCHS        = 4
NUM_MINIBATCHES = 8
ENTROPY_COEF  = 0.001   
MAX_GRAD_NORM = 0.5
TARGET_KL     = _float_env("MARS_TARGET_KL", 0.02)   

# Curriculum
ADVANCE_THRESHOLD = _float_env("MARS_ADVANCE_THRESHOLD", 0.90)  # every prior stage must clear this to advance
STOP_THRESHOLD    = _float_env("MARS_STOP_THRESHOLD", 0.90)     # stop once top stage hits this
MIN_UPDATES_PER_STAGE = _int_env("MARS_MIN_UPDATES_PER_STAGE", 30)  # min time before another advance
ADAPTIVE_SAMPLING = _int_env("MARS_ADAPTIVE_SAMPLING", 1)  # 0 = uniform over unlocked stages
MAX_STAGE = 3

# Vectorised env helpers
v_reset = jax.vmap(env.reset, in_axes=(0, None, None, None))
v_step = jax.vmap(env.step, in_axes=(0, 0))
v_observe = jax.vmap(env.observe)
v_sample_action = jax.vmap(agent.sample_action, in_axes=(0, None, 0))
v_calc_log_prob = jax.vmap(agent.calc_log_prob, in_axes=(0, 0, None))

OBS_DIM = 16   # 14-dim physics state + 2-dim current wind gust

# running mean/std for return normalisation
class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


# GAE (advantages + returns in real reward space)
@jax.jit
def compute_gae(rewards, values, next_value, dones):
    def body(carry, transition):
        gae, next_val = carry
        r, v, d = transition
        delta = r + GAMMA * next_val * (1.0 - d) - v
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - d) * gae
        return (gae, v), gae

    initial_gae = jnp.zeros_like(next_value)
    _, advantages = jax.lax.scan(body, (initial_gae, next_value), (rewards, values, dones), reverse=True)
    returns = advantages + values
    return advantages, returns


def make_lr_schedule(resume_update):

    steps_per_update = EPOCHS * NUM_MINIBATCHES
    total_opt_steps = TOTAL_UPDATES * steps_per_update

    return optax.linear_schedule(init_value=LEARNING_RATE * (1.0 - resume_update / TOTAL_UPDATES), end_value=0.0, transition_steps=max(1, total_opt_steps - resume_update * steps_per_update),)

def create_states(rng, resume_update=0):
    actor_net = agent.Actor()
    critic_net = agent.Critic()
    dummy = jnp.zeros((1, OBS_DIM))

    rng, a_rng, c_rng = jax.random.split(rng, 3)

    if resume_update > 0:
        with open(os.path.join(WEIGHTS_DIR, f'actor_checkpoint_{resume_update:04d}.pkl'), 'rb') as f:
            actor_params = pickle.load(f)
        with open(os.path.join(WEIGHTS_DIR, f'critic_checkpoint_{resume_update:04d}.pkl'), 'rb') as f:
            critic_params = pickle.load(f)

    else:
        actor_params = actor_net.init(a_rng, dummy)['params']
        critic_params = critic_net.init(c_rng, dummy)['params']

    schedule = make_lr_schedule(resume_update)
    actor_tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(schedule))
    critic_tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(schedule))

    actor_state = TrainState.create(apply_fn=actor_net.apply, params=actor_params, tx=actor_tx)
    critic_state = TrainState.create(apply_fn=critic_net.apply, params=critic_params, tx=critic_tx)

    return actor_state, critic_state

@jax.jit
def actor_update(state, states, actions, old_log_probs, advantages):

    def loss_fn(params):
        return agent.actor_loss_fn(params, state.apply_fn, states, actions, old_log_probs, advantages, clip_ratio=CLIP_EPSILON, ent_coef=ENTROPY_COEF)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, aux), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, aux


@jax.jit
def critic_update(state, states, target_values):

    def loss_fn(params):
        return agent.critic_loss_fn(params, state.apply_fn, states, target_values)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, value_loss), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, value_loss


# roll out the deterministic policy for one episode and print it
def trace_episode(actor_state, stage, seed=99):

    rng = jax.random.PRNGKey(seed)
    env_state = env.reset(rng, jnp.int32(stage), fixed_stage=jnp.int32(stage))
    total = 0.0

    for i in range(env.MAX_STEPS):
        mean, _ = actor_state.apply_fn({'params': actor_state.params}, env.observe(env_state)[None])
        action = mean[0]
        env_state, phys, reward, done = env.step(env_state, action)
        total += float(reward)

        if i < 2 or done or i % 40 == 0:
            print(f"    step {i:3d} | z={float(phys[2]):6.1f} | vz={float(phys[5]):6.2f} "
                  f"| sp={float(jnp.linalg.norm(phys[3:6])):5.2f} | thr={float(jnp.mean(action)):.2f} "
                  f"| r={float(reward):7.2f} | cum={total:8.2f}")
            
        if done:
            print(f"    -> done @ step {i}, total={total:.2f}")
            break


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    RESUME_UPDATE = 0
    stage = START_STAGE
    last_advance_update = RESUME_UPDATE
    stage_weights = (jnp.arange(4) <= stage).astype(jnp.float32)

    rng = jax.random.PRNGKey(SEED)
    rng, net_rng = jax.random.split(rng)
    actor_state, critic_state = create_states(net_rng, RESUME_UPDATE)
    ret_rms = RunningMeanStd()

    rng, reset_rng = jax.random.split(rng)
    env_states = v_reset(jax.random.split(reset_rng, NUM_ENVS), stage, -1, stage_weights)

    print(f"Training: {NUM_ENVS} envs x {NUM_STEPS} steps, {TOTAL_UPDATES} updates, "
          f"start stage {stage} (backend={jax.default_backend()})")
    start_time = time.time()

    if RESUME_UPDATE == 0:
        with open(LOG_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['Update', 'Avg_Reward', 'Avg_Ep_Length', 'Miss_Dist_m',
                 'Impact_Speed_ms', 'Policy_Loss', 'Value_Loss', 'Stage',
                 'Success_Rate', 'Entropy', 'KL', 'Time']
                + [f'Succ_S{s}' for s in range(MAX_STAGE + 1)])

    minibatch_size = (NUM_ENVS * NUM_STEPS) // NUM_MINIBATCHES
    stop_training = False
    best_score = -1.0 

    for update in range(RESUME_UPDATE, TOTAL_UPDATES):

        if update % LOG_EVERY == 0:
            print(f"\n=== trace @ update {update} (stage {stage}) ===")
            trace_episode(actor_state, stage)

        b_states, b_next_states, b_actions = [], [], []
        b_rewards, b_values, b_log_probs, b_dones, b_stages = [], [], [], [], []

        ret_mean, ret_std = ret_rms.mean, ret_rms.std

        # rollout
        for _ in range(NUM_STEPS):
            obs_in = v_observe(env_states)
            mean, log_std = actor_state.apply_fn({'params': actor_state.params}, obs_in)
            value_norm = critic_state.apply_fn({'params': critic_state.params}, obs_in)

            rng, action_rng = jax.random.split(rng)
            action = v_sample_action(mean, log_std, jax.random.split(action_rng, NUM_ENVS))
            log_prob = v_calc_log_prob(action, mean, log_std)

            next_env_states, next_phys, reward, done = v_step(env_states, action)

            b_states.append(obs_in)
            b_next_states.append(next_phys)
            b_actions.append(action)
            b_rewards.append(reward)
            b_values.append(value_norm * ret_std + ret_mean)  
            b_log_probs.append(log_prob)
            b_dones.append(done)
            b_stages.append(env_states.stage)

            rng, reset_rng = jax.random.split(rng)
            fresh = v_reset(jax.random.split(reset_rng, NUM_ENVS), stage, -1, stage_weights)
            env_states = jax.tree_util.tree_map(lambda nxt, fr: jnp.where(done[:, None] if nxt.ndim > 1 else done, fr, nxt), next_env_states, fresh)

        b_states = jnp.stack(b_states)
        b_next_states = jnp.stack(b_next_states)
        b_actions = jnp.stack(b_actions)
        b_rewards = jnp.stack(b_rewards)
        b_values = jnp.stack(b_values)
        b_log_probs = jnp.stack(b_log_probs)
        b_dones = jnp.stack(b_dones).astype(jnp.float32)
        b_stages = jnp.stack(b_stages)

        next_value_norm = critic_state.apply_fn({'params': critic_state.params},
                                                v_observe(env_states))
        next_value = next_value_norm * ret_std + ret_mean

        advantages, returns = compute_gae(b_rewards, b_values, next_value, b_dones)

        ret_rms.update(np.asarray(returns))
        target_values = (returns - ret_rms.mean) / ret_rms.std

        flat_states = b_states.reshape(-1, OBS_DIM)
        flat_actions = b_actions.reshape(-1, 8)
        flat_log_probs = b_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_targets = target_values.reshape(-1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        num_transitions = flat_states.shape[0]
        last_metrics = (0.0, 0.0, 0.0, 0.0, 0.0)
        stop_epochs = False
        for _ in range(EPOCHS):
            if stop_epochs:
                break
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, num_transitions)
            for start in range(0, num_transitions, minibatch_size):
                idx = perm[start:start + minibatch_size]
                actor_state, (p_loss, ent_loss, kl, clip_frac) = actor_update(
                    actor_state, flat_states[idx], flat_actions[idx],
                    flat_log_probs[idx], flat_advantages[idx])
                critic_state, v_loss = critic_update(
                    critic_state, flat_states[idx], flat_targets[idx])
                last_metrics = (p_loss, v_loss, ent_loss, kl, clip_frac)

                # stop early if the policy moved too far
                if float(kl) > 1.5 * TARGET_KL:
                    stop_epochs = True
                    break

        # metrics
        if update % LOG_EVERY == 0:
            p_loss, v_loss, ent_loss, kl, clip_frac = (float(x) for x in last_metrics)

            done_mask = b_dones.astype(bool)
            term_states = b_next_states[done_mask]   
            n_term = int(term_states.shape[0])

            per_stage_sr = [0.0] * (MAX_STAGE + 1)   
            per_stage_cnt = [0] * (MAX_STAGE + 1)

            if n_term > 0:
                tvel = term_states[:, 3:6]
                tpos = term_states[:, 0:3]
                tq = term_states[:, 6:10]
                vz_ok = tvel[:, 2] >= env.SAFE_Z_VELOCITY
                vxy_ok = jnp.linalg.norm(tvel[:, 0:2], axis=1) <= env.SAFE_XY_VELOCITY
                upright_ok = tq[:, 0] > 0.95
                on_pad = jnp.linalg.norm(tpos[:, 0:2], axis=1) < env.PAD_RADIUS
                grounded = tpos[:, 2] <= 502.0   
                succ_vec = np.asarray(vz_ok & vxy_ok & upright_ok & on_pad & grounded)
                successes = int(succ_vec.sum())
                success_rate = successes / n_term
                avg_miss = float(jnp.mean(jnp.linalg.norm(tpos[:, 0:2], axis=1)))
                avg_impact = float(jnp.mean(jnp.linalg.norm(tvel, axis=1)))

                # Per-stage success
                term_stage = np.asarray(b_stages.astype(jnp.int32))[np.asarray(done_mask)]
                for s in range(MAX_STAGE + 1):
                    m = term_stage == s
                    per_stage_cnt[s] = int(m.sum())
                    if per_stage_cnt[s] > 0:
                        per_stage_sr[s] = float(succ_vec[m].mean())

            else:
                successes, success_rate, avg_miss, avg_impact = 0, 0.0, 0.0, 0.0

            # Average undiscounted return per episode
            not_done = jnp.concatenate(
                [jnp.ones((1, NUM_ENVS)), jnp.cumprod(1.0 - b_dones[:-1], axis=0)], axis=0)
            avg_reward = float(jnp.mean(jnp.sum(b_rewards * not_done, axis=0)))
            avg_ep_len = (NUM_ENVS * NUM_STEPS) / max(1.0, float(jnp.sum(b_dones)))

            # advance only when EVERY unlocked stage is mastered after minimum time
            all_mastered = all(per_stage_sr[s] >= ADVANCE_THRESHOLD and per_stage_cnt[s] >= 20 for s in range(stage + 1))

            if not FREEZE_STAGE and (update - last_advance_update) >= MIN_UPDATES_PER_STAGE:
                if stage < MAX_STAGE and all_mastered:
                    stage += 1
                    last_advance_update = update
                    print(f"  *** CURRICULUM ADVANCE -> stage {stage} "
                          f"(all stages S0-S{stage-1} >= {ADVANCE_THRESHOLD:.0%}) ***")
                    
                elif stage == MAX_STAGE and all_mastered and per_stage_sr[stage] >= STOP_THRESHOLD:
                    print(f"  *** ALL STAGES MASTERED (>= {STOP_THRESHOLD:.0%}) - stopping early ***")
                    stop_training = True

            # adaptive sampling - practise the weakest unlocked stage most, floor keeps mastered ones rehearsed
            w = [0.0] * (MAX_STAGE + 1)
            for s in range(stage + 1):
                w[s] = (0.15 + (1.0 - per_stage_sr[s])) if ADAPTIVE_SAMPLING else 1.0
            stage_weights = jnp.array(w)

            # checkpoint the peak high-stage policy so a later collapse never loses it
            hs_score = per_stage_sr[2] + 2.0 * per_stage_sr[3]
            if hs_score > best_score and update > 0:
                best_score = hs_score
                with open(os.path.join(WEIGHTS_DIR, 'actor_best.pkl'), 'wb') as f:
                    pickle.dump(actor_state.params, f)
                with open(os.path.join(WEIGHTS_DIR, 'critic_best.pkl'), 'wb') as f:
                    pickle.dump(critic_state.params, f)

            elapsed = time.time() - start_time
            per_stage_str = " ".join(
                f"S{s}={per_stage_sr[s]:.0%}" for s in range(MAX_STAGE + 1))
            print(f"Up {update:04d} | St {stage} | R {avg_reward:8.2f} | Len {avg_ep_len:6.1f} "
                  f"| Succ {success_rate:5.1%} [{per_stage_str}] | Miss {avg_miss:6.1f}m "
                  f"| Imp {avg_impact:5.1f} | VL {v_loss:7.3f} | PL {p_loss:7.4f} | "
                  f"Ent {ent_loss:6.2f} | KL {kl:6.4f}")

            with open(LOG_CSV, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [update, avg_reward, avg_ep_len, avg_miss, avg_impact,
                     p_loss, v_loss, stage, success_rate, ent_loss, kl, elapsed]
                    + per_stage_sr)

        if update % 100 == 0 and update > 0:
            with open(os.path.join(WEIGHTS_DIR, f'actor_checkpoint_{update:04d}.pkl'), 'wb') as f:
                pickle.dump(actor_state.params, f)
            with open(os.path.join(WEIGHTS_DIR, f'critic_checkpoint_{update:04d}.pkl'), 'wb') as f:
                pickle.dump(critic_state.params, f)

        if stop_training:
            break

    with open(os.path.join(WEIGHTS_DIR, 'actor_weights.pkl'), 'wb') as f:
        pickle.dump(actor_state.params, f)

    with open(os.path.join(WEIGHTS_DIR, 'critic_weights.pkl'), 'wb') as f:
        pickle.dump(critic_state.params, f)

    print("Training complete. Saved weights/actor_weights.pkl + critic_weights.pkl")


if __name__ == "__main__":
    main()