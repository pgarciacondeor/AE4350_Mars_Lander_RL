# Mars Lander — PPO powered descent

Reinforcement-learning agent (PPO, JAX) that performs a 6-DOF powered
descent and soft landing on a 30 m pad atop a mountain in Mars
conditions (low gravity, thin-atmosphere drag, wind gusts), using
multi-task curriculum learning over four difficulty stages.

## Method

- **Environment** (`env.py`, `physics.py`): rigid-body lander with 8 throttled
  engines, quaternion attitude, Mars gravity + exponential-atmosphere drag, and
  a time-varying horizontal wind-gust disturbance that ramps up with the
  curriculum stage. A braking-envelope reward term discourages descending
  faster than `v_safe(h)=√(2·a_brake·h)` (prevents diving).
- **Agent** (`agent.py`): separate Actor and Critic MLPs, tanh,
  orthogonal init, diagonal-Gaussian policy over the 8 throttles with a clamped
  log_std. Observation is pad-relative (14 physics dimensions + 2 gust dimensions).
- **Training** (`train.py`): PPO with GAE, return normalization, advantage
  normalization, linear-decayed LR. Curriculum advances when the hardest current
  stage's success exceeds a threshold of 90%, per-stage success is tracked so easy
  stages can't mask a failing hard stage. Best (and final) weights are saved.

Behaviour is controlled by env vars (defaults in parentheses), e.g.:

| var | meaning |
|-----|---------|
| `MARS_NUM_ENVS` (1000), `MARS_NUM_STEPS` (256) | parallel envs / rollout length |
| `MARS_TOTAL_UPDATES` (2000) | PPO updates |
| `MARS_SEED` (42) | RNG seed |
| `MARS_GAMMA` (0.99) | discount |
| `MARS_ADVANCE_THRESHOLD` (0.80), `MARS_STOP_THRESHOLD` (0.90) | curriculum advance / early-stop |
| `MARS_ENVELOPE_COEF` (0.1) | braking-envelope strength |
| `MARS_START_STAGE` (0), `MARS_FREEZE_STAGE` (0) | start / pin curriculum stage |

## Files

`physics.py` dynamics, `env.py` env + reward + curriculum, `agent.py` networks +
PPO loss, `train.py` training loop, `analysis.py` figures (trajectories,
state-action map, Monte-Carlo dispersion, robustness, training curves),
`test_physics_stages.py` / `test_reward.py` sanity checks.
