import os
import csv
import glob
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Larger fonts
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 23,
    "axes.labelsize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
    "lines.linewidth": 2.6,
    "lines.markersize": 8,
})

import jax
import jax.numpy as jnp

import env
import agent
import physics
from physics import INITIAL_MASS, DRY_MASS

ACTOR = agent.Actor()
v_observe = jax.vmap(env.observe)
os.makedirs("figures", exist_ok=True)
PAD_Z = 500.0

def load_actor(weights_dir="weights"):
    for name in ("actor_best.pkl", "actor_weights.pkl"):
        p = os.path.join(weights_dir, name)
        if os.path.exists(p):
            print(f"loaded {p}")
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"no actor weights in {weights_dir}/")

def policy_mean(params, obs):
    mean, _ = ACTOR.apply({"params": params}, obs)
    return mean

def _save(fig, fname, pdf=False, tight=True, pad=0.04):
    
    kw = dict(bbox_inches="tight", pad_inches=pad) if tight else dict()
    fig.savefig(fname, dpi=150, **kw)
    if pdf:
        fig.savefig(fname.replace(".png", ".pdf"), **kw)
    plt.close(fig)
    print(f"saved {fname}" + (" (+pdf)" if pdf else ""))

def run_flight(params, stage=0, seed=42, gravity_scale=1.0, drag_scale=1.0, controller=None):
    rng = jax.random.PRNGKey(seed)
    env_state = env.reset(rng, jnp.int32(stage), fixed_stage=jnp.int32(stage))

    keys = ["t", "x", "y", "z", "vx", "vy", "vz", "speed",
            "tilt_deg", "mean_throttle", "windx", "windy", "horiz_dist", "mass"]
    h = {k: [] for k in keys}
    done, step = False, 0
    while (not done) and step < env.MAX_STEPS:
        s = env_state.physics_state
        wind = np.array(env.current_wind(env_state))
        q0 = float(jnp.clip(s[6], -1.0, 1.0))
        action = controller(s) if controller is not None else policy_mean(params, env.observe(env_state))
        h["t"].append(step * env.DT)
        h["x"].append(float(s[0])); h["y"].append(float(s[1])); h["z"].append(float(s[2]))
        h["vx"].append(float(s[3])); h["vy"].append(float(s[4])); h["vz"].append(float(s[5]))
        h["speed"].append(float(jnp.linalg.norm(s[3:6])))
        h["tilt_deg"].append(np.degrees(2.0 * np.arccos(abs(q0))))
        h["mean_throttle"].append(float(jnp.mean(action)))
        h["windx"].append(wind[0]); h["windy"].append(wind[1])
        h["horiz_dist"].append(float(jnp.linalg.norm(s[0:2])))
        h["mass"].append(float(s[13]))
        env_state, _, _, done = env.step(env_state, action, gravity_scale, drag_scale)
        step += 1

    h = {k: np.array(v) for k, v in h.items()}
    safe = (h["vz"][-1] >= env.SAFE_Z_VELOCITY
            and np.linalg.norm([h["vx"][-1], h["vy"][-1]]) <= env.SAFE_XY_VELOCITY
            and h["tilt_deg"][-1] < np.degrees(2 * np.arccos(0.95))
            and h["horiz_dist"][-1] < env.PAD_RADIUS)
    print(f"flight stage {stage}: vz={h['vz'][-1]:.2f} speed={h['speed'][-1]:.2f} "
          f"miss={h['horiz_dist'][-1]:.1f}m tilt={h['tilt_deg'][-1]:.1f}deg "
          f"-> {'SUCCESS' if safe else 'fail'} ({len(h['t'])} steps)")
    return h

def plot_trajectory_report(h, fname="figures/trajectory.png", stage=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    title = "Powered-descent trajectory" + (f" - stage {stage}" if stage is not None else "")

    # side profile
    a = axes[0]
    r = h["horiz_dist"]
    alt = h["z"] - PAD_Z
    rr = np.linspace(0, max(r.max() * 1.05, env.PAD_RADIUS * 2), 300)
    terr = np.array(env.get_terrain_elevation(jnp.array(rr), jnp.zeros_like(jnp.array(rr)))) - PAD_Z
    a.fill_between(rr, terr, terr.min() - 20, color="peru", alpha=0.35, lw=0)
    a.plot(rr, terr, color="peru", lw=1.2)
    a.plot([0, env.PAD_RADIUS], [0, 0], color="red", lw=4, solid_capstyle="butt")
    sc = a.scatter(r, alt, c=h["t"], cmap="viridis", s=10)
    a.scatter(r[0], alt[0], marker="o", color="black", s=40, zorder=5)
    a.scatter(r[-1], alt[-1], marker="X", color="crimson", s=70, zorder=5)
    a.set_xlabel("horizontal distance from pad [m]"); a.set_ylabel("altitude above pad [m]")
    a.set_title("Side profile"); a.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=a, pad=0.01); cb.set_label("time [s]")

    # top-down ground track
    b = axes[1]
    th = np.linspace(0, 2 * np.pi, 100)
    b.fill(env.PAD_RADIUS * np.cos(th), env.PAD_RADIUS * np.sin(th),
           color="red", alpha=0.18)
    b.plot(env.PAD_RADIUS * np.cos(th), env.PAD_RADIUS * np.sin(th),
           color="red", lw=1.5)
    sc2 = b.scatter(h["x"], h["y"], c=h["t"], cmap="viridis", s=10)
    b.scatter(h["x"][0], h["y"][0], marker="o", color="black", s=40, zorder=5)
    b.scatter(h["x"][-1], h["y"][-1], marker="X", color="crimson", s=70, zorder=5)
    b.set_xlabel("x [m]"); b.set_ylabel("y [m]"); b.set_title("Ground track (top-down)")
    b.set_aspect("equal", adjustable="datalim"); b.grid(alpha=0.3)

    # braking profile
    c = axes[2]
    c.plot(-h["vz"], alt, color="tab:blue", lw=2)
    c.axvline(-env.SAFE_Z_VELOCITY, color="red", ls="--", lw=1.2)
    c.text(-env.SAFE_Z_VELOCITY, alt.max() * 0.96, f" safe {abs(env.SAFE_Z_VELOCITY):.1f} m/s",
           color="red", fontsize=16, va="top", ha="left")
    c.scatter([-h["vz"][-1]], [alt[-1]], marker="X", color="crimson", s=70, zorder=5)
    c.set_xlabel("descent rate $-v_z$ [m/s]"); c.set_ylabel("altitude above pad [m]")
    c.set_title("Braking profile"); c.grid(alpha=0.3)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="black", label="start"),
               Line2D([0], [0], marker="X", color="w", markerfacecolor="crimson", label="touchdown"),
               Line2D([0], [0], color="red", lw=3, label="landing pad / safe limit"),
               Line2D([0], [0], color="peru", lw=3, label="terrain")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=18,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(title, fontsize=23)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, fname, pdf=True)

def plot_trajectory_dashboard(h, fname="figures/flight_dashboard.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    t = h["t"]; alt = h["z"] - PAD_Z

    ax = axes[0, 0]
    ax.plot(t, alt, color="tab:green", lw=2)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.text(t[-1], 0, " pad", fontsize=16, va="bottom", ha="right", color="k")
    ax.set_title("Altitude above pad"); ax.set_xlabel("time [s]"); ax.set_ylabel("h [m]"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, h["vz"], color="tab:orange", lw=2, label="$v_z$")
    ax.plot(t, h["speed"], color="tab:purple", lw=1.5, ls="--", label="$|v|$")
    ax.axhline(env.SAFE_Z_VELOCITY, color="red", ls=":", lw=1.2)
    ax.set_title("Velocity"); ax.set_xlabel("time [s]"); ax.set_ylabel("[m/s]"); ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16)

    ax = axes[0, 2]
    ax.plot(t, h["mean_throttle"] * 100, color="tab:blue", lw=2)
    ax.set_title("Mean engine throttle"); ax.set_xlabel("time [s]"); ax.set_ylabel("throttle [%]")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, h["tilt_deg"], color="tab:red", lw=2)
    lim = np.degrees(2 * np.arccos(0.95))
    ax.axhline(lim, color="k", ls="--", lw=1)
    ax.text(t[-1], lim, "upright limit ", fontsize=16, va="bottom", ha="right")
    ax.set_title("Attitude (tilt)"); ax.set_xlabel("time [s]"); ax.set_ylabel("[deg]"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, h["horiz_dist"], color="tab:gray", lw=2)
    ax.axhline(env.PAD_RADIUS, color="red", ls="--", lw=1)
    ax.text(t[-1], env.PAD_RADIUS, "pad radius ", fontsize=16, va="bottom", ha="right", color="red")
    ax.set_title("Horizontal distance"); ax.set_xlabel("time [s]"); ax.set_ylabel("[m]"); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t, h["windx"], color="tab:cyan", lw=1.5, label="gust $a_x$")
    ax.plot(t, h["windy"], color="tab:olive", lw=1.5, label="gust $a_y$")
    ax.set_title("Wind-gust disturbance"); ax.set_xlabel("time [s]"); ax.set_ylabel("[m/s$^2$]"); ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16)

    fig.suptitle("Landing behaviour over time", fontsize=23)
    _save(fig, fname, pdf=True)

def _read_log(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for r in reader:
            try:
                vals = {k: float(r[k]) for k in fields}
            except (TypeError, ValueError):
                continue
            for k, v in vals.items():
                rows.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in rows.items()}

def generate_training_plots(pattern="logs/training_log_seed*.csv", fname="figures/training_dashboard.png"):
    logs = [_read_log(p) for p in sorted(glob.glob(pattern)) if os.path.getsize(p) > 0]
    logs = [l for l in logs if "Update" in l and len(l["Update"]) > 1]
    if not logs:
        print("no training logs found, skipping training dashboard")
        return
    multi = len(logs) > 1

    def agg(col):
        present = [l for l in logs if col in l]
        if not present:
            return None
        n = min(len(l[col]) for l in present)
        x = present[0]["Update"][:n]
        stack = np.stack([l[col][:n] for l in present])
        return x, stack.mean(0), stack.std(0)

    def panel(ax, col, title, ylab, ref=None, ref_label=None):
        r = agg(col)
        if r is None:
            return
        x, m, sd = r
        ax.plot(x, m, lw=2, color="tab:blue")
        if multi:
            ax.fill_between(x, m - sd, m + sd, alpha=0.25, color="tab:blue")
        if ref is not None:
            ax.axhline(ref, color="red", ls="--", lw=1)
            ax.text(x[0], ref, f" {ref_label}", color="red", fontsize=16, va="bottom", ha="left")
        ax.set_title(title); ax.set_xlabel("update"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    panel(axes[0, 0], "Avg_Reward", "Episode return", "reward")
    panel(axes[0, 1], "Impact_Speed_ms", "Impact speed", "m/s",
          ref=env.SAFE_XY_VELOCITY, ref_label="safe")

    a = axes[0, 2]
    tot = agg("Success_Rate")
    if tot is not None:
        x, m, sd = tot
        a.plot(x, m, lw=2.5, color="black", label="total")
        if multi:
            a.fill_between(x, m - sd, m + sd, alpha=0.18, color="black")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for s in range(4):
        r = agg(f"Succ_S{s}")
        if r is not None:
            a.plot(r[0], r[1], lw=1.6, color=colors[s], label=f"stage {s}")
    a.set_title("Success rate (total & per stage)"); a.set_xlabel("update")
    a.set_ylabel("fraction"); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)
    a.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=16)

    panel(axes[1, 0], "Miss_Dist_m", "Miss distance", "m")
    panel(axes[1, 1], "Value_Loss", "Critic (value) loss", "MSE")
    panel(axes[1, 2], "Stage", "Curriculum stage (hardest in mix)", "stage")

    fig.suptitle(f"PPO training ({len(logs)} seed{'s' if multi else ''}, mean$\\pm$std)", fontsize=23)
    _save(fig, fname, pdf=True)


def _obs_vec(alt, vz):
    z = PAD_Z + alt
    phys = jnp.array([0.0, 0.0, z, 0.0, 0.0, vz, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, INITIAL_MASS])
    return jnp.concatenate([phys, jnp.zeros(2)])

def state_action_map(params, fname="figures/state_action_map.png", alt_max=150.0):
    alts = np.linspace(0, alt_max, 90)
    vzs = np.linspace(-15, 5, 90)
    AA, VV = np.meshgrid(alts, vzs)
    obs = jax.vmap(_obs_vec)(jnp.array(AA.ravel()), jnp.array(VV.ravel()))
    thr = np.array(jnp.mean(policy_mean(params, obs), axis=-1)).reshape(AA.shape)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    pc = ax.pcolormesh(AA, VV, thr * 100, cmap="viridis", shading="auto")
    ax.axhline(env.SAFE_Z_VELOCITY, color="red", ls="--", lw=1.4)
    ax.text(alt_max * 0.99, env.SAFE_Z_VELOCITY, "safe $v_z$ ", color="red",
            fontsize=18, va="bottom", ha="right")
    # overlay the analytical braking envelope -v_safe(h); the learned boundary tracks it
    alt_line = np.linspace(0, alt_max, 300)
    v_env = -np.sqrt(2.0 * env.A_BRAKE * alt_line)
    m = v_env >= vzs.min()
    ax.plot(alt_line[m], v_env[m], color="white", lw=3.0,
            label="braking envelope $-v_{\\mathrm{safe}}(h)$")
    ax.set_ylim(vzs.min(), vzs.max())
    ax.legend(loc="upper right", framealpha=0.9)
    cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.set_label("commanded mean throttle [%]")
    ax.set_xlabel("altitude above pad [m]"); ax.set_ylabel("vertical velocity $v_z$ [m/s]")
    ax.set_title("Learned throttle control law")
    fig.tight_layout()
    _save(fig, fname, pdf=True)

def evaluate_success(params, key, n=2048, stage=1, gravity_scale=1.0, drag_scale=1.0):
    states = jax.vmap(lambda k: env.reset(k, jnp.int32(stage), fixed_stage=jnp.int32(stage)))(
        jax.random.split(key, n))

    def body(carry, _):
        states, done_prev, succ, impact = carry
        action, _ = ACTOR.apply({"params": params}, v_observe(states))
        nxt, np_, _, done = jax.vmap(
            lambda s, a: env.step(s, a, gravity_scale, drag_scale))(states, action)
        newly = done & (~done_prev)
        ok = ((np_[:, 5] >= env.SAFE_Z_VELOCITY)
              & (jnp.linalg.norm(np_[:, 3:5], axis=1) <= env.SAFE_XY_VELOCITY)
              & (np_[:, 6] > 0.95)
              & (jnp.linalg.norm(np_[:, 0:2], axis=1) < env.PAD_RADIUS)
              & (np_[:, 2] <= 502.0))
        succ = jnp.where(newly, ok, succ)
        impact = jnp.where(newly, jnp.linalg.norm(np_[:, 3:6], axis=1), impact)
        return (nxt, done | done_prev, succ, impact), None

    init = (states, jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n))
    (_, _, succ, impact), _ = jax.lax.scan(body, init, None, length=env.MAX_STEPS)
    return float(jnp.mean(succ)), float(jnp.mean(impact))

def robustness_curves(params, stage=1, n=2048, fname="figures/robustness.png"):
    key = jax.random.PRNGKey(0)
    scales = np.linspace(0.6, 1.4, 9)
    grav = [evaluate_success(params, key, n, stage, gravity_scale=float(s))[0] for s in scales]
    drag = [evaluate_success(params, key, n, stage, drag_scale=float(s))[0] for s in scales]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales * 100, np.array(grav) * 100, "-o", label="gravity perturbed")
    ax.plot(scales * 100, np.array(drag) * 100, "-s", label="drag perturbed")
    ax.axvline(100, color="k", ls="--", lw=1)
    ax.text(100, 2, " training\n condition", fontsize=16, va="bottom", ha="left", color="k")
    ax.set_xlabel("environment parameter relative to training [%]")
    ax.set_ylabel("success rate [%]")
    ax.set_title("Post-training robustness to environment changes")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=18)
    _save(fig, fname, pdf=True)

# Analytical baseline: constant-divergence vertical controller
def baseline_controller(state):
    alt = jnp.maximum(state[2] - PAD_Z, 0.0)
    vz = state[5]
    vz_target = -jnp.clip(0.5 * alt + 0.5, 0.5, 10.0)
    thr = 0.457 + 0.2 * (vz_target - vz)
    return jnp.full(8, jnp.clip(thr, 0.0, 1.0))

def _eval_success(act, key, n=512, stage=1, engine_health=None):
    states = jax.vmap(lambda k: env.reset(k, jnp.int32(3), fixed_stage=jnp.int32(stage)))(
        jax.random.split(key, n))
    health = jnp.ones(8) if engine_health is None else engine_health

    def body(carry, _):
        states, done_prev, succ = carry
        action = act(states) * health
        nxt, np_, _, done = jax.vmap(lambda s, a: env.step(s, a))(states, action)
        newly = done & (~done_prev)
        ok = ((np_[:, 5] >= env.SAFE_Z_VELOCITY)
              & (jnp.linalg.norm(np_[:, 3:5], axis=1) <= env.SAFE_XY_VELOCITY)
              & (np_[:, 6] > 0.95)
              & (jnp.linalg.norm(np_[:, 0:2], axis=1) < env.PAD_RADIUS)
              & (np_[:, 2] <= 502.0))
        succ = jnp.where(newly, ok, succ)
        return (nxt, done_prev | done, succ), None

    init = (states, jnp.zeros(n, bool), jnp.zeros(n, bool))
    (_, _, succ), _ = jax.lax.scan(body, init, None, length=env.MAX_STEPS)
    return float(jnp.mean(succ))

# Combined robustness figure
def robustness_panel(params, n=2048, fname="figures/robustness.png"):
    key = jax.random.PRNGKey(0)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True)

    scales = np.linspace(0.6, 1.4, 9)
    grav = [evaluate_success(params, key, n, 2, gravity_scale=float(s))[0] for s in scales]
    drag = [evaluate_success(params, key, n, 2, drag_scale=float(s))[0] for s in scales]
    a = axes[0]
    a.plot(scales * 100, np.array(grav) * 100, "-o", label="gravity perturbed")
    a.plot(scales * 100, np.array(drag) * 100, "-s", label="drag perturbed")
    a.axvline(100, color="k", ls="--", lw=1)
    a.text(101, 4, "nominal", fontsize=16, va="bottom", ha="left", color="k")
    a.set_xlabel("parameter relative to training [%]"); a.set_ylabel("success rate [%]")
    a.set_ylim(-3, 105); a.set_title("Gravity / drag scaling (Stage 2)")
    a.grid(alpha=0.3); a.legend(loc="lower center")

    rl = lambda s: ACTOR.apply({"params": params}, v_observe(s))[0]
    healths = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    lost = [(1 - h) * 100 for h in healths]
    b = axes[1]
    for stage in (1, 2, 3):
        sr = [_eval_success(rl, key, 512, stage, engine_health=jnp.ones(8).at[0].set(h))
              for h in healths]
        b.plot(lost, np.array(sr) * 100, "-o", label=f"stage {stage}")
        print(f"engine-fail stage {stage}: {[round(v,2) for v in sr]}")
    b.set_xlabel("thrust lost on one engine [%]"); b.set_ylabel("success rate [%]")
    b.set_ylim(-3, 105); b.set_title("Single-engine thrust loss (zero-shot)")
    b.grid(alpha=0.3); b.legend(loc="upper right")

    fig.suptitle("Post-training robustness to off-nominal conditions")
    _save(fig, fname, pdf=True)

def ablation_comparison(fname="figures/ablation_comparison.png"):
    
    conditions = [
        ("full method",       "logs/training_log_seed*.csv",                "tab:blue"),
        ("no envelope",       "logs/training_log_abl_noenvelope_seed*.csv", "tab:red"),
        ("no lateral term",   "logs/training_log_abl_nolateral_seed*.csv",  "tab:orange"),
        ("no adaptive curr.", "logs/training_log_abl_noadaptive_seed*.csv", "tab:green"),
        ("no gusts",          "logs/training_log_abl_nogusts_seed*.csv",    "tab:purple"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for ax, stage in zip(axes, [1, 2, 3]):
        for label, pat, col in conditions:
            paths = sorted(glob.glob(pat))
            curves = []
            for p in paths:
                r = _read_log(p)
                if "Update" in r and f"Succ_S{stage}" in r:
                    curves.append((r["Update"], r[f"Succ_S{stage}"]))
            if not curves:
                continue
            n = min(len(u) for u, _ in curves)
            x = curves[0][0][:n]
            S = np.stack([s[:n] for _, s in curves])
            m = S.mean(0); sd = S.std(0)
            ax.plot(x, m, lw=2, color=col, label=label)
            if len(curves) > 1:
                ax.fill_between(x, m - sd, m + sd, alpha=0.18, color=col)
        ax.set_title(f"Stage {stage}")
        ax.set_xlabel("update"); ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("success rate")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=18)
    fig.suptitle("Ablation study - per-stage success across training", fontsize=23)
    _save(fig, fname, pdf=True)
    print(f"saved {fname}")

def sensitivity_curves(fname="figures/sensitivity.png"):

    full = "logs/training_log_seed1.csv"
    sweeps = [
        ("envelope coef (MARS_ENVELOPE_COEF)", [
            ("0.00", "logs/training_log_abl_noenvelope_seed1.csv"),
            ("0.05", "logs/training_log_sens_env_005.csv"),
            ("0.10", "logs/training_log_sens_env_010.csv"),
            ("0.15 (used)", full),
            ("0.30", "logs/training_log_sens_env_030.csv"),
        ]),
        ("discount factor (MARS_GAMMA)", [
            ("0.970", "logs/training_log_sens_gam_097.csv"),
            ("0.990 (used)", full),
            ("0.997\n(recomm.)", "logs/training_log_sens_gam_0997.csv"),
        ]),
        ("lateral coef (MARS_LATERAL_COEF)", [
            ("0.0", "logs/training_log_abl_nolateral_seed1.csv"),
            ("1.0", "logs/training_log_sens_lat_10.csv"),
            ("3.0 (used)", full),
            ("5.0", "logs/training_log_sens_lat_50.csv"),
        ]),
    ]

    fig, axes = plt.subplots(len(sweeps), 1, figsize=(10, 4.0 * len(sweeps)), constrained_layout=True)
    for ax, (title, vals) in zip(axes, sweeps):
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(vals)))
        for (val_label, path), c in zip(vals, cmap):
            if not os.path.exists(path):
                continue
            r = _read_log(path)
            if "Update" in r and "Succ_S3" in r:
                ax.plot(r["Update"], r["Succ_S3"], lw=1.8, color=c, label=val_label)
        ax.set_xlabel("update"); ax.set_ylabel("stage-3 success rate")
        ax.set_title(title); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=14, title="value")

    fig.suptitle("Sensitivity analysis - stage-3 success across training", fontsize=23)
    _save(fig, fname, pdf=True)
    print(f"saved {fname}")

def _draw_trajectory_3d(ax, h, stage=None, show_legend=True, compact=False, draw_title=True, baseline=None):

    x, y, z = h["x"], h["y"], h["z"]
    data_max = float(np.abs(np.concatenate([x, y])).max())
    step = 150.0 if data_max <= 330 else 200.0
    ntick = max(1, int(np.floor(data_max / step)))
    xyticks = step * np.arange(-ntick, ntick + 1)
    lim = float(max(data_max * 1.15, abs(xyticks[-1]) * 1.35, env.PAD_RADIUS * 3))

    g = np.linspace(-lim, lim, 130)
    X, Y = np.meshgrid(g, g)
    Z = np.array(env.get_terrain_elevation(jnp.array(X), jnp.array(Y)))
    z_floor = float(min(Z.min(), z.min())) - 30.0
    z_top = float(z.max()) * 1.02

    ax.plot_surface(X, Y, Z, cmap="copper", alpha=0.45, edgecolor="none", rstride=3, cstride=3, zorder=0)
    th = np.linspace(0, 2 * np.pi, 80)

    ax.plot(env.PAD_RADIUS * np.cos(th), env.PAD_RADIUS * np.sin(th), np.full_like(th, PAD_Z), color="red", lw=2)
    proj = dict(color="0.45", lw=1.4, alpha=0.9)
    ax.plot(np.full_like(x, -lim), y, z, **proj)
    ax.plot(x, np.full_like(y, lim), z, **proj)
    ax.plot(x, y, np.full_like(z, z_floor), **proj)
    ax.plot(x, y, z, color="tab:blue", lw=2.5, zorder=5, label="trajectory")

    ax.scatter(x[0], y[0], z[0], color="black", s=40, zorder=6)
    ax.scatter(x[-1], y[-1], z[-1], color="crimson", marker="X", s=80, zorder=6)

    if baseline is not None:
        bx, by, bz = baseline["x"], baseline["y"], baseline["z"]
        inb = (np.abs(bx) <= lim) & (np.abs(by) <= lim) & (bz >= z_floor) & (bz <= z_top)
        ax.plot(bx[inb], by[inb], bz[inb], color="darkorange", lw=2.0, ls="--", zorder=4, label="baseline")
        
        ixy = (np.abs(bx) <= lim) & (np.abs(by) <= lim)
        ax.plot(bx[ixy], by[ixy], np.full(int(ixy.sum()), z_floor), color="darkorange", lw=1.2, ls=":", alpha=0.85, zorder=3)

        if inb.any():
            ax.scatter(bx[inb][-1], by[inb][-1], bz[inb][-1], color="darkorange", marker="X", s=55, zorder=6)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(z_floor, z_top)
    ax.set_xlabel("x [m]", labelpad=2, fontsize=13)
    ax.set_ylabel("y [m]", labelpad=2, fontsize=13)
    ax.set_zlabel("altitude [m]", labelpad=16, fontsize=13)
    ax.tick_params(axis="both", labelsize=9, pad=0)
    ax.tick_params(axis="z", pad=7)          
    ax.set_xticks(xyticks); ax.set_yticks(xyticks)
    ax.zaxis.set_major_locator(plt.MaxNLocator(6))
    
    try:
        ax.set_box_aspect(None, zoom=0.9)
    except TypeError:
        pass

    if draw_title:
        tsize = 18 if not compact else 16
        if compact:
            ax.set_title(f"Stage {stage}" if stage is not None else "Descent trajectory", fontsize=tsize)
        else:
            ax.set_title(f"Descent trajectory - stage {stage}" if stage is not None
                         else "Descent trajectory", fontsize=tsize)
            
    ax.view_init(elev=24, azim=-60)
    if show_legend:
        ax.legend(handles=[Line2D([0], [0], color="tab:blue", lw=2.5, label="trajectory"), Line2D([0], [0], color="0.45", lw=1.4, label="projections"), Line2D([0], [0], color="red", lw=2, label="landing pad")], loc="upper left", bbox_to_anchor=(0.0, 0.95), fontsize=16, framealpha=0.85)

def plot_trajectory_3d(h, fname="figures/trajectory_3d.png", stage=None, baseline=None):

    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    _draw_trajectory_3d(ax, h, stage=stage, show_legend=False, draw_title=False, baseline=baseline)
    handles = [Line2D([0], [0], color="tab:blue", lw=2.5, label="PPO policy"), Line2D([0], [0], color="red", lw=2, label="landing pad")]
    
    if baseline is not None:
        handles.insert(1, Line2D([0], [0], color="darkorange", lw=2, ls="--", label="baseline"))

    fig.subplots_adjust(left=0.0, right=0.82, top=1.0, bottom=0.04)
    ttl = f"Descent trajectory - stage {stage}" if stage is not None else "Descent trajectory"
    fig.text(0.40, 0.90, ttl, ha="center", va="top", fontsize=15)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.40, 0.845), bbox_transform=fig.transFigure, ncol=len(handles), fontsize=9, framealpha=0.85, handlelength=1.4, columnspacing=1.1)
    _save(fig, fname, pdf=True, tight=False)

def plot_trajectory_3d_grid(stage_hists, fname="figures/trajectory3d_grid.png"):

    fig = plt.figure(figsize=(15, 12))
    for i, (s, h) in enumerate(stage_hists):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        _draw_trajectory_3d(ax, h, stage=s, show_legend=(i == 0), compact=True)

    fig.suptitle("Descent trajectories by curriculum stage (3D, with planar projections)", fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, fname, pdf=True)

def monte_carlo(params, key, n=1000, stage=1):
    states = jax.vmap(lambda k: env.reset(k, jnp.int32(stage), fixed_stage=jnp.int32(stage)))(
        jax.random.split(key, n))

    def body(carry, _):
        states, dprev, tx, ty, tvz, tsp, gnd = carry
        action, _ = ACTOR.apply({"params": params}, v_observe(states))
        nxt, p, _, done = jax.vmap(lambda s, a: env.step(s, a))(states, action)
        newly = done & (~dprev)
        ground = p[:, 2] <= 502.0
        tx = jnp.where(newly, p[:, 0], tx); ty = jnp.where(newly, p[:, 1], ty)
        tvz = jnp.where(newly, p[:, 5], tvz)
        tsp = jnp.where(newly, jnp.linalg.norm(p[:, 3:6], axis=1), tsp)
        gnd = jnp.where(newly, ground, gnd)
        return (nxt, done | dprev, tx, ty, tvz, tsp, gnd), None

    z = jnp.zeros(n)
    init = (states, jnp.zeros(n, bool), z, z, z, z, jnp.zeros(n, bool))
    (_, _, tx, ty, tvz, tsp, gnd), _ = jax.lax.scan(body, init, None, length=env.MAX_STEPS)

    return dict(x=np.array(tx), y=np.array(ty), vz=np.array(tvz), speed=np.array(tsp), grounded=np.array(gnd))

def plot_dispersion(mc, stage, fname="figures/dispersion.png"):

    g = mc["grounded"]
    x, y, vz = mc["x"][g], mc["y"][g], mc["vz"][g]
    n = len(x)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    a = axes[0]
    a.scatter(x, y, s=10, c="tab:blue", alpha=0.45, label="Monte Carlo")
    th = np.linspace(0, 2 * np.pi, 100)
    a.plot(env.PAD_RADIUS * np.cos(th), env.PAD_RADIUS * np.sin(th), "k--", lw=1.5, label=f"pad ({env.PAD_RADIUS:.0f} m)")
    
    if n > 5:
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * 3 * np.sqrt(np.maximum(vals, 1e-9))
        a.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, fill=False, edgecolor="red", lw=1.8, label="3$\\sigma$ ellipse"))
        
    a.scatter(0, 0, marker="*", color="gold", edgecolors="black", linewidths=1.0, s=240, zorder=7, label="target")
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.set_title("Terminal horizontal position"); a.grid(alpha=0.3)
    a.legend(loc="upper left", fontsize=16, framealpha=0.85)

    b = axes[1]
    b.scatter(np.arange(n), vz, s=10, c="tab:blue", alpha=0.45, label="Monte Carlo")
    b.axhline(env.SAFE_Z_VELOCITY, color="red", ls="--", lw=1.4)
    b.text(0, env.SAFE_Z_VELOCITY, " safe limit", color="red", fontsize=18, va="bottom", ha="left")

    on_pad = np.linalg.norm(np.c_[x, y], axis=1) < env.PAD_RADIUS
    succ = np.mean((vz >= env.SAFE_Z_VELOCITY) & on_pad) if n else 0.0
    b.set_xlabel("simulation #"); b.set_ylabel("terminal $v_z$ [m/s]")
    b.set_title("Terminal vertical velocity"); b.grid(alpha=0.3)

    fig.suptitle(f"Monte-Carlo landing dispersion - stage {stage}, {n} runs", fontsize=23)
    _save(fig, fname, pdf=True)

def verify_simulator(fname="figures/sim_verification.png"):
    g = 3.721
    base = jnp.array([0., 0., 1000., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., INITIAL_MASS])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)

    us = jnp.linspace(0, 1, 60)
    az = np.array(jax.vmap(lambda u: physics.calc_dynamics(base, jnp.ones(8) * u)[5])(us))
    dir_z = float(1 / np.sqrt(1 + 2 * np.sin(np.radians(15)) ** 2))
    u_hover = float(INITIAL_MASS * g / (8 * physics.T_MAX * dir_z))

    a = axes[0]
    a.plot(np.array(us) * 100, az, lw=2.8, label="simulator")
    a.axhline(0, color="k", lw=0.8)
    a.axvline(u_hover * 100, color="red", ls="--", lw=1.8, label=f"analytic hover\n{u_hover*100:.1f}%")
    a.set_xlabel("throttle [%]"); a.set_ylabel("vertical accel [m/s$^2$]")
    a.set_title("Thrust & hover point"); a.grid(alpha=0.3); a.legend(loc="upper left")

    sj = base
    tt, vzs = [], []
    for i in range(200):
        sj = physics.euler_step(sj, jnp.zeros(8), env.DT)
        tt.append(i * env.DT); vzs.append(float(sj[5]))

    tt = np.array(tt); vzs = np.array(vzs)
    analytic_v = -g * tt
    err_v = float(np.max(np.abs(vzs - analytic_v)))

    b = axes[1]
    b.plot(tt, vzs, lw=2.8, color="tab:blue", label="simulator")
    b.plot(tt[::14], analytic_v[::14], "o", color="red", ms=10, label="analytic $-g\\,t$")
    b.set_xlabel("time [s]"); b.set_ylabel("$v_z$ [m/s]")
    b.set_title("Free fall"); b.grid(alpha=0.3); b.legend(loc="lower left")
    b.text(0.5, 0.93, f"max error = {err_v:.0e} m/s", transform=b.transAxes, fontsize=15, va="top", ha="center", color="0.35")

    speeds = np.linspace(0, 120, 60)
    rho = float(physics.calc_density(1000.0))
    drag_an = 0.5 * rho * speeds ** 2 * physics.CD * physics.AREA / INITIAL_MASS
    sim_drag = np.array([(float(physics.calc_dynamics(base.at[5].set(-v), jnp.zeros(8))[5]) + g) for v in speeds])
    err_d = float(np.max(np.abs(sim_drag - drag_an)))

    c = axes[2]
    c.plot(speeds, sim_drag, lw=2.8, color="tab:blue", label="simulator")
    c.plot(speeds[::6], drag_an[::6], "o", color="red", ms=10, label="analytic $\\frac{1}{2}\\rho v^2 C_d A/m$")
    c.set_xlabel("speed [m/s]"); c.set_ylabel("drag decel. [m/s$^2$]")
    c.set_title(f"Drag model ($\\rho$={rho:.4f} kg/m$^3$)")
    c.grid(alpha=0.3); c.legend(loc="upper left")
    c.text(0.97, 0.05, f"max error = {err_d:.0e}", transform=c.transAxes, fontsize=15, va="bottom", ha="right", color="0.35")

    fig.suptitle("Simulator verification")
    _save(fig, fname, pdf=True)

def neuron_analysis(params, fname="figures/neuron_maps.png", n_lesion=1024):
    W0, b0 = params['Dense_0']['kernel'], params['Dense_0']['bias']
    W1, b1 = params['Dense_1']['kernel'], params['Dense_1']['bias']
    W2 = np.asarray(params['Dense_2']['kernel']); b2 = params['Dense_2']['bias']

    alts = np.linspace(0, 150, 90); vzs = np.linspace(-15, 5, 90)
    AA, VV = np.meshgrid(alts, vzs)
    raw = jax.vmap(_obs_vec)(jnp.array(AA.ravel()), jnp.array(VV.ravel()))
    x = agent.make_obs(raw)
    h1 = np.array(jnp.tanh(jnp.tanh(x @ W0 + b0) @ W1 + b1))   # last hidden layer
    infl = np.abs(W2.mean(axis=1)); meanw = W2.mean(axis=1)      # influence on collective throttle
    order = np.argsort(infl)[::-1]
    show = [next(j for j in order if meanw[j] > 0),              # top throttle-raising unit
            next(j for j in order if meanw[j] < 0)]              # top throttle-cutting unit
    
    top = order[:4]                                             
    v_env = -np.sqrt(2 * env.A_BRAKE * alts); m = v_env >= vzs.min()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, j in zip(axes, show):
        pc = ax.pcolormesh(AA, VV, h1[:, j].reshape(AA.shape), cmap="coolwarm", vmin=-1, vmax=1, shading="auto")
        ax.plot(alts[m], v_env[m], "k--", lw=2)
        mid = len(alts[m]) // 2
        ax.text(alts[m][mid] + 4, v_env[m][mid], "$-v_{\\mathrm{safe}}$", fontsize=12, color="black", va="center", ha="left", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        role = "raises throttle" if meanw[j] > 0 else "cuts throttle"
        ax.set_title(f"neuron {j}: {role}", fontsize=16); ax.set_xlabel("altitude above pad [m]")

    axes[0].set_ylabel("vertical velocity $v_z$ [m/s]")
    fig.colorbar(pc, ax=axes, shrink=0.85, label="neuron activation")
    fig.suptitle("Most throttle-influential hidden neurons vs. the braking envelope (dashed $-v_{\\mathrm{safe}}$)",
                 fontsize=18)
    _save(fig, fname, pdf=True, pad=0.05)

    # lesioning: zero the four neurons above and re-measure success
    Wj = [jnp.array(v) for v in (W0, b0, W1, b1, params['Dense_2']['kernel'], b2)]

    def mean_fn(obs16, mask):
        xx = agent.make_obs(obs16)
        hh = jnp.tanh(jnp.tanh(xx @ Wj[0] + Wj[1]) @ Wj[2] + Wj[3]) * mask
        return 0.5 * (jnp.tanh(hh @ Wj[4] + Wj[5]) + 1)
    
    def succ(mask, stage=2, n=n_lesion):
        states = jax.vmap(lambda k: env.reset(k, jnp.int32(stage), fixed_stage=jnp.int32(stage)))(
            jax.random.split(jax.random.PRNGKey(0), n))
        def body(c, _):
            st, dp, sc = c
            a = mean_fn(v_observe(st), mask)
            nxt, np_, _, d = jax.vmap(lambda s, aa: env.step(s, aa))(st, a)
            newly = d & (~dp)
            ok = ((np_[:, 5] >= env.SAFE_Z_VELOCITY)
                  & (jnp.linalg.norm(np_[:, 3:5], axis=1) <= env.SAFE_XY_VELOCITY)
                  & (np_[:, 6] > 0.95) & (jnp.linalg.norm(np_[:, 0:2], axis=1) < env.PAD_RADIUS)
                  & (np_[:, 2] <= 502.0))
            sc = jnp.where(newly, ok, sc)
            return (nxt, dp | d, sc), None
        
        init = (states, jnp.zeros(n, bool), jnp.zeros(n, bool))
        (_, _, sc), _ = jax.lax.scan(body, init, None, length=env.MAX_STEPS)
        return float(jnp.mean(sc))
    
    full = jnp.ones(256)
    base = succ(full); les = succ(full.at[jnp.array(top)].set(0.0))
    print(f"lesion stage-2: intact={base:.1%}, top-4 neurons {top.tolist()} zeroed={les:.1%}")
    return base, les

if __name__ == "__main__":

    params = load_actor()
    verify_simulator()
    generate_training_plots()
    state_action_map(params)

    stage_hists = []

    for s in range(4):
        h = run_flight(params, stage=s, seed=7 + s)
        bl = run_flight(params, stage=s, seed=7 + s, controller=baseline_controller)
        stage_hists.append((s, h))

        plot_trajectory_report(h, fname=f"figures/trajectory_stage{s}.png", stage=s)
        plot_trajectory_3d(h, fname=f"figures/trajectory3d_stage{s}.png", stage=s, baseline=bl)

    plot_trajectory_3d_grid(stage_hists)
    plot_trajectory_dashboard(run_flight(params, stage=1, seed=7))

    for s in (1, 2, 3):
        mc = monte_carlo(params, jax.random.PRNGKey(0), n=1000, stage=s)
        plot_dispersion(mc, s, fname=f"figures/dispersion_stage{s}.png")

    robustness_panel(params)
    neuron_analysis(params)