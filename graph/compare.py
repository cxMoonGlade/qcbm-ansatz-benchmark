#!/usr/bin/env python3
"""
10k-step training report:
- Curves: loss, MMD, KL, grad_norm, step time
- Param drift norm
- Loss landscape (high-curvature slice, late checkpoint)
- Gradient-norm (log10) map & 3D surface on low-curvature slice (early checkpoint)

Usage:
  python make_10k_report.py
  (edit RESULTS_ROOT / RUN_NAME below, or pass --run_dir)
"""

from __future__ import annotations
from pathlib import Path
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp
import optax
import pandas as pd
from itertools import product
from matplotlib import cm
from matplotlib.colors import Normalize

jax.config.update("jax_enable_x64", True)

# ----------------- CONFIG -----------------
# If you prefer to hardcode, set RUN_NAME="resultXYZ"
RESULTS_ROOT = Path("/home/cx/Documents/qcbm-ansatz-benchmark/data/results/Qubits8")
RUN_NAME     = "result4"  # e.g. "result4"; if None auto-picks newest valid

# Grid sizes for landscapes
N_GRID   = 151
LOWCURV_CANDIDATES = 512

# -----------------------------------------

def pick_run(root: Path, run_name: str | None):
    if run_name:
        d = root / run_name
        if not d.exists(): raise FileNotFoundError(d)
        return d
    cands = [p for p in root.glob("result*") if p.is_dir()]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in cands:
        if (d/"params.npy").exists() or (d/"final_params.npy").exists():
            return d
            \
    raise FileNotFoundError(f"No valid runs in {root}")

run_dir = pick_run(RESULTS_ROOT, RUN_NAME)
out_dir = run_dir / "figures_10k"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[info] Using run_dir = {run_dir}")

# ---------- helpers ----------
def load(path: Path):
    return np.load(path, mmap_mode="r") if path.exists() else None

def ensure_grad_norm(run: Path):
    gnorm = load(run/"grad_norm.npy")
    if gnorm is not None: return np.asarray(gnorm)
    grads = load(run/"grads.npy")
    if grads is None: return None
    arr = np.asarray(grads)
    if arr.ndim >= 2:
        return np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
    return None

def align_min_len(*arrays):
    avail = [a for a in arrays if a is not None]
    L = min(len(a) for a in avail) if avail else 0
    return tuple(None if a is None else np.asarray(a)[:L] for a in arrays)

def moving_avg(x, w=50):
    if x is None: return None
    w = int(max(1, w))
    if w >= len(x): return x
    c = np.convolve(x, np.ones(w), 'valid')/w
    pad = np.full(w-1, c[0])
    return np.concatenate([pad, c])

def line_plot(x, ys, labels, title, ylabel, fname, ylog=False):
    plt.figure(figsize=(9,5))
    for y, lab in zip(ys, labels):
        if y is not None: plt.plot(x, y, label=lab)
    if ylog: plt.yscale("log")
    plt.xlabel("Step"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir/fname, dpi=160); plt.close()
    print("[saved]", out_dir/fname)

# ---------- load logs ----------
loss = load(run_dir/"loss.npy")
mmd  = load(run_dir/"mmd.npy")
kl   = load(run_dir/"kl.npy")
step_time = load(run_dir/"step_time.npy")
params_hist = load(run_dir/"params.npy")  # (steps, [...], D)
final_params = load(run_dir/"final_params.npy")
grad_norm = ensure_grad_norm(run_dir)

# align series lengths (use min length across available arrays)
loss, mmd, kl, step_time, grad_norm = align_min_len(loss, mmd, kl, step_time, grad_norm)
steps = np.arange(len(loss) if loss is not None else len(grad_norm))

# ---------- training curves ----------
line_plot(steps, [loss, moving_avg(loss, 50)], ["loss", "loss (MA50)"],
          "Loss vs step", "Loss", "loss_curve.png", ylog=False)

line_plot(steps, [mmd, kl], ["MMD", "KL"], "MMD & KL vs step",
          "Value", "mmd_kl_curve.png", ylog=False)

if grad_norm is not None:
    log10g = np.log10(grad_norm + 1e-12)
    line_plot(steps, [log10g, moving_avg(log10g, 50)], ["log10||grad||", "MA50"],
              "Gradient magnitude over training", "log10 ||grad||", "gradnorm_log10_curve.png")
else:
    print("[warn] No grad_norm/grads found; skipping gradient curve.")

# Step time (detect units)
if step_time is not None:
    st = np.asarray(step_time, dtype=float)
    # heuristic: if mostly < 5 => seconds; convert to ms
    if np.nanmedian(st) < 5.0: st_ms = st * 1000.0
    else: st_ms = st
    line_plot(np.arange(len(st_ms)),
              [st_ms, moving_avg(st_ms, 50)],
              ["step time (ms)", "MA50"],
              "Step time per iteration", "ms", "step_time_ms.png")
else:
    print("[warn] No step_time.npy found; skipping time plot.")

# Param drift norm (from first step)
if params_hist is not None:
    PH = np.asarray(params_hist)
    if PH.ndim == 3: PH = PH[:,0,:]     # (steps, D)
    base = PH[0]
    drift = np.linalg.norm(PH - base, axis=1)
    line_plot(np.arange(len(PH)), [drift], ["||params - init||2"],
              "Parameter drift vs step", "L2 drift", "param_drift.png")
else:
    print("[warn] No params.npy found; skipping drift plot.")

# ================= Landscapes & plateau map =================
# Build model so we can evaluate loss/grad on slices
# (same target_probs pipeline you already use)
# -- find repo root containing 'src'
try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd()
    for p in [REPO_ROOT, *REPO_ROOT.parents]:
        if (p/"src").exists(): REPO_ROOT = p; break
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from src.train.mmd2 import build_mmdagg_prob
from src.train.qcbm import QCBM
from src.circuits.ansatz1 import hardware_efficient_ansatz  # change if needed

# Build target_probs (Qubits8)
df = pd.read_csv(REPO_ROOT / "data_2d" / "Qubits8" / "train.csv")
n_bits = 8
bit_cols = [f"q{i}" for i in range(n_bits)]
bitstrings = df[bit_cols].astype(str).agg("".join, axis=1)
counts = bitstrings.value_counts().sort_index()
all_bits = ["".join(seq) for seq in product("01", repeat=n_bits)]
probs_full = pd.Series(0.0, index=all_bits, dtype=float)
probs_full.update(counts / counts.sum())

devices = jax.devices("gpu")
device = devices[0] if devices else jax.devices("cpu")[0]
target_probs = jax.device_put(jnp.asarray(probs_full.values, dtype=jnp.float64), device)

R, C = 2, 4
d = probs_full.values.size  # e.g., 256 for 8 qubits
mmd_eval = build_mmdagg_prob(
    d,
    kernel="laplace_gaussian",   # or "all", "all_matern_l1_l2", …
    number_bandwidths=10,
    weights_type="centred",      # nicer than uniform; feel free to change
    dtype=jnp.float64,
    return_details=False,        # we only need the scalar for viz
    use_sqrt=True,               # MMD (set False if you prefer MMD^2)
)
model = QCBM(
    ansatz=hardware_efficient_ansatz,
    n_qubits=n_bits,
    L=R*C,
    mmd_fn=mmd_eval,             # ← pass the evaluator, not the builder
    target_probs=target_probs,
)
model.build_circuits()

def loss_scalar(p):
    val, _ = model.loss(p)
    return val
loss_scalar = jax.jit(loss_scalar)

# hvp / direction pickers
def hvp_at(p, v): return jax.jvp(jax.grad(loss_scalar), (p,), (v,))[1]

def pick_hessian_top(p, key, iters=30):
    def power(key, ortho=None):
        v = jax.random.normal(key, p.shape)
        if ortho is not None: v = v - jnp.dot(v, ortho)*ortho
        v = v / (jnp.linalg.norm(v)+1e-12)
        for _ in range(iters):
            w = hvp_at(p, v)
            if ortho is not None: w = w - jnp.dot(w, ortho)*ortho
            v = w / (jnp.linalg.norm(w)+1e-12)
        return v
    k1,k2 = jax.random.split(key)
    U = power(k1); V = power(k2, U)
    return U, V

def pick_low_curv(p, key, candidates=512):
    vs = jax.random.normal(key, (candidates,) + p.shape)
    vs = vs / (jnp.linalg.norm(vs, axis=tuple(range(1, vs.ndim)), keepdims=True)+1e-12)
    rq = jax.vmap(lambda v: jnp.dot(v, hvp_at(p, v)))(vs)
    idx = jnp.argsort(jnp.abs(rq))
    U = vs[idx[0]]
    V = vs[idx[1]] - jnp.dot(vs[idx[1]], U)*U
    U = U / (jnp.linalg.norm(U)+1e-12)
    V = V / (jnp.linalg.norm(V)+1e-12)
    return U, V

def make_slice(p0, mode="hessian-top", N=N_GRID, radius=None, key=0, return_dirs=False):
    key = jax.random.PRNGKey(key)
    if mode == "hessian-top":
        U, V = pick_hessian_top(p0, key)
        cu = float(jnp.dot(U, hvp_at(p0, U))); cv = float(jnp.dot(V, hvp_at(p0, V)))
        rad_u = min(6.0, 2.0/(abs(cu)**0.5 + 1e-6)) if radius is None else radius
        rad_v = min(6.0, 2.0/(abs(cv)**0.5 + 1e-6)) if radius is None else radius
    elif mode == "low-curv":
        U, V = pick_low_curv(p0, key, candidates=LOWCURV_CANDIDATES)
        s = float(jnp.std(p0)) or 1.0
        rad_u = rad_v = (4.0*s if radius is None else radius)
    else:
        raise ValueError
    a = jnp.linspace(-rad_u, rad_u, N); b = jnp.linspace(-rad_v, rad_v, N)
    A, B = jnp.meshgrid(a, b, indexing="ij")
    def to_params(alpha_beta):
        a,b = alpha_beta; return p0 + a*U + b*V
    @jax.jit
    def eval_batch(ab_batch):
        Ps = jax.vmap(to_params)(ab_batch)
        return jax.vmap(loss_scalar)(Ps)
    AB = jnp.stack([A.ravel(), B.ravel()], axis=1)
    Z = eval_batch(AB).reshape(N, N)
    if return_dirs:
        return np.asarray(A), np.asarray(B), np.asarray(Z), U, V
    return np.asarray(A), np.asarray(B), np.asarray(Z)

def surface3d(A, B, Z, title, fname, zlabel):
    A = np.asarray(A); B = np.asarray(B); Z = np.asarray(Z)
    fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(Z.min()), vmax=float(Z.max()))
    colors = cm.viridis(norm(Z))
    ax.plot_surface(A, B, Z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    m = cm.ScalarMappable(norm=norm, cmap="viridis"); m.set_array(Z)
    cbar = fig.colorbar(m, ax=ax, shrink=0.65, aspect=16, pad=0.1); cbar.set_label(zlabel)
    ax.set_xlabel("α (dir 1)"); ax.set_ylabel("β (dir 2)"); ax.set_zlabel(zlabel)
    ax.set_title(title); plt.tight_layout(); plt.savefig(out_dir/fname, dpi=170); plt.close(fig)
    print("[saved]", out_dir/fname)

def contour(A, B, Z, title, fname, label="Value", vmin=None, vmax=None, mark_center=True):
    ij_min = np.unravel_index(np.argmin(Z), Z.shape)
    ij_max = np.unravel_index(np.argmax(Z), Z.shape)
    plt.figure(figsize=(8,7))
    cf = plt.contourf(A, B, Z, levels=50, vmin=vmin, vmax=vmax)
    plt.colorbar(cf, label=label)
    if mark_center: plt.scatter([0],[0], s=40, facecolors="none", edgecolors="w")
    plt.scatter(A[ij_min], B[ij_min], marker="v", s=80)
    plt.scatter(A[ij_max], B[ij_max], marker="^", s=80)
    plt.xlabel("α (dir 1)"); plt.ylabel("β (dir 2)")
    plt.title(title); plt.tight_layout(); plt.savefig(out_dir/fname, dpi=170); plt.close()
    print("[saved]", out_dir/fname)

# Choose checkpoints
if params_hist is not None:
    PH = np.asarray(params_hist)
    if PH.ndim == 3: PH = PH[:,0,:]
    p0_plateau = jnp.asarray(PH[0])                          # early
    p0_mountain = jnp.asarray(PH[int(0.9*len(PH))-1])        # late
else:
    if final_params is None: raise RuntimeError("Need params.npy or final_params.npy for landscapes.")
    p0_plateau = jnp.asarray(final_params)   # fallback
    p0_mountain = jnp.asarray(final_params)

# High-curvature loss (late)
A_m, B_m, Z_m = make_slice(p0_mountain, mode="hessian-top", N=N_GRID)
contour(A_m, B_m, Z_m, "Loss landscape — high-curvature (late)", "landscape_hcurv_contour.png", "Loss")
surface3d(A_m, B_m, Z_m, "Loss surface — high-curvature (late)", "surface_loss_hcurv.png", "Loss")

# Low-curvature gradient-norm (early)
A_p, B_p, _, U_p, V_p = make_slice(p0_plateau, mode="low-curv", N=N_GRID, return_dirs=True)

def to_params_plateau(ab):
    a,b = ab; return p0_plateau + a*U_p + b*V_p

loss_jit = jax.jit(loss_scalar)
grad_rms_fn = jax.jit(lambda p: optax.global_norm(jax.grad(loss_jit)(p)) / jnp.sqrt(p.size))

@jax.jit
def eval_grad(ab_batch):
    Ps = jax.vmap(to_params_plateau)(ab_batch)
    return jax.vmap(grad_rms_fn)(Ps)

ABp = jnp.stack([jnp.asarray(A_p).ravel(), jnp.asarray(B_p).ravel()], axis=1)

# chunk to be safe
def run_chunks(fn, X, chunk=8000):
    outs = []
    for s0 in range(0, X.shape[0], chunk):
        outs.append(fn(X[s0:s0+chunk]))
    return jnp.concatenate(outs)

G = run_chunks(eval_grad, ABp).reshape(A_p.shape)
Glog = np.log10(np.asarray(G)+1e-12)

# stats
ctr = (A_p.shape[0]//2, A_p.shape[1]//2)
print(f"[plateau slice @ init] center log10||g||_RMS={Glog[ctr]:.3f}, "
      f"median={np.median(Glog):.3f}")
for thr in (1e-3,1e-5,1e-7):
    print(f"  frac(RMS<{thr:g}) = {float((G<thr).mean()):.4f}")

contour(A_p, B_p, Glog, "Gradient RMS (log10) — low-curvature (early)", "plateau_grad_rms_log10_contour.png",
        label="log10 ||∇Loss||_RMS")
surface3d(A_p, B_p, Glog, "Gradient RMS (log10) surface — low-curvature (early)",
          "surface_plateau_grad_rms_log10.png", "log10 ||∇Loss||_RMS")
