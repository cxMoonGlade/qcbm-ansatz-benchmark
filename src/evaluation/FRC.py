# src/evaluation/FRC.py
import os, sys, json, itertools
from pathlib import Path

import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

# --- repo root (script or notebook) ---
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    REPO_ROOT = Path.cwd()
    for p in [REPO_ROOT, *REPO_ROOT.parents]:
        if (p / "src").exists():
            REPO_ROOT = p
            break
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- layout ---
DATA_ROOT    = REPO_ROOT / "data_2d" / "Qubits8"
RESULTS_ROOT = REPO_ROOT / "data" / "results" / "Qubits8"
train_csv    = DATA_ROOT / "train.csv"
shots        = 10_000
n_qubits     = 8
d            = 2 ** n_qubits
bit_cols     = [f"q{i}" for i in range(n_qubits)]

# --- project imports ---
from src.circuits.specs import get_ansatz_spec   # <— new shared spec
from src.train.qcbm import QCBM
from src.train.mmd2 import build_mmdagg_prob     # MMD used only to eval loss

# --- helpers ---
def all_bitstrings(n): return [''.join(map(str, t)) for t in itertools.product([0,1], repeat=n)]
def bits_to_strings(arr01): return [''.join(map(str, row)) for row in np.asarray(arr01, dtype=int)]

def load_params_any_shape(run_dir: Path) -> np.ndarray:
    """
    Return numpy array of shape:
      (P,), (B,P), (T,P) or (T,B,P). We pick the final time slice if T exists.
    """
    if (run_dir / "final_params.npy").exists():
        arr = np.load(run_dir / "final_params.npy")
    elif (run_dir / "params.npy").exists():
        arr = np.load(run_dir / "params.npy", mmap_mode="r")  # (T,P) or (T,B,P)
        arr = arr[-1]  # final time slice
    else:
        raise FileNotFoundError(f"No params found in {run_dir}")
    return np.asarray(arr)

def pick_best_params(model: QCBM, arr: np.ndarray) -> jnp.ndarray:
    """
    Resolve arr to a single (P,) jnp.float64 by:
      - If (T,P): take last.
      - If (B,P): evaluate loss for each row and take argmin.
      - If (T,B,P): we already reduced to last T, so treat as (B,P).
      - If already (P,): just cast.
    """
    x = arr
    if x.ndim == 1:  # (P,)
        return jnp.asarray(x, dtype=jnp.float64)

    if x.ndim == 2:  # (B,P)
        losses = []
        for b in range(x.shape[0]):
            p = jnp.asarray(x[b], dtype=jnp.float64)
            L, _ = model.loss(p)
            losses.append(float(L))
        best = int(np.argmin(losses))
        return jnp.asarray(x[best], dtype=jnp.float64)

    raise ValueError(f"Unsupported param shape after time-slice: {x.shape}")

# --- load data & target probs (for loss eval) ---
train_df = pd.read_csv(train_csv)
S_set = set(all_bitstrings(n_qubits))
D_set = set(train_df[bit_cols].astype(str).agg("".join, axis=1).str.strip())

# target probabilities from train data (same as training target)
from itertools import product
all_bits = [''.join(bs) for bs in product('01', repeat=n_qubits)]
counts = train_df[bit_cols].astype(str).agg(''.join, axis=1).value_counts().sort_index()
probs_full = pd.Series(0.0, index=all_bits, dtype=float)
probs_full.update(counts / counts.sum())
target_probs = jnp.asarray(probs_full.values, dtype=jnp.float64)

# MMD evaluator (float64) just for picking best params by loss
mmd_eval = build_mmdagg_prob(
    d,
    kernel="laplace_gaussian",
    number_bandwidths=10,
    weights_type="centred",
    dtype=jnp.float64,
    return_details=False,
    use_sqrt=False,
)

# --- evaluate each result{1..4} ---
all_results = []
for ansatz_id in (1, 2, 3, 4):
    run_dir = RESULTS_ROOT / f"result{ansatz_id}"
    if not run_dir.exists():
        print(f"[warn] {run_dir} not found; skipping")
        continue

    # unified spec (ansatz, L, param_count) — ansatz4 uses keep_edges=16, extras=4
    ans_fn, L, pc, meta = get_ansatz_spec(
        ansatz_id=ansatz_id,
        n_bits=n_qubits,
        R=2, C=4,
        L1=4, L_M=3,
        keep_edges=16, extras=4,
        train_df=train_df, bit_cols=bit_cols
    )

    # build model first (needed to score batched params)
    model = QCBM(
        ansatz=ans_fn,
        n_qubits=n_qubits,
        L=L,
        mmd_fn=mmd_eval,
        target_probs=target_probs,
        shots=None,
        dtype=jnp.float64,
    )
    model.build_circuits()

    # load params array of any shape and pick the best single vector
    try:
        arr = load_params_any_shape(run_dir)
        params = pick_best_params(model, arr)  # (P,)
    except Exception as e:
        print(f"[warn] skip {run_dir}: {e}")
        continue

    # optional sanity check vs spec count (do not hard fail if mismatch)
    if params.ndim != 1 or params.shape[0] != pc:
        print(f"[warn] {run_dir.name}: param size {params.shape[-1]} != spec {pc}")

    # sample and compute F/R/C
    samples = np.asarray(model.circuit2(params))  # (shots, n_qubits)
    bs = bits_to_strings(samples)

    G_train = [b for b in bs if b in D_set]
    G_new   = [b for b in bs if b not in D_set]
    G_sol   = [b for b in G_new if b in S_set]
    g_sol   = set(G_sol)
    Q       = len(bs)

    e = len(D_set) / len(S_set) if len(S_set) else 0.0  # training coverage
    R = len(G_sol) / Q if Q else 0.0
    R_ = (1 - e)
    R_tilde = R / R_ if R_ > 0 else 0.0

    S_minus_D = len(S_set) - len(D_set)
    C = len(g_sol) / S_minus_D if S_minus_D > 0 else 0.0
    C_ = 1 - (1 - 1 / S_minus_D) ** Q if S_minus_D > 0 else 1.0
    C_tilde = C / C_ if C_ > 0 else 0.0

    F = len(G_sol) / len(G_new) if len(G_new) > 0 else 0.0

    res = {
        "ansatz": ansatz_id,
        "shots": shots,
        "Fidelity": F,
        "Rate": R_tilde,
        "Coverage": C_tilde,
        "counts": {
            "|G_train|": len(G_train),
            "|G_new|":   len(G_new),
            "|G_sol|":   len(G_sol),
            "|unique(G_sol)|": len(g_sol),
            "Q": Q
        }
    }
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    all_results.append(res)

    print(f"[ok] {run_dir.name} ({meta['name']}): "
          f"F={F:.3f}  R~={R_tilde:.3f}  C~={C_tilde:.3f}  "
          f"G_train={len(G_train)}  G_new={len(G_new)}  G_sol={len(G_sol)}")

# combined table
combined_path = RESULTS_ROOT.parent / "metrics.json"
combined_path.write_text(json.dumps(all_results, indent=2))
if all_results:
    df = pd.DataFrame(all_results).sort_values("ansatz")
    print("\n=== Metrics summary ===")
    print(df.to_string(index=False))
else:
    print("[warn] No results were evaluated.")
