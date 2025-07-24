## labraries for training

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import pennylane as qml
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


import sys

# 1. current file（train.py）abspath
current_file = os.path.abspath(__file__)

# 2. back to root path（report1）
report1_dir = os.path.abspath(os.path.join(current_file, "../../../"))
# 3. 把项目根路径加入 sys.path
if report1_dir not in sys.path:
    sys.path.insert(0, report1_dir)


from mmdagg_probs import mmdagg_prob
from qcbm import QCBM


## for mi ansatz
def mutual_information_matrix(bits: jnp.ndarray) -> jnp.ndarray:
    """
    bits : (N, n)  0/1 ndarray
    return: (n, n)  float64 JAX array,  diag -> 0
    """
    N, n = bits.shape
    bits = bits.astype(jnp.int32)                  # 确保 0/1 -> 0/1 int

    # --- 1. single qubit edge probs P(q_k = 1) ---------------------------------
    pk1 = bits.mean(axis=0)                       # (n,)  float64
    pk0 = 1.0 - pk1                               # (n,)

    # --- 2. 2 qubits unite probs P(q_i = a, q_j = b) ------------------------
    #    P11(i,j) = mean( bits[:,i] & bits[:,j] )
    bT        = bits.T                            # (n, N)
    P11       = (bT[:, None, :] & bT[None, :, :]).mean(axis=-1)  # (n, n)
    
    P10 = pk1[:, None] - P11                      # (n, n)
    P01 = pk1[None, :] - P11
    P00 = 1.0 - (P11 + P10 + P01)

    # --- 3. mutal info I(i,j) = Σ_{a,b∈{0,1}} P_ab log( P_ab / (P_a·P_b) ) --
    eps  = 1e-12
    P_ab = jnp.stack([P00, P01, P10, P11], axis=0)       # (4, n, n)
    logt = jnp.log( jnp.clip(P_ab, eps) )
    pk0_col = pk0[:, None]          # (n,1)
    pk1_col = pk1[:, None]          # (n,1)
    logm = jnp.log( jnp.clip(
        jnp.stack([pk0_col*pk0,
                pk0_col*pk1,
                pk1_col*pk0, 
                pk1_col*pk1],
                axis=0), eps) )
    Iij  = jnp.sum(P_ab * (logt - logm), axis=0)         # (n, n)

    # --- 4. diag reset & return jnp.ndarray -------------------------------
    Iij = Iij.at[jnp.diag_indices(n)].set(0.0)
    return Iij

# ------------  init target probs & model & params ------------
from itertools import product

import pandas as pd
data_path = os.path.join(report1_dir, "data_2d/Qubits12/train.csv")
df = pd.read_csv(data_path)
n_bits = 12
bit_cols = [f"q{i}" for i in range(n_bits)]
bitstrings = (
    df[bit_cols]
    .astype(str)
    .agg("".join, axis=1)
)
counts = bitstrings.value_counts().sort_index()
all_bits = ["".join(seq) for seq in product("01", repeat=n_bits)]
probs_full = pd.Series(0.0, index=all_bits, dtype=float)   # float64
probs_full.update(counts / counts.sum())                   # 归一化

gpu = jax.devices("gpu")[0]
target_probs = jax.device_put(jnp.asarray(probs_full.values, dtype=jnp.float64), gpu)

print("target_probs shape:", target_probs.shape,
      "dtype:", target_probs.dtype,
      "sum =", float(target_probs.sum()))

# ------------ parameter counts ------------
# P = 222, n = 12, L = 10
def count_params1(n_bits: int, L: int) -> int:
    """
    return params requested for ansatz1
    """
    assert L % 2 == 0, "for ansatz1, L must be even number"
    return int((3 * L / 2 + 1) * n_bits - (L / 2))

# R = 3, C = 4, PL = 45, L = 5, P = 225 
def count_params2(R: int, C: int, L: int, periodic: bool = False) -> int:
    n = R * C
    per_layer = 2 * n + (R * (C - 1) + C * (R - 1))
    if periodic:
        per_layer += R + C
    return per_layer * L

# R = 3, C = 4, PL = 41, L = 5, P = 205
def count_params3(R: int, C: int, L: int, add_dt: bool = False) -> int:
    n = R * C
    per_layer = 2*n + R*(C-1) + (R-1)*C + (1 if add_dt else 0)
    return per_layer * L

# n = 12, L = 5, keep_edges = 20, extras = 6, P = 226
def count_params4(n: int, L: int, keep_edges: int, extras: int = 6) -> int:
    return 2*L*n + L*keep_edges + extras  

# ------------ ansatz factory ------------

from src.circuits.ansatz1 import hardware_efficient_ansatz
from src.circuits.ansatz2 import ising_structured_ansatz
from src.circuits.ansatz3 import eh2d_ansatz
from src.circuits.ansatz4 import mi_ansatz
## Control # of Params around 100


ansatz = ising_structured_ansatz
n_qubits= 12
mmd_fn = mmdagg_prob
R = 3
C = 4
keep_edges = 20
L1 = 10
L2 = 5

def ansatz_set(ansatz):
    if ansatz == hardware_efficient_ansatz:
        pc = count_params1(n_bits, L1)
        L = L1
        id = 1

    if ansatz == ising_structured_ansatz:
        pc = count_params2(R, C, L2, False)
        L = L2
        id = 2

    if ansatz == eh2d_ansatz:
        pc = count_params3(R, C, L2)
        L = L2
        id = 3

    if ansatz == mi_ansatz:
        pc = count_params4(n_qubits, L2, keep_edges)
        L = L2
        id = 4
        bit_np = df[bit_cols].values
        mi_mat = mutual_information_matrix(bit_np)
        triu_i, triu_j = jnp.triu_indices(n_qubits, k=1)
        mi_flat   = mi_mat[triu_i, triu_j]
        top_idx   = jnp.argsort(-mi_flat)[:keep_edges]
        mi_edges  = [(int(triu_i[k]), int(triu_j[k])) for k in top_idx]  # [(i,j),...]
        def ansatz_mi(params, wires, *, L=None, **kw):
            return mi_ansatz(
                params, wires,
                mi_edges = mi_edges,
                L = L,
                **kw
            )
        ansatz = ansatz_mi
    return ansatz, L, pc, id
ansatz, L, pc, id = ansatz_set(ansatz)

model = QCBM(ansatz, n_bits, L, mmd_fn, target_probs)
key = jax.random.PRNGKey(0)
params = jax.random.normal(key, shape=(pc,))


# ------------ training set up ------------
import optax
import catalyst

# opt = optax.adam(1e-2)

lr_sched = optax.exponential_decay(  # 从 1e‑2 → 每 200 步 × 0.9
    init_value=1e-2, transition_steps=200, decay_rate=0.9, staircase=True
)
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_sched, b2=0.999)
)


def update_step(i, params, opt_state, loss_log):
    loss_val, grads = catalyst.value_and_grad(model.loss)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    loss_log = loss_log.at[i].set(loss_val)
    grad_norm = optax.global_norm(grads)
    catalyst.debug.print("Step {i}: loss = {loss}, grad_norm = {g}", i=i, loss=loss_val, g=grad_norm)

    return (params, opt_state, loss_log)

@qml.qjit     
def optimization(params, n_steps: int = 1000):
    opt_state = opt.init(params)
    loss_log  = jnp.zeros(n_steps, dtype=params.dtype)
    params, opt_state, loss_log = qml.for_loop(
        0, n_steps, 1
    )(update_step)(params, opt_state, loss_log)
    return params, loss_log


# ------------ training steps ------------

trained_params, loss_hist = optimization(params)

jax.block_until_ready(trained_params)      # wait until training ends

# ───── save ─────
import numpy as np
from pathlib import Path
run_id = f"result{id}"


out_dir = Path("../../data/results/Qubits12")/run_id
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / f"params.npy",  jax.device_get(trained_params))
np.save(out_dir / f"loss.npy", jax.device_get(loss_hist))

print(f"model params have saved to: {out_dir / 'params.npy'}")
print(f"Loss record has saved to: {out_dir / 'loss.npy'}")