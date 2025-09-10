# src/circuits/specs.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Callable, Dict, Any

from src.circuits.ansatz1 import hardware_efficient_ansatz
from src.circuits.ansatz2 import ising_structured_ansatz
from src.circuits.ansatz3 import eh2d_ansatz
from src.circuits.ansatz4 import mi_ansatz

# --- canonical param-count formulas (single layer notation mirrors your training) ---
def count_params1(n_bits: int, L: int) -> int:
    # hardware_efficient_ansatz
    return int((3 * L + 1) * n_bits - L)

def count_params2(R: int, C: int, L: int, periodic: bool = False) -> int:
    # ising_structured_ansatz
    n = R * C
    per_layer = 2 * n + (R * (C - 1) + C * (R - 1))
    if periodic:
        per_layer += R + C
    return per_layer * L

def count_params3(R: int, C: int, L: int, add_dt: bool = False) -> int:
    # eh2d_ansatz
    n = R * C
    per_layer = 2 * n + R * (C - 1) + (R - 1) * C + (1 if add_dt else 0)
    return per_layer * L

def count_params4(n_bits: int, L: int, keep_edges: int, extras: int = 4) -> int:
    # mi_ansatz (with extras)
    return 2 * L * n_bits + L * keep_edges + extras

# --- MI helper: build edges by mutual information from a train DataFrame ---
def mutual_information_matrix(bits_01: np.ndarray) -> np.ndarray:
    eps = 1e-12
    N, d = bits_01.shape
    X = bits_01.astype(np.int32)
    MI = np.zeros((d, d), dtype=np.float64)
    p1 = X.mean(axis=0); p0 = 1.0 - p1
    for i in range(d):
        for j in range(i+1, d):
            both1 = np.mean((X[:, i] == 1) & (X[:, j] == 1))
            i1j0  = p1[i] - both1
            i0j1  = p1[j] - both1
            i0j0  = 1.0 - (both1 + i1j0 + i0j1)
            pij = np.array([i0j0, i0j1, i1j0, both1], dtype=np.float64)
            pi  = np.array([p0[i], p1[i]], dtype=np.float64)
            pj  = np.array([p0[j], p1[j]], dtype=np.float64)
            pij = np.clip(pij, eps, 1); pi = np.clip(pi, eps, 1); pj = np.clip(pj, eps, 1)
            denom = np.array([pi[0]*pj[0], pi[0]*pj[1], pi[1]*pj[0], pi[1]*pj[1]])
            MI[i, j] = MI[j, i] = float(np.sum(pij * np.log(pij / denom)))
    return MI

def build_mi_ansatz(n_bits: int, L: int, keep_edges: int, extras: int,
                    train_df, bit_cols):
    bit_np = train_df[bit_cols].astype(int).to_numpy()
    mi_mat = mutual_information_matrix(bit_np)
    triu_i, triu_j = np.triu_indices(n_bits, k=1)
    mi_flat = mi_mat[triu_i, triu_j]
    top_idx = np.argsort(-mi_flat)[:keep_edges]
    mi_edges = [(int(triu_i[k]), int(triu_j[k])) for k in top_idx]

    def ansatz_mi(params, wires, *, L=None, **kw):
        return mi_ansatz(params, wires, mi_edges=mi_edges, L=L, extras=extras, **kw)

    return ansatz_mi

# --- unified spec: returns (ansatz_fn, L, param_count, meta) ---
def get_ansatz_spec(ansatz_id: int,
                    n_bits: int,
                    *,
                    R: int = 2, C: int = 4,
                    L1: int = 4, L_M: int = 3,
                    keep_edges: int = 16, extras: int = 4,
                    train_df=None, bit_cols=None) -> Tuple[Callable, int, int, Dict[str, Any]]:
    if ansatz_id == 1:
        L = L1
        fn = hardware_efficient_ansatz
        pc = count_params1(n_bits, L)
        return fn, L, pc, {"name": "ansatz1"}

    if ansatz_id == 2:
        L = L1
        fn = ising_structured_ansatz
        pc = count_params2(R, C, L, periodic=False)
        return fn, L, pc, {"name": "ansatz2", "R": R, "C": C}

    if ansatz_id == 3:
        L = L1
        fn = eh2d_ansatz
        pc = count_params3(R, C, L, add_dt=False)
        return fn, L, pc, {"name": "ansatz3", "R": R, "C": C}

    if ansatz_id == 4:
        L = L_M
        if train_df is None or bit_cols is None:
            raise ValueError("ansatz4 requires train_df and bit_cols to build MI edges")
        fn = build_mi_ansatz(n_bits, L, keep_edges, extras, train_df, bit_cols)
        pc = count_params4(n_bits, L, keep_edges, extras)
        return fn, L, pc, {"name": "ansatz4", "keep_edges": keep_edges, "extras": extras}

    raise ValueError(f"Unknown ansatz_id={ansatz_id}")
