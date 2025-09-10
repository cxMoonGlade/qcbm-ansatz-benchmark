# src/evaluation/FP2.py
# ------------------------------------------------------
# Expressibility via pairwise state fidelities (Section II-A).
# Compares empirical per-bin probability to Haar per-bin mass:
#   Haar PDF: p(F) = (2^n - 1) (1 - F)^(2^n - 2)
#   Haar CDF: 1 - (1 - F)^(2^n - 1)
# Saves: per-ansatz plot PNGs + CSV/JSON summary.
# Float64 throughout.  Pure NumPy + default.qubit (no JAX tracing).
# ------------------------------------------------------

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
np.set_printoptions(suppress=True)

import pennylane as qml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- repo import (no JAX needed here) ----------------------------------------
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Centralized specs (ansatz funcs + param counts)
from src.circuits.specs import get_ansatz_spec  # returns (ansatz_fn, L, P, meta)
# get_ansatz_spec signature expects n_bits and can build MI edges for ansatz4
# when provided train_df + bit_cols.  See specs.py.  (We pass its ansatz_fn
# directly into the QNode so edges/extras are honored.)  :contentReference[oaicite:4]{index=4}

# ---------- Haar law (Section II-A) ----------
def haar_pdf(F: np.ndarray, n_qubits: int) -> np.ndarray:
    """p(F) = (M-1) (1 - F)^(M-2), where M = 2^n."""
    M = 2 ** n_qubits
    return (M - 1) * (1.0 - F) ** (M - 2)

def haar_cdf(F: np.ndarray, n_qubits: int) -> np.ndarray:
    """CDF(F) = 1 - (1 - F)^(M-1), where M = 2^n."""
    M = 2 ** n_qubits
    return 1.0 - (1.0 - F) ** (M - 1)

# ---------- Utilities ----------
def maybe_load_mi_df(path_csv: str | None, n_bits: int, bit_prefix: str):
    """Return (train_df, bit_cols) if CSV provided; else (None, None)."""
    if not path_csv:
        return None, None
    import pandas as pd
    df = pd.read_csv(path_csv)
    cols = [f"{bit_prefix}{i}" for i in range(n_bits)]
    have = [c for c in cols if c in df.columns]
    if len(have) != n_bits:
        # fallback: any columns starting with prefix; keep exactly n_bits if possible
        cand = [c for c in df.columns if c.startswith(bit_prefix)]
        cand_sorted = sorted(
            cand,
            key=lambda c: int(c[len(bit_prefix):]) if c[len(bit_prefix):].isdigit() else 10**9
        )
        have = cand_sorted[:n_bits]
    if len(have) != n_bits:
        print(f"[warn] --mi-csv loaded but found {len(have)} bit columns (need {n_bits}); will try anyway.")
    return df, have

def build_state_qnode(ansatz_fn, n_qubits: int, L: int, P: int):
    """
    Returns (qnode, param_sampler, label)
    qnode(params: (P,)) -> statevector (complex128, shape (2^n,))
    """
    # Plain NumPy statevector device (no JIT, no tracers)
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    wires = list(range(n_qubits))

    @qml.qnode(dev, interface=None, diff_method=None)
    def state_qnode(params):
        ansatz_fn(params, wires=wires, L=L)
        return qml.state()

    def param_sampler(rng: np.random.Generator, K: int) -> np.ndarray:
        # Uniform in [-pi, pi], float64 NumPy
        return rng.uniform(-np.pi, np.pi, size=(K, P)).astype(np.float64, copy=False)

    return state_qnode, param_sampler

def sample_states(qnode, param_sampler, *, K: int, P: int,
                  rng: np.random.Generator, chunk: int = 256) -> np.ndarray:
    """Evaluate K random circuits -> states; chunked for memory. Returns (K, 2^n) complex128."""
    out: List[np.ndarray] = []
    done = 0
    while done < K:
        m = min(chunk, K - done)
        params = param_sampler(rng, m)             # (m, P) float64
        # Evaluate one-by-one to keep memory low and avoid PL transforms surprises
        states = [np.asarray(qnode(p)) for p in params]  # each (2^n,), complex128
        out.append(np.stack(states, axis=0))
        done += m
    return np.concatenate(out, axis=0)

def pairwise_fidelities(states: np.ndarray,
                        max_pairs: Optional[int],
                        rng: np.random.Generator) -> np.ndarray:
    """
    Compute |<psi_i|psi_j>|^2 for random pairs among states.
    - max_pairs=None uses all pairs (may be large).
    - Otherwise samples without replacement up to max_pairs pairs.
    """
    K, D = states.shape
    # normalize (safety)
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    states = states / np.clip(norms, 1e-16, None)

    total_pairs = K * (K - 1) // 2
    if (max_pairs is None) or (max_pairs >= total_pairs):
        ii, jj = np.triu_indices(K, k=1)
    else:
        # sample disjoint pair indices
        ii_all, jj_all = np.triu_indices(K, k=1)
        sel = rng.choice(ii_all.size, size=max_pairs, replace=False)
        ii, jj = ii_all[sel], jj_all[sel]

    overlaps = np.sum(states[ii].conj() * states[jj], axis=1)
    F = np.abs(overlaps) ** 2
    return F.astype(np.float64, copy=False)

# ---------- Plotting ----------
def plot_expr_panel(F: np.ndarray,
                    n_qubits: int,
                    bins: int,
                    title: str,
                    out_png: Path,
                    x_zoom: float = 0.03) -> Dict[str, float]:
    """
    Plot empirical per-bin probability vs Haar expected per-bin mass.
    Also renders residuals and a zoomed inset near F≈0.
    Returns dict with KL, FP2, etc.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    F = np.asarray(F, dtype=np.float64)
    hist, edges = np.histogram(F, bins=bins, range=(0.0, 1.0), density=False)
    total = int(hist.sum())
    hist_prob = hist / max(total, 1)  # per-bin probability
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    # Haar expected mass per bin (must sum to 1 over bins)
    exp_mass = np.diff(haar_cdf(edges, n_qubits))

    # ---- Metrics ----
    eps = 1e-12
    kl = float(np.sum(np.where(hist_prob > 0,
                               hist_prob * (np.log(hist_prob + eps) - np.log(exp_mass + eps)),
                               0.0)))
    fp2_emp  = float(np.sum(hist_prob ** 2))
    fp2_haar = float(np.sum(exp_mass ** 2))
    fp2_off  = float(abs(fp2_emp - fp2_haar))

    # ---- Figure ----
    fig = plt.figure(figsize=(10.4, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.1, 1.25], hspace=0.12)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)

    # Main: empirical histogram (per-bin prob) vs Haar expected mass (line)
    ax_main.bar(centers, hist_prob, width=width, align="center", alpha=0.85,
                label=r"Empirical $\hat{P}_{\mathrm{PQC}}(F)$")
    ax_main.plot(centers, exp_mass, linewidth=2.0, label="Haar PDF (shape)")

    ax_main.set_ylabel("Probability (per bin)")
    ax_main.set_title(title)
    ax_main.legend(loc="upper right")

    # Residuals: (empirical - Haar) per bin
    resid = hist_prob - exp_mass
    ax_res.axhline(0.0, linewidth=1.0)
    ax_res.bar(centers, resid, width=width, align="center")
    ax_res.set_xlabel(r"Fidelity $F = |\langle \psi_\theta | \psi_\phi \rangle|^2$")
    ax_res.set_ylabel("Residual")

    # Zoomed inset near F≈0
    iax = inset_axes(ax_main, width="40%", height="40%", loc="upper right")
    mask = centers <= x_zoom
    iax.bar(centers[mask], hist_prob[mask], width=width, align="center", alpha=0.85)
    iax.plot(centers[mask], exp_mass[mask], linewidth=1.5)
    iax.set_xlim(0.0, x_zoom)
    # Hide clutter
    iax.set_xticks([]); iax.set_yticks([])

    # Metrics banner
    fig.suptitle(f"KL={kl:.3e} | FP2={fp2_emp:.3e} (off={fp2_off:.3e}, Haar={fp2_haar:.3e})",
                 y=0.99, fontsize=11)

    # Save AFTER plotting
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return dict(KL=kl, FP2=fp2_emp, FP2_haar=fp2_haar, FP2_offset=fp2_off)


# ---------- Main runner ----------
def run_expressibility(ansatz_ids, n_qubits, K, pairs, bins, seed, mi_csv, bit_prefix, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # pre-load MI dataframe if needed by ansatz 4
    train_df, bit_cols = maybe_load_mi_df(mi_csv, n_qubits, bit_prefix)

    rows = []
    summary_json = {}

    for aid in ansatz_ids:
        # Build spec (ansatz function already wired with MI edges for 4)
        if aid == 4 and train_df is None:
            print("[warn] ansatz4 requested but no --mi-csv provided; skipping ansatz 4.")
            continue

        # pass n_bits and, for ansatz4, the MI inputs; uses defaults for keep_edges/extras
        if aid == 4:
            ans_fn, L, P, meta = get_ansatz_spec(
                aid, n_qubits, train_df=train_df, bit_cols=bit_cols
            )
        else:
            ans_fn, L, P, meta = get_ansatz_spec(aid, n_qubits)

        print(f"[info] ansatz{aid}: L={L}, P={P}, meta={meta}")

        qnode, param_sampler = build_state_qnode(ans_fn, n_qubits, L, P)

        # Sample K states
        states = sample_states(qnode, param_sampler, K=K, P=P, rng=rng, chunk=128)

        # Decide number of pairs
        if isinstance(pairs, str) and pairs.lower() == "auto":
            max_pairs = min(200_000, K * (K - 1) // 2)
        elif (pairs is None) or (pairs == 0):
            max_pairs = None
        else:
            max_pairs = int(pairs)

        F = pairwise_fidelities(states, max_pairs=max_pairs, rng=rng)

        title = f"{meta.get('name', f'ansatz{aid}')} — Expressibility (Haar KL, FP2)"
        png_path = outdir / f"expr_ansatz{aid}.png"
        stats = plot_expr_panel(F, n_qubits, bins, title, png_path, x_zoom=0.05)

        row = dict(ansatz=aid, n_qubits=n_qubits, K=K, bins=bins,
                   pairs=(max_pairs if max_pairs is not None else "all"), **stats)
        rows.append(row)
        summary_json[f"ansatz{aid}"] = row

    # Save tables
    if rows:
        import csv
        csv_path = outdir / "expressibility_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        with open(outdir / "expressibility_summary.json", "w") as f:
            json.dump(summary_json, f, indent=2)
        print(f"[done] wrote plots + summary to {outdir}")
    else:
        print("[warn] No results produced (nothing to evaluate).")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Expressibility via Sec. II.A (PQC random states)")
    p.add_argument("--ansatz", type=int, nargs="+", required=True, help="Ansatz IDs, e.g. 1 2 3 4")
    p.add_argument("--n-bits", type=int, default=8, help="Number of qubits (default: 8)")
    p.add_argument("--K", type=int, default=1200, help="Number of random parameter samples")
    p.add_argument("--pairs", type=str, default="auto",
                   help="'auto' (≈min(2e5, K*(K-1)/2)) or an integer for number of pairs")
    p.add_argument("--bins", type=int, default=75, help="Histogram bins for fidelity/angle")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed")
    # Ans4 MI-edge builder inputs (optional but REQUIRED for ansatz 4)
    p.add_argument("--mi-csv", type=str, default=None,
                   help="CSV file with training data to derive MI edges for ansatz 4")
    p.add_argument("--bit-prefix", type=str, default="q",
                   help="Column prefix for bit columns in --mi-csv (default: 'q')")
    p.add_argument("--out", type=str, default="data/expressibility",
                   help="Output directory for figures & summary")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_expressibility(
        ansatz_ids=list(args.ansatz),
        n_qubits=args.n_bits,
        K=args.K,
        pairs=args.pairs,         # 'auto' or int
        bins=args.bins,
        seed=args.seed,
        mi_csv=args.mi_csv,
        bit_prefix=args.bit_prefix,
        outdir=args.out,
    )
