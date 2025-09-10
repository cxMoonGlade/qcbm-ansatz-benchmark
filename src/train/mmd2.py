# mmdagg_prob.py  — Aggregated MMD^2 for discrete probability vectors
# Compatible with your mmdagg kernel naming & weights. Uses Hamming geometry.

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

# ---- import helpers from your code (reuse names/semantics) ----
# If this file sits next to your mmd_agg.py you can:
from src.train.mmd_agg import kernel_matrix, create_weights   # ← from your uploaded file

# ---------- 1) Bitstring geometry (Hamming / L2 on {0,1}^n) ----------
def _bit_matrix(n_bits: int, dtype=jnp.int32):
    d = 1 << n_bits
    idx = jnp.arange(d, dtype=jnp.uint32)[:, None]        # (d,1)
    offs = jnp.arange(n_bits, dtype=jnp.uint32)[None, :]  # (1,n)
    bits = (idx >> offs) & 1
    return bits.astype(dtype)                              # (d,n_bits)

def _pairwise_dist_from_bits(bits: jnp.ndarray, out_dtype=jnp.float64):
    diff = jnp.abs(bits[:, None, :] - bits[None, :, :])    # (d,d,n)
    D_l1 = diff.sum(axis=-1).astype(out_dtype)             # Hamming
    D_l2 = jnp.sqrt(D_l1)                                  # Euclidean on {0,1}
    return D_l1, D_l2

def _make_dist_for_pmfs(d: int, dtype=jnp.float64):
    n_bits = int(jnp.log2(d))
    if (1 << n_bits) != d:
        raise ValueError(f"d={d} is not a power of 2; expected bitstrings.")
    bits = _bit_matrix(n_bits)
    return _pairwise_dist_from_bits(bits, out_dtype=dtype)

# ---------- 2) Bandwidth schedule (median trick, per metric) ----------
def _median_nonzero(upper_tri):
    # upper_tri: vector of upper-tri entries (including diag)
    v = upper_tri[upper_tri > 0]
    return jnp.median(v)

def _compute_bandwidths_from_dist(pairwise, number_bandwidths: int):
    # dd = sorted nonzero distances
    tri = pairwise[jnp.triu_indices(pairwise.shape[0], k=0)]
    lam_min = _median_nonzero(tri)            # robust lower
    lam_min = jnp.maximum(lam_min, 1e-1) / 2  # match your mmdagg scaling
    lam_max = jnp.maximum(tri.max(), 3e-1) * 2
    power = (lam_max / lam_min) ** (1.0 / (number_bandwidths - 1))
    return jnp.array([lam_min * (power ** i) for i in range(number_bandwidths)])

# ---------- 3) Build an aggregated MMD^2 evaluator (precompute Ks) ----------
@partial(jax.jit, static_argnums=(5,))
def _core_eval(delta, K_stack, weights, N, d, use_sqrt):
    # delta: (d,), K_stack: (N,d,d), weights:(N,)
    # value_i = delta^T K_i delta
    # Use vmap for efficiency
    def quad(K):
        return delta @ (K @ delta)
    vals = jax.vmap(quad)(K_stack)          # (N,)
    mmd2 = jnp.sum(weights * vals)
    return jnp.sqrt(jnp.maximum(mmd2, 0.0) + 1e-20) if use_sqrt else mmd2

def build_mmdagg_prob(
    d: int,
    *,
    kernel: str = "laplace_gaussian",
    number_bandwidths: int = 10,
    weights_type: str = "uniform",
    bandwidths_l1: jnp.ndarray | None = None,
    bandwidths_l2: jnp.ndarray | None = None,
    dtype=jnp.float64,
    return_details: bool = False,
    use_sqrt: bool = False,   # if True return MMD (not MMD^2)
):
    """
    Prepare an aggregated MMD evaluator for probability vectors of length d.
    Returns a function f(p, q) -> (loss [, details]).
    """

    # 1) Geometry & pairwise distances
    D_l1, D_l2 = _make_dist_for_pmfs(d, dtype=dtype)

    # 2) Bandwidths (auto if not provided)
    #    For mixed "l1"/"l2" collections we keep separate sequences.
    if bandwidths_l1 is None:
        bandwidths_l1 = _compute_bandwidths_from_dist(D_l1, number_bandwidths)
    if bandwidths_l2 is None:
        bandwidths_l2 = _compute_bandwidths_from_dist(D_l2, number_bandwidths)

    # 3) Expand kernel list (order: l1 first, then l2) like your mmdagg
    if kernel in ("laplace", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1",
                  "matern_3.5_l1", "matern_4.5_l1"):
        kernel_bandwidths_l_list = [(kernel, bandwidths_l1, "l1")]
    elif kernel in ("gaussian", "imq", "matern_0.5_l2", "matern_1.5_l2",
                    "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2"):
        kernel_bandwidths_l_list = [(kernel, bandwidths_l2, "l2")]
    elif kernel in ("laplace_gaussian", "gaussian_laplace"):
        kernel_bandwidths_l_list = [("laplace", bandwidths_l1, "l1"),
                                    ("gaussian", bandwidths_l2, "l2")]
    elif kernel == "all_matern_l1":
        ks = [f"matern_{i}.5_l1" for i in range(5)]
        kernel_bandwidths_l_list = [(k, bandwidths_l1, "l1") for k in ks]
    elif kernel == "all_matern_l2":
        ks = [f"matern_{i}.5_l2" for i in range(5)]
        kernel_bandwidths_l_list = [(k, bandwidths_l2, "l2") for k in ks]
    elif kernel == "all_matern_l1_l2":
        ks = [f"matern_{i}.5_l{ell}" for ell in (1,2) for i in range(5)]
        bws_list = [bandwidths_l1]*5 + [bandwidths_l2]*5
        ls_list  = ["l1"]*5 + ["l2"]*5
        kernel_bandwidths_l_list = list(zip(ks, bws_list, ls_list))
    elif kernel == "all":
        ks = [f"matern_{i}.5_l{ell}" for ell in (1,2) for i in range(5)] + ["gaussian","imq"]
        bws_list = [bandwidths_l1]*5 + [bandwidths_l2]*7
        ls_list  = ["l1"]*5 + ["l2"]*7
        kernel_bandwidths_l_list = list(zip(ks, bws_list, ls_list))
    else:
        raise ValueError(f"Unknown kernel flag: {kernel}")

    # 4) Weights (same rule as your mmdagg)
    weights_per_bw = create_weights(number_bandwidths, weights_type) / len(kernel_bandwidths_l_list)

    # 5) Precompute all kernel matrices and weights → stack
    K_list = []
    W_list = []
    details = {}
    idx = 0
    for kname, bws, l in kernel_bandwidths_l_list:
        pairwise = D_l1 if l == "l1" else D_l2  # distance matrix
        for i in range(number_bandwidths):
            bw = bws[i]
            K = kernel_matrix(pairwise, l, kname, bw)     # (d,d)
            K_list.append(K.astype(dtype))
            W_list.append(weights_per_bw[i])
            if return_details:
                details[f"Single test {idx+1}"] = {
                    "Kernel": kname, "Metric": l,
                    "Bandwidth": float(bw), "Weight": float(weights_per_bw[i])
                }
            idx += 1

    K_stack = jnp.stack(K_list, axis=0)                   # (N,d,d)
    weights = jnp.asarray(W_list, dtype=dtype)            # (N,)
    N = K_stack.shape[0]

    # 6) Final evaluator
    @partial(jax.jit, static_argnums=(2,))
    def mmd_eval(p: jnp.ndarray, q: jnp.ndarray, return_details_flag: bool=False):
        p = jnp.asarray(p, dtype=dtype)
        q = jnp.asarray(q, dtype=dtype)
        assert p.shape == (d,) and q.shape == (d,), f"expected (d,), got {p.shape},{q.shape}"
        # ensure probabilities sum to ~1 (optional normalize)
        s1 = jnp.sum(p); s2 = jnp.sum(q)
        p = jnp.where(jnp.abs(s1-1.0) > 1e-6, p / (s1 + 1e-12), p)
        q = jnp.where(jnp.abs(s2-1.0) > 1e-6, q / (s2 + 1e-12), q)

        delta = p - q
        val = _core_eval(delta, K_stack, weights, N, d, use_sqrt)
        if return_details_flag:
            # attach per-test raw quadratic forms too (cheap extra pass)
            def quad(K):
                return delta @ (K @ delta)
            vals = jax.vmap(quad)(K_stack)
            # materialize to host scalars for logging
            det = {k: {**v, "MMD2": float(vals[i])} for i,(k,v) in enumerate(details.items())} if return_details else {}
            return val, det
        return val

    return mmd_eval
