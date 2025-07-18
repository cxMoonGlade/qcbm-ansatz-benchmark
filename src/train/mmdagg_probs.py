# mmdagg_prob.py
# ======================================================
#  MMDAgg on probability vectors (discrete distributions)
#  - 支持多核 + 多带宽 + 权重聚合
#  - 纯 JAX，可 jit 到 GPU / TPU
# ======================================================

import jax
jax.config.update("jax_enable_x64", True)
# ==============================================
#  mmdagg_probs.py  ―  Prob‑vector MMDAgg (JAX)
#  kernels: laplace / gaussian / imq / matern ν∈{0.5,1.5,2.5}
# ==============================================

import jax.numpy as jnp
from functools import partial

# ---------- 0) 常量映射 ----------
_KID = dict(laplace=0, gaussian=1, imq=2,
            matern05=3, matern15=4, matern25=5)
_IDK = {v: k for k, v in _KID.items()}
_MID = {"l1": 0, "l2": 1}                        # metric id


# ---------- 1) 距离矩阵（一次性构好） ----------
def _make_dist(d: int, dtype=jnp.float32):
    eye  = jnp.eye(d, dtype=dtype)
    off  = 1.0 - eye
    D_l1 = off * 2.0
    D_l2 = off * jnp.sqrt(2.0)
    return D_l1, D_l2


# ---------- 2) 单核矩阵 ----------
def _kernel_matrix(dist, k_id: int, bw):
    r = dist / bw

    def matern(nu):
        if nu == 0.5:
            return jnp.exp(-r)
        if nu == 1.5:
            return (1 + jnp.sqrt(3)*r) * jnp.exp(-jnp.sqrt(3)*r)
        if nu == 2.5:
            return (1 + jnp.sqrt(5)*r + 5*r**2/3) * jnp.exp(-jnp.sqrt(5)*r)

    return jax.lax.switch(
        k_id,
        [
            lambda: jnp.exp(-r),            # laplace  (L1)
            lambda: jnp.exp(-r**2),         # gaussian (L2)
            lambda: (1 + r**2)**-0.5,       # imq      (L2)
            lambda: matern(0.5),
            lambda: matern(1.5),
            lambda: matern(2.5),
        ],
    )


# ---------- 3) JIT 数值核心 ----------
def _build_core(D_l1, D_l2):
    """闭包捕获距离矩阵 → 避免当作 JIT 参数传入"""
    @jax.jit
    def _core(delta, k_ids, m_ids, bws):
        """返回 (T,) MMD 值"""
        def body(i, acc):
            dist = jax.lax.select(m_ids[i] == 0, D_l1, D_l2)
            K    = _kernel_matrix(dist, k_ids[i], bws[i])
            acc  = acc.at[i].set(delta @ K @ delta)
            return acc

        m_init = jnp.zeros_like(bws)
        return jax.lax.fori_loop(0, bws.size, body, m_init)

    return _core


# ---------- 4) 权重 ----------
def _create_weights(N: int, mode: str = "uniform") -> jnp.ndarray:
    w = jnp.ones(N)
    if mode == "decreasing":
        w = 1.0 / jnp.arange(1, N + 1)
    elif mode == "increasing":
        w = 1.0 / jnp.arange(N, 0, -1)
    elif mode == "centred":
        idx = jnp.arange(N)
        mid = (N - 1) / 2
        w = 1.0 / (jnp.abs(idx - mid) + 1)
    return w / w.sum()


# ---------- 5) kernel / metric 序列 ----------
def _ids_for_kernel(flag: str, n_bw: int):
    def seq(kname, metric):
        k = _KID[kname]; m = _MID[metric]
        return [k] * n_bw, [m] * n_bw

    if flag == "laplace_gaussian":
        k1, m1 = seq("laplace",  "l1")
        k2, m2 = seq("gaussian", "l2")
        return k1 + k2, m1 + m2

    if flag == "gaussian_laplace":
        k1, m1 = seq("gaussian", "l2")
        k2, m2 = seq("laplace",  "l1")
        return k1 + k2, m1 + m2

    if flag == "all":
        names = [
            ("laplace",   "l1"),
            ("gaussian",  "l2"),
            ("imq",       "l2"),
            ("matern05",  "l1"), ("matern15", "l1"), ("matern25", "l1"),
            ("matern05",  "l2"), ("matern15", "l2"), ("matern25", "l2"),
        ]
        k_ids, m_ids = [], []
        for kname, metric in names:
            k_vec, m_vec = seq(kname, metric)
            k_ids += k_vec
            m_ids += m_vec
        return k_ids, m_ids

    raise ValueError(f"Unknown kernel flag '{flag}'")


# ---------- 6) 公开 API ----------
def mmdagg_prob(
    p, q,
    *,
    kernel="laplace_gaussian",
    number_bandwidths=10,
    base_bw=1.0,
    ratio=2.0,
    weights_type="uniform",
    build_details: bool = True,
    dtype=jnp.float32,
):
    """
    Parameters
    ----------
    p, q   : (d,) 概率向量 (sum=1)
    kernel : 'laplace_gaussian' | 'gaussian_laplace' | 'all'
    number_bandwidths : 每核带宽数
    base_bw, ratio    : 带宽 λ₀·ratioᵏ
    weights_type      : uniform / decreasing / increasing / centred
    build_details     : False → 仅返回标量 loss（训练时用）

    Returns
    -------
    loss            : jnp.float32 scalar
    details (opt.)  : dict（人类可读）
    """
    p = jnp.asarray(p, dtype=dtype)
    q = jnp.asarray(q, dtype=dtype)
    assert p.shape == q.shape and p.ndim == 1, "p,q shape mismatch"
    d      = p.size
    delta  = p - q

    # ---- 预计算距离矩阵 ----
    D_l1, D_l2 = _make_dist(d, dtype)
    _core      = _build_core(D_l1, D_l2)

    # ---- 带宽向量 ----
    bw_seq = base_bw * ratio ** jnp.arange(number_bandwidths, dtype=dtype)

    # ---- kernel / metric id ----
    k_ids_l, m_ids_l = _ids_for_kernel(kernel, number_bandwidths)
    k_ids = jnp.array(k_ids_l, dtype=jnp.int32)
    m_ids = jnp.array(m_ids_l, dtype=jnp.int32)
    bws   = jnp.tile(bw_seq, k_ids.size // number_bandwidths)

    # ---- 数值核心 ----
    mmd_vals = _core(delta, k_ids, m_ids, bws)          # (T,)

    # ---- 权重聚合 ----
    B = number_bandwidths           # 每核带宽数（静态）
    K = len(k_ids_l) // B           # 核数
    w_bw   = _create_weights(B, weights_type)           # (B,)
    weights = jnp.tile(w_bw, K) / K                     # (T,)
    loss    = jnp.sum(weights * mmd_vals)

    if not build_details:
        return loss                                     # <<< 训练路径

    # ---- details dict (Python，非 JIT) ----
    details = {}
    for i in range(bws.size):
        details[f"Single test {i+1}"] = {
            "Kernel":   _IDK[int(k_ids_l[i])],
            "Metric":   "l1" if m_ids_l[i] == _MID["l1"] else "l2",
            "Bandwidth": float(bws[i]),
            "Weight":    float(weights[i]),
            "MMD":       float(mmd_vals[i]),
        }

    return loss, details
