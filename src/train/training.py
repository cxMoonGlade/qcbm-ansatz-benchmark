# %%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pennylane as qml
import jax
import jax.numpy as jnp

# import functools

import sys
sys.path.append('../..')
from src.train.mmdagg_probs import mmdagg_prob

import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd
from itertools import product
import sys
sys.path.append('../..')

import optax
import catalyst

from src.train.qcbm import QCBM
from src.circuits.ansatz1 import hardware_efficient_ansatz
from src.circuits.ansatz2 import ising_structured_ansatz
from src.circuits.ansatz3 import eh2d_ansatz
from src.circuits.ansatz4 import mi_ansatz

# %%
def mutual_information_matrix(bits: jnp.ndarray) -> jnp.ndarray:
    """
    bits : (N, n)  0/1 ndarray (CPU numpy 亦可)
    return: (n, n)  float64 JAX array,  对角线已置 0
    """
    N, n = bits.shape
    bits = bits.astype(jnp.int32)                  # 确保 0/1 -> 0/1 int

    # --- 1. 单比特边缘概率 P(q_k = 1) ---------------------------------
    pk1 = bits.mean(axis=0)                       # (n,)  float64
    pk0 = 1.0 - pk1                               # (n,)

    # --- 2. 二比特联合概率 P(q_i = a, q_j = b) ------------------------
    #    只需算 P11，其他通过边缘补全更快：
    #    P11(i,j) = mean( bits[:,i] & bits[:,j] )
    bT        = bits.T                            # (n, N)
    P11       = (bT[:, None, :] & bT[None, :, :]).mean(axis=-1)  # (n, n)
    # 边缘补全：
    P10 = pk1[:, None] - P11                      # (n, n)
    P01 = pk1[None, :] - P11
    P00 = 1.0 - (P11 + P10 + P01)

    # --- 3. 互信息 I(i,j) = Σ_{a,b∈{0,1}} P_ab log( P_ab / (P_a·P_b) ) --
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

    # --- 4. 对角线清零 & 返回 JAX 数组 -------------------------------
    Iij = Iij.at[jnp.diag_indices(n)].set(0.0)
    return Iij


def count_params1(n_bits: int, L: int) -> int:
    """
    return params requested for ansatz1
    """
    assert L % 2 == 0, "for ansatz1, L must be even number"
    return int((3 * L / 2 + 1) * n_bits - (L / 2))

def count_params2(R: int, C: int, L: int, periodic: bool= False) -> int:
    n = R*C
    per_layer = 2 * n + (R * (C - 1) + C * (R - 1))
    if periodic:
        per_layer += R + C
    return per_layer * L

def count_params3(shape=(2,4), L=4, add_dt=False):
    R, C = shape; n = R * C
    per_layer = 2*n + R*(C-1) + (R-1)*C + (1 if add_dt else 0)
    return per_layer * L

def count_params4(n=8, L=3, keep_edges=24, extras=4):
    return 2*L*n + L*keep_edges + extras  


def ansatz_set(ansatz):
    n_bits = 8
    n_qubits = 8
    bit_cols = [f"q{i}" for i in range(n_bits)]
    df = pd.read_csv("../../data_2d/train.csv")
    if ansatz == hardware_efficient_ansatz:
        L = 8
        pc = count_params1(n_bits, L)
        id = 1

    if ansatz == ising_structured_ansatz:
        L = 4
        pc = count_params2(2, 4, 4, False)
        id = 2

    if ansatz == eh2d_ansatz:
        L = 4
        pc = count_params3()
        id = 3

    if ansatz == mi_ansatz:
        L = 3
        keep_edges = 16
        pc = count_params4()
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

# %%
# init target probs & model & params 

def main():
    

    df = pd.read_csv("../../data_2d/train.csv")
    n_bits = 8
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

    # %%
    
    ## Control # of Params around 100


    ansatz = mi_ansatz
    n_qubits= 8
    mmd_fn = mmdagg_prob


    ansatz, L, pc, id = ansatz_set(ansatz)

        
    model = QCBM(ansatz, n_bits, L, mmd_fn, target_probs)
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, shape=(pc,))




    # %%
    # ------------ quick sanity check ------------
    key  = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (pc,), dtype=jnp.float64)

    # loss_val = model.loss(params)
    # print("loss =", loss_val, "dtype:", loss_val.dtype, "device:", loss_val.device)

    # grads = jax.grad(model.loss)(params)
    # print("grads shape", grads.shape, "dtype:", grads.dtype, "device:", grads.device)


    # %%
    # opt
    # opt = optax.adam(1e-2)

    lr_sched = optax.exponential_decay(  # 从 1e‑2 → 每 200 步 × 0.9
        init_value=1e-2, transition_steps=200, decay_rate=0.9, staircase=True
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),   # 可选
        optax.adam(lr_sched, b2=0.999)
    )
    
    def update_step(i, params, opt_state, loss_log):
        loss_val, grads = catalyst.value_and_grad(model.loss)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_log = loss_log.at[i].set(loss_val)
        catalyst.debug.print("Step {i}: loss = {loss}", i=i, loss=loss_val)
        return (params, opt_state, loss_log)

    @qml.qjit     # 将整个 for‑loop 优化流程一次性编译
    def optimization(params, n_steps: int = 1000):
        opt_state = opt.init(params)
        loss_log  = jnp.zeros(n_steps, dtype=params.dtype)
        params, opt_state, loss_log = qml.for_loop(
            0, n_steps, 1
        )(update_step)(params, opt_state, loss_log)
        return params, loss_log



    # %%
    trained_params, loss_hist = optimization(params, n_steps=1000)


    # %%
    jax.block_until_ready(trained_params)      # 保证 GPU 计算结束

    # ───── 1. 保存 ─────
    import numpy as np
    from pathlib import Path
    run_id = f"result{id}"


    out_dir = Path("../../data/results")/run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"params.npy",  jax.device_get(trained_params))
    np.save(out_dir / f"loss.npy", jax.device_get(loss_hist))

    print(f"模型参数已保存至: {out_dir / 'params.npy'}")
    print(f"Loss 记录已保存至: {out_dir / 'loss.npy'}")

    # %%
    # ───── 2. 载入并恢复模型 ─────
    import scipy
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    import numpy as np
    out_dir = Path("../../data/results/result4")
    loaded_params = jnp.asarray(np.load(out_dir/"params.npy"), dtype=jnp.float64)

    model = QCBM(ansatz, n_bits, L, mmdagg_prob, target_probs)  # 跟训练时相同
    loss_loaded = model.loss(loaded_params).block_until_ready()
    print("Loss after reload:", float(loss_loaded))

    probs_trained = model.circuit(loaded_params).block_until_ready()
    kl   = scipy.stats.entropy(target_probs, probs_trained)
    l1   = jnp.mean(jnp.abs(target_probs - probs_trained))
    gpu = jax.devices("gpu")[0]
    mmd, _ = mmdagg_prob(jax.device_put(target_probs, gpu), 
                        jax.device_put(probs_trained, gpu),
                        kernel="laplace_gaussian", number_bandwidths=10)
    mask = target_probs > 0          # 只看非零目标概率
    kl_nonzero = scipy.stats.entropy(
        target_probs[mask],
        probs_trained[mask] / probs_trained[mask].sum()
    )



    print(f"KL   : {kl:.4e}")
    print("KL (target>0 only):", kl_nonzero)
    print(f"L1   : {l1:.4e}")
    print(f"MMD  : {mmd:.4e}")

    # %%
    import matplotlib.pyplot as plt

    params_loaded   = loaded_params
    loss_hist_loaded = np.load(f"{out_dir}/loss.npy")

    print("First 5 losses:", loss_hist_loaded[:5])
    print("Final   loss :", loss_hist_loaded[-1])

    plt.semilogy(loss_hist_loaded)
    plt.xlabel("Step")
    plt.ylabel("Loss (log‑scale)")
    plt.title("Training curve")
    plt.show()


    # %%
    DATA_DIR   = "../../data"
    def csv_to_probs(path: str, n_bits=8, dtype=jnp.float64):
        """把 csv → (2**n_bits,) 经验概率向量"""
        df = pd.read_csv(path)
        bit_cols = [f"q{i}" for i in range(n_bits)]
        bitstrings = (
            df[bit_cols]
            .astype(str)
            .agg("".join, axis=1)
        )
        counts = bitstrings.value_counts().sort_index()
        all_bits = ["".join(seq) for seq in product("01", repeat=n_bits)]
        probs = pd.Series(0.0, index=all_bits, dtype=float)
        probs.update(counts / counts.sum())
        return jnp.asarray(probs.values, dtype=dtype)

    def kl_div(p, q, eps=1e-10):
        p = jnp.clip(p, eps, 1.0)
        q = jnp.clip(q, eps, 1.0)
        p = p / p.sum();  q = q / q.sum()
        return jnp.sum(p * jnp.log(p / q))

    # ---------- 1. 读入训练好的参数 ----------
    params_loaded = params_loaded

    # ---------- 2. 建立 QCBM 电路 ----------
    model = QCBM(
        ansatz       = ansatz,
        n_qubits     = 8,
        L            = 4,
        mmd_fn       = mmdagg_prob,
        target_probs = jnp.zeros(256),   # 占位即可
        dtype        = jnp.float64,
    )

    # ---------- 3. JIT‑ed 三指标函数 ----------
    @jax.jit
    def three_metrics(target_probs, params):
        probs = model.circuit(params)
        kl    = kl_div(target_probs, probs)
        l1    = jnp.mean(jnp.abs(target_probs - probs))
        mmd   = mmdagg_prob(
            target_probs, probs,
            kernel="laplace_gaussian", number_bandwidths=10,
            build_details=False, dtype=jnp.float64
        )
        return kl, l1, mmd

    # ---------- 4. 计算 & 打印 ----------
    splits = {
        "TEST"  : csv_to_probs("../../data_2d/test.csv"),
        "UNSEEN": csv_to_probs("../../data_2d/unseen.csv"),
    }

    for name, tgt in splits.items():
        kl, l1, mmd = three_metrics(tgt, params_loaded)
        # block_until_ready() 确保数字已从 GPU 同步回来
        kl, l1, mmd = map(lambda x: float(x.block_until_ready()), (kl, l1, mmd))
        print(f"{name:6s}:  KL = {kl:.4e}   L1 = {l1:.4e}   MMD = {mmd:.4e}")


if __name__ == "__main__":
    main()