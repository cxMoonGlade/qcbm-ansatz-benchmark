## labraries for training

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["JAX_TRACEBACK_FILTERING"]="off"



import pennylane as qml
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


# === explicit CLI mapping: put this near the top of gpu_train.py =============
import argparse, sys
from pathlib import Path

# Ensure repo root is importable when running: python ./src/train/gpu_train.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the four ansatz modules explicitly
from src.circuits.ansatz1 import hardware_efficient_ansatz
from src.circuits.ansatz2 import ising_structured_ansatz
from src.circuits.ansatz3 import eh2d_ansatz
from src.circuits.ansatz4 import mi_ansatz

ANSATZ_FNS = {
    1: hardware_efficient_ansatz,
    2: ising_structured_ansatz,
    3: eh2d_ansatz,
    4: mi_ansatz,
}

def parse_ansatz_id(default_id: int = 2) -> int:
    """Support: -1/-2/-3/-4 or -a/--ansatz 1..4; env override ANSATZ=1..4."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_mutually_exclusive_group()
    g.add_argument("-a", "--ansatz", type=int, choices=[1, 2, 3, 4])
    g.add_argument("-1", dest="one",   action="store_true")
    g.add_argument("-2", dest="two",   action="store_true")
    g.add_argument("-3", dest="three", action="store_true")
    g.add_argument("-4", dest="four",  action="store_true")
    args, _ = p.parse_known_args()

    if args.ansatz: return args.ansatz
    if args.one:    return 1
    if args.two:    return 2
    if args.three:  return 3
    if args.four:   return 4
    env = os.getenv("ANSATZ")
    if env in {"1","2","3","4"}: return int(env)
    return default_id

ANSATZ_ID = parse_ansatz_id(default_id=1)
ANSA_FN   = ANSATZ_FNS[ANSATZ_ID]
print(f"[info] Using ansatz{ANSATZ_ID}: {ANSA_FN.__module__}.{ANSA_FN.__name__}")


# 1. current file（train.py）abspath
current_file = os.path.abspath(__file__)

# 2. back to root path（report1）
report1_dir = os.path.abspath(os.path.join(current_file, "../../../"))
# 3. 把项目根路径加入 sys.path
if report1_dir not in sys.path:
    sys.path.insert(0, report1_dir)


from qcbm import QCBM



# ------------  init target probs & model & params ------------
from itertools import product

import pandas as pd
data_path = os.path.join(report1_dir, "data_2d/Qubits8/train.csv")
df = pd.read_csv(data_path)
n_bits = 8
L1 =4
L_M =3
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


# ------------ ansatz factory ------------

from src.circuits.ansatz1 import hardware_efficient_ansatz
from src.circuits.ansatz2 import ising_structured_ansatz
from src.circuits.ansatz3 import eh2d_ansatz
from src.circuits.ansatz4 import mi_ansatz
## Control # of Params around 100
from src.train.mmd2 import build_mmdagg_prob
d = 256
mmd_eval = build_mmdagg_prob(
    d,
    kernel="laplace_gaussian",        # or "all", "all_matern_l1_l2", ...
    number_bandwidths=5,
    weights_type="centred",
    dtype=jnp.float64,
    return_details=False,             # True if you want per-test info
    use_sqrt=False,                   # True if you want MMD (not MMD^2)
)



from src.circuits.specs import get_ansatz_spec

bit_cols = [f"q{i}" for i in range(n_bits)]
ANSA_FN, L, pc, meta = get_ansatz_spec(
    ansatz_id=ANSATZ_ID,   # whatever you parse
    n_bits=n_bits,
    R=2, C=4,              # your grid for ansatz2/3
    L1=4, L_M=3,           # your layer choices
    keep_edges=16, extras=4,  # ansatz4 settings
    train_df=df, bit_cols=bit_cols   # needed for ansatz4
)
print(f"[info] Using {meta['name']} with L={L}, param_count={pc}")





model = QCBM(ansatz=ANSA_FN, n_qubits=n_bits, L=L, mmd_fn=mmd_eval, target_probs = target_probs)
model.build_circuits()
key = jax.random.PRNGKey(0)
params = jax.random.normal(key, shape=(pc,))

from jax import remat, value_and_grad, jit, lax
import optax

lr_sched = optax.exponential_decay(  # 从 1e‑2 → 每 200 步 × 0.9
    init_value=1e-2, transition_steps=200, decay_rate=0.9, staircase=True
)
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr_sched, b2=0.999)
)

def single_loss(p):
    return model.loss(p)

batched_loss = jax.vmap(single_loss, in_axes=(0,))

def batch_loss_fn(params_batch):
    losses, metrics = batched_loss(params_batch)  # shape (batch_size,)
    mean_loss = jnp.mean(losses)
    mean_metrics = {k: jnp.mean(v) for k, v in metrics.items()}
    return mean_loss, mean_metrics

grad_fn = jax.jit(jax.value_and_grad(batch_loss_fn, has_aux = True))

import time, numpy as np
from jax.experimental import io_callback  # add `# type: ignore` if Pylance nags

def _now_host(_=None):
    # MUST be a 0-D NumPy scalar, not a vector
    return np.array(time.perf_counter(), dtype=np.float64)

@jax.jit
def update_step(i, params, opt_state,
                loss_log, mmd_log, kl_log, params_log, grad_log, grad_norm_log, time_log):
    t0 = io_callback(_now_host, jax.ShapeDtypeStruct((), jnp.float64), None)

    (loss_val, aux), grads = grad_fn(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    t1 = io_callback(_now_host, jax.ShapeDtypeStruct((), jnp.float64), None)
    dt = t1 - t0
    dt_ms = dt * jnp.array(1e3, jnp.float64)
    gnorm = optax.global_norm(grads)

    loss_log      = loss_log.at[i].set(loss_val)
    mmd_log       = mmd_log.at[i].set(aux["mmd"])
    kl_log        = kl_log.at[i].set(aux["kl"])
    params_log    = params_log.at[i].set(params)
    grad_log      = grad_log.at[i].set(grads)
    grad_norm_log = grad_norm_log.at[i].set(gnorm)
    time_log      = time_log.at[i].set(dt_ms)  # OK now: scalar into (n_steps,)

    jax.debug.print("Step {i}: loss={loss:.6f} mmd={mmd:.6f} kl={kl:.6f} ||g||={gn:.6f} dt_ms={dt_ms:.3f}ms",
                    i=i, loss=loss_val, mmd=aux["mmd"], kl=aux["kl"], gn=gnorm, dt_ms=dt_ms)
    return params, opt_state, loss_log, mmd_log, kl_log, params_log, grad_log, grad_norm_log, time_log

def _train(params, n_steps: int, chunk: int):
    opt_state = opt.init(params)
    dtype = params.dtype

    loss_log      = jnp.zeros((n_steps,), dtype=dtype)
    mmd_log       = jnp.zeros((n_steps,), dtype=dtype)
    kl_log        = jnp.zeros((n_steps,), dtype=dtype)
    params_log    = jnp.zeros((n_steps,) + params.shape, dtype=dtype)
    grad_log      = jnp.zeros((n_steps,) + params.shape, dtype=dtype)
    grad_norm_log = jnp.zeros((n_steps,), dtype=dtype)
    time_log      = jnp.zeros((n_steps,), dtype=jnp.float64)

    def body(i, carry):
        (params, opt_state, loss_h, mmd_h, kl_h, params_h, grads_h, gnorm_h, time_h) = carry
        return update_step(i, params, opt_state, loss_h, mmd_h, kl_h, params_h, grads_h, gnorm_h, time_h)

    (params, opt_state, loss_log, mmd_log, kl_log, params_log, grad_log, grad_norm_log, time_log) = \
        jax.lax.fori_loop(0, n_steps, body,
                          (params, opt_state, loss_log, mmd_log, kl_log, params_log, grad_log, grad_norm_log, time_log))

    logs = {"loss": loss_log, "mmd": mmd_log, "kl": kl_log,
            "params": params_log, "grads": grad_log, "grad_norm": grad_norm_log,
            "step_time": time_log}
    return params, opt_state, logs


if __name__ == "__main__":
    batch_size = 8  
    steps = 10000
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, shape=(batch_size, pc))  # (batch_size, num_params)
    params = jax.device_put(params, gpu)
    target_probs = jax.device_put(jnp.asarray(probs_full.values, dtype=jnp.float64), gpu)

    trained_params, opt_state, logs = _train(params, steps, 50)

    jax.block_until_ready(trained_params)      # wait until training ends

    # ───── save ─────
    from pathlib import Path
    run_id = f"result{ANSATZ_ID}"


    out_dir = Path("data/results/Qubits8")/run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"final_params.npy",  jax.device_get(trained_params))
    np.save(out_dir / f"loss.npy", jax.device_get(logs["loss"]))
    np.save(out_dir / f"mmd.npy", jax.device_get(logs["mmd"]))
    np.save(out_dir / f"kl.npy", jax.device_get(logs["kl"]))
    np.save(out_dir / f"params.npy", jax.device_get(logs["params"]))
    np.save(out_dir / f"grads.npy", jax.device_get(logs["grads"]))
    np.save(out_dir / f"grad_norm.npy", jax.device_get(logs["grad_norm"]))
    np.save(out_dir / f"step_time.npy", jax.device_get(logs["step_time"]))



    print(f"model params have saved to: {out_dir / 'fianl_params.npy'}")
    print(f"Loss record has saved to: {out_dir / '/'}")