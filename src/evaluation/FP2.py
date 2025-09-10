import pennylane as qml, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import itertools, sys, os
sys.path.append('../..')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)
shots = None


def make_state_fn(ansatz, hyper_kwargs):
    @qml.qnode(dev, interface="jax")
    def circuit(params):
        ansatz(params, wires=range(n_qubits), **hyper_kwargs)
        return qml.state()
    return circuit

def frame_potential_2(states):
    states = states / jnp.linalg.norm(states, axis=1, keepdims=True)
    overlap = states @ states.conj().T
    fp2 = jnp.mean(jnp.abs(overlap) ** 4)
    return float(fp2)

from src.circuits.ansatz1 import hardware_efficient_ansatz
from src.circuits.ansatz2 import ising_structured_ansatz
from src.circuits.ansatz3 import eh2d_ansatz
from src.circuits.ansatz4 import mi_ansatz
from src.train.training import ansatz_set

ansatzs = [hardware_efficient_ansatz, ising_structured_ansatz, eh2d_ansatz, mi_ansatz]
param_sampler = lambda k, shape: jax.random.normal(k, shape, dtype=jnp.float64)
K = 500
all_fp2 = []
labels = []
for a in ansatzs:
    ansatz_fn, L, pc, idx = ansatz_set(a)
    circ = make_state_fn(ansatz_fn, {"L": L})
    key = jax.random.PRNGKey(0)
    params_all = param_sampler(key, (K, pc))   # (K, pc)
    circ_vmapped = jax.vmap(circ)
    states = circ_vmapped(params_all)          # (K, 2**n)
    fp2 = frame_potential_2(states)
    haar_fp2 = (3*(2**n_qubits) - 2) / ((2**n_qubits) * (2**n_qubits + 1))
    express = abs(fp2 - haar_fp2)
    print(f"Ansatz {idx}: FP2={fp2:.4e}  |ΔFP2|={express:.4e}")
    fp2s = []
    for i in range(K):
        others = jnp.concatenate([states[:i], states[i+1:]], axis=0)
        overlap = states[i] @ others.conj().T
        fp2 = jnp.mean(jnp.abs(overlap) ** 4)
        fp2s.append(float(fp2))
    all_fp2.append(fp2s)
    labels.append(a)



# 画histogram

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
for fp2s, label in zip(all_fp2, labels):
    sns.histplot(fp2s, label=label, bins=40, kde=True, stat='density', element='step')
haar_fp2 = (3 * (2 ** n_qubits) - 2) / ((2 ** n_qubits) * (2 ** n_qubits + 1))
plt.axvline(haar_fp2, ls="--", color="black", label="Haar")
plt.xlabel("Frame Potential")
plt.ylabel("Density")
plt.title("Frame Potential Distribution (Expressibility)")
plt.legend()
plt.tight_layout()
plt.show()