import pennylane as qml
import jax, jax.numpy as jnp
from functools import partial

def eh2d_ansatz(
    params,
    wires,
    shape=(2, 4),          # (R, C) 网格
    L=4,
    trotter_steps=2,
    add_dt=False,
):
    """Entangled-Hamiltonian (EH) ansatz on a 2-D lattice."""
    R, C = shape
    n = 8

    ptr = 0
    for _ in range(L):
        # ----- 1. 取本层所有可训练参数 ------------------------
        hx = params[ptr : ptr + n] ; ptr += n          # local X
        hz = params[ptr : ptr + n] ; ptr += n          # local Z

        J_row = params[ptr : ptr + R * (C - 1)]        # 横向耦合
        ptr += R * (C - 1)

        J_col = params[ptr : ptr + (R - 1) * C]        # 纵向耦合
        ptr += (R - 1) * C

        if add_dt:
            dt = jnp.abs(params[ptr]); ptr += 1
        else:
            dt = 1.0

        # ----- 2. 拼 Hamiltonian ----------------------------
        coeffs, ops = [], []

        # local terms
        for i, w in enumerate(wires):
            coeffs += [hx[i], hz[i]]
            ops    += [qml.PauliX(w), qml.PauliZ(w)]

        # row‑wise ZZ
        k = 0
        for r in range(R):
            for c in range(C - 1):
                a = r * C + c
                b = a + 1
                coeffs.append(J_row[k])
                ops.append(qml.PauliZ(wires[a]) @ qml.PauliZ(wires[b]))
                k += 1

        # column‑wise ZZ
        k = 0
        for r in range(R - 1):
            for c in range(C):
                a = r * C + c
                b = a + C
                coeffs.append(J_col[k])
                ops.append(qml.PauliZ(wires[a]) @ qml.PauliZ(wires[b]))
                k += 1

        H = qml.Hamiltonian(coeffs, ops)

        # ----- 3. Trotter 演化 -------------------------------
        qml.ApproxTimeEvolution(H, dt, trotter_steps)
