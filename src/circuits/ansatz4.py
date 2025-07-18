import pennylane as qml

def mi_ansatz(params, wires, *, mi_edges, L=3):
    """MI‑Screened ZZ ansatz (≈100 params for n=8, k=24, L=3)"""
    n, ptr = 8, 0
    for _ in range(L):
        # single‑qubit rotations
        for w in wires:
            qml.RZ(params[ptr], wires=w); ptr += 1
            qml.RX(params[ptr], wires=w); ptr += 1
        # screened entanglers
        for (i,j) in mi_edges:
            qml.IsingZZ(params[ptr], wires=[wires[i], wires[j]]); ptr += 1
    # 4 extra global RZ’s
    for k in range(4):
        qml.RZ(params[ptr], wires=wires[k % n]); ptr += 1