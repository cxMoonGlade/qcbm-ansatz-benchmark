import pennylane as qml
import numpy as np

def hardware_efficient_ansatz(params, wires, L):
    n = 8
    ptr = 0

    for l in range(L):
        qml.RX(params[ptr], wires= 0)
        ptr += 1
        qml.RZ(params[ptr], wires= 0)
        ptr += 1

        for i in range(1, n - 1):
            qml.RZ(params[ptr], wires=i)
            ptr += 1
            qml.RX(params[ptr], wires=i)
            ptr += 1

        qml.RX(params[ptr], wires= n-1)
        ptr += 1
        qml.RZ(params[ptr], wires= n-1)
        ptr += 1
        qml.RX(params[ptr], wires= n-1)
        ptr += 1
        
        # Nearest-neighbor entangling layer with XX gates
        for i in range(n - 1):
            qml.IsingXX(params[ptr], wires=[i, i+1])
            ptr += 1
