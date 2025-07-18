import pennylane as qml
import numpy as np

def ising_structured_ansatz(params, wires, shape = (2, 4), L = 4, periodic = False):
    n = 8
    R, C  = shape
    ptr = 0
    for layer in range(L):
        for w in wires:
            qml.RZ(params[ptr], wires = w)
            ptr += 1
            qml.RX(params[ptr], wires= w)
            ptr += 1
        for r in range(R):
            for c in range(C - 1):
                a = r * C + c
                b = a+1
                qml.IsingZZ(params[ptr], wires = [wires[a], wires[b]])
                ptr += 1
            if periodic:
                a = r*C + (C-1) 
                b = r*C
                qml.IsingZZ(params[ptr], wires=[wires[a], wires[b]])
                ptr += 1

        for r in range(R-1):
            for c in range(C):
                a = r * C + c
                b = a + C
                qml.IsingZZ(params[ptr], wires=[wires[a], wires[b]])
                ptr += 1
            if periodic:
                for c in range(C):
                    a = (R-1)*C + c
                    b = c
                    qml.IsingZZ(params[ptr], wires=[wires[a], wires[b]])
                    ptr += 1