import jax
import jax.numpy as jnp
import pennylane as qml

class QCBM:
    def __init__(
            self, 
            ansatz, 
            n_qubits: int, 
            L: int, 
            mmd_fn,
            target_probs, 
            device = None, 
            shots = None,
            dtype=jnp.float64, 
        ):
        self.ansatz = ansatz
        self.n_qubits = n_qubits
        self.L = L
        self.mmd_fn = mmd_fn
        self.shots = shots
        self.dtype = dtype

        self.target_probs = (
            target_probs if isinstance(target_probs, (jnp.ndarray, jax.Array))
            else jnp.array(target_probs)
        )

        self.circuit = None
        self.circuit2 = None
        
    def build_circuits(self):
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        dev2 = qml.device("default.qubit", wires = self.n_qubits, shots = 10_000)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(params):
            self.ansatz(params, wires = range(self.n_qubits), L = self.L)
            return qml.probs(wires=range(self.n_qubits))
        
        self.circuit = circuit
        
        # @qml.qjit
        @qml.qnode(dev2, interface="jax", )
        def circuit2(params):
            self.ansatz(params, wires= range(self.n_qubits), L = self.L)
            return qml.sample()
        
        self.circuit2 =  circuit2

    def loss(self, params):
        if self.circuit is None:
            raise RuntimeError("Call build_circuits() before using loss or circuit")

        params = params if isinstance(params, (jnp.ndarray, jax.Array)) else jnp.array(params)
        qcbm_probs = self.circuit(params)

        # match the evaluator dtype (you built it with dtype=float64)
        p = self.target_probs.astype(jnp.float64)
        q = qcbm_probs.astype(jnp.float64)

        loss_mmd = self.mmd_fn(p, q)

        def kl_div(p, q, eps=1e-10):
            p = jnp.clip(p, eps, 1.0); q = jnp.clip(q, eps, 1.0)
            p = p / p.sum();           q = q / q.sum()
            return jnp.sum(p * jnp.log(p / q))

        loss_kl = kl_div(self.target_probs, qcbm_probs)
        loss = 0.5 * loss_mmd + 0.5 * loss_kl
        return loss, {"mmd": loss_mmd, "kl": loss_kl}

    

