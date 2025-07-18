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

        self.dev = device or qml.device("lightning.gpu", wires=n_qubits, shots=shots)
        self.dev2 = qml.device("lightning.gpu", wires = n_qubits, shots = 10_000)

        
        @qml.qjit
        @qml.qnode(self.dev, interface="jax")
        def circuit(params):
            self.ansatz(params, wires = range(self.n_qubits), L = self.L)
            return qml.probs(wires=range(self.n_qubits))
        
        self.circuit = circuit
        
        @qml.qjit
        @qml.qnode(self.dev2, interface="jax")
        def circuit2(params):
            self.ansatz(params, wires= range(self.n_qubits), L = self.L)
            return qml.sample()
        
        self.circuit2 =  circuit2

    def loss(self, params):
        params = (
            params if isinstance(params, (jnp.ndarray, jax.Array))
            else jnp.array(params)
        ) 
        qcbm_probs = self.circuit(params)

        loss_mmd= self.mmd_fn(
            self.target_probs,
            qcbm_probs,
            kernel="laplace_gaussian",
            number_bandwidths=10,
            weights_type="uniform",
            build_details=False,  
            dtype=self.dtype,
        )
        def kl_div(p, q, eps=1e-10):
            """
            KL(p || q)  for discrete probs.
            - 两边都 clip 到 [eps, 1] 再归一化，保证 grad 和数值稳定。
            - 返回标量 (float64 if inputs are float64)
            """
            p = jnp.clip(p, eps, 1.0)
            q = jnp.clip(q, eps, 1.0)
            p = p / p.sum()
            q = q / q.sum()
            return jnp.sum(p * jnp.log(p / q))

        loss = loss_mmd * 0.7 + kl_div(self.target_probs, qcbm_probs) * 0.3
        return loss
    

