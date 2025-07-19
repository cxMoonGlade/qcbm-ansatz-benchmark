# qcbm-ansatz-benchmark

This repository provides scripts and Jupyter notebooks to train Quantum Circuit Born Machine (QCBM) models with different circuit ansatzes and evaluate their expressivity, generalization, and performance on Ising model datasets.

## How to Use

### 1. Training

* **Select an Ansatz:**
  In `train_processes.ipynb`, navigate to the sixth code cell labeled
  `# ------------ ansatz factory ------------`.

* **Choose Your Ansatz:**
  Set the ansatz name to one of the following options:

  * `hardware_efficient_ansatz`
  * `ising_structured_ansatz`
  * `eh2d_ansatz`
  * `mi_ansatz`

* **Run the Notebook:**
  After choosing your desired ansatz, simply click "Run All" to execute all cells and start the training process.

### 2. FRC Evaluation

* To evaluate **Fidelity, Rate, and Coverage (FRC)**, open and run `last.ipynb`.

## Notes

* The notebooks are designed for flexible experimentation with different ansatz types.
* Please ensure that all required dependencies are installed as specified in your environment.
