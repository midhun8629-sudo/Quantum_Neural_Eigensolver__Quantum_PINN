# Quantum Neural Eigensolver (Quantum PINN)

## 📌 Overview
This repository implements an unsupervised Physics-Informed Neural Network (PINN) to solve the time-independent Schrödinger equation. By treating the energy eigenvalue ($E$) as a learnable parameter alongside the network weights, this Quantum PINN autonomously discovers both the ground-state wavefunction ($\psi$) and the ground-state energy of a 1D Quantum Harmonic Oscillator, entirely without labeled data.

## 🔬 Scientific Context & Materials Informatics Application
In computational materials science, determining the electronic structure of a material requires solving complex eigenvalue problems. Traditional Density Functional Theory (DFT) solvers, such as **VASP**, rely on iterative numerical matrix diagonalizations over discretized grids or plane-wave basis sets. 

This project explores a fundamentally different, mesh-free paradigm. By using a continuous neural network ansatz to parameterize the wavefunction, we bypass grid-resolution limits. The network uses automatic differentiation to compute exact kinetic energy operators, making it a highly scalable approach that serves as a foundational step toward deep learning-based many-body quantum solvers.

## 🧠 Methodology & Model Architecture
The implementation is contained in `Quantum_Neural_Eigensolver_(Quantum_PINN) (1).ipynb` and consists of two main components:

1. **The Wavefunction Network (Ansatz):**
   * A Multilayer Perceptron (MLP) maps the continuous spatial coordinate $x$ to the probability amplitude $\psi(x)$.
   * **Physical Boundary Conditions:** The raw network output is multiplied by a Gaussian envelope ($e^{-x^2}$) to strictly enforce that the wavefunction decays to zero at infinity, massively accelerating training convergence.
   * **Learnable Eigenvalue:** The scalar energy value $E$ is optimized concurrently as an `nn.Parameter`.

2. **The Physics Loss (Hamiltonian Residual):**
   * **Kinetic Energy:** Computed exactly via double automatic differentiation ($\nabla^2 \psi$).
   * **Potential Energy:** Evaluated using the harmonic oscillator potential ($V(x) = \frac{1}{2}x^2$).
   * **Residual:** Minimizes the variance of the Schrödinger equation $( \hat{H}\psi - E\psi )^2$.
   * **Normalization:** A Monte Carlo integration penalty is added to enforce $\langle\psi|\psi\rangle = 1$, preventing the network from collapsing to the trivial solution ($\psi = 0$).

## 💻 Tech Stack
* **Deep Learning & Autograd:** PyTorch
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib

## 🚀 How to Run
1. Ensure you have Python installed with `torch`, `numpy`, and `matplotlib`.
2. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook "Quantum_Neural_Eigensolver_(Quantum_PINN) (1).ipynb"
