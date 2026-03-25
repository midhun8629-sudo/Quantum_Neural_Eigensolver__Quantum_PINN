# Adaptive Multi-Defect Identification via Inverse PINNs

## 📌 Overview
This repository contains a PyTorch-based computational mechanics framework designed to solve complex inverse problems in solid mechanics. Specifically, it employs an **Inverse Physics-Informed Neural Network (PINN)** to autonomously detect, localize, and quantify multiple hidden defects (e.g., micro-voids, material degradation, structural damage) within a continuous material domain by analyzing its spatial deformation field.

This approach bridges deep learning and non-destructive evaluation (NDE), offering a purely mesh-free, data-driven alternative to traditional iterative finite element updating methods.

## 🔬 Scientific Context & Materials Engineering Application
In structural health monitoring and materials characterization, internal damage often cannot be directly observed. Instead, engineers must infer the location and severity of defects from measurable surface displacements or strain fields. 

Traditional inverse solvers rely on computationally expensive gradient-based optimization over heavy Finite Element Method (FEM) meshes. This project utilizes a deep learning surrogate to invert the governing physical equations. The network takes the observable physical state (deformation) as an input and outputs the underlying spatial parameter field (damage distribution).

## 🧠 Methodology & Model Architecture
The primary implementation is located in: `Adaptive_Multi_Defect_Identification_(Inverse_PINN) (1).ipynb`

The inverse PINN pipeline is structured as follows:
1.  **Synthetic Data Generation (Forward Proxy):** * Simulates a continuous material domain containing a highly parameterized topology of defects (varying in $(x, y)$ coordinates, radius $r$, and severity $val$).
    * Generates the synthetic surface deformation mapping resulting from these internal weak spots.
2.  **Inverse Physics-Informed Architecture:**
    * A dense Multilayer Perceptron (MLP) built in PyTorch.
    * **Inputs:** Spatial coordinates $(x, y)$ and the corresponding measured deformation/displacement.
    * **Outputs:** The predicted spatial damage parameter (e.g., reduction in localized stiffness).
3.  **Loss Formulation:**
    * Minimizes the discrepancy between the measured deformation and the PINN's parameterized structural predictions.
    * Utilizes adaptive weighting to handle the multi-scale nature of simultaneous minor and severe defects.

## 💻 Tech Stack
* **Deep Learning Framework:** PyTorch
* **Mathematics & Matrix Operations:** NumPy
* **Visualization (Contour & Field Plots):** Matplotlib
* **Paradigm:** Mesh-free Inverse Modeling, Physics-Informed Machine Learning

## 🚀 How to Run
1.  Ensure your environment has Python 3.8+ installed with `torch`, `numpy`, and `matplotlib`.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Adaptive_Multi_Defect_Identification_(Inverse_PINN) (1).ipynb"
    ```
3.  Execute the cells sequentially to:
    * Initialize the multi-defect array (e.g., center weak spot, off-axis severe damage).
    * Generate the synthetic deformation tensor.
    * Train the inverse neural network to back-calculate the defect locations.
    * Visualize the predicted damage field versus the exact synthetic ground truth.

## 📊 Evaluation Metrics
The accuracy of the inverse solver is quantified using:
* **Defect Localization Error:** The spatial offset between the predicted and actual defect centroids.
* **Severity Quantification:** The error in predicted localized stiffness reduction.
* **Global $L_2$ Norm:** The overall topological reconstruction accuracy across the continuous 2D domain.