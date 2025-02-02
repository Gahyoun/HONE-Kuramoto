import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from scipy.linalg import eigh
from networkx.algorithms.community import louvain_partitions
from concurrent.futures import ThreadPoolExecutor  # Multi-threading for parallel execution

"""
------------------------------------------------------------------------------------
        HONE-Kuramoto Model: Synchronization Variability Analysis Framework
------------------------------------------------------------------------------------
This repository provides a Python implementation of a **Harmonic Oscillator Network Embedding (HONE)** 
model coupled with a **damped Kuramoto model**. The framework analyzes synchronization 
variability using a multi-layer ensemble approach.

📌 **Key Features:**
- **Laplacian-based synchronization analysis** to quantify global coherence.
- **Node-based and phase-based variability** to measure local synchronization stability.
- **Parallelized simulations** using multi-threading for efficient computation.
- **Periodic Boundary Conditions (PBC)** applied for accurate phase modeling.
- **Velocity and phase synchronization metrics** computed across multiple realizations.

📌 **Applications:**
- Network synchronization analysis in biological, social, and engineered systems.
- Evaluating stability in multi-agent systems and coupled oscillators.
- Investigating phase-locking phenomena in large-scale networks.

"""

# ===========================
# 1. COMPUTE SYNCHRONIZATION VARIABILITY METRICS
# ===========================

def compute_velocity_variability(ensemble_positions):
    """
    Compute synchronization variability in velocity magnitudes across multiple realizations.

    The function calculates the variance of velocity magnitudes at each node over 
    different initial conditions (ensemble simulations). The mean variance across 
    all nodes quantifies synchronization stability.

    Parameters:
        ensemble_positions (list of lists): 
            - List of node position histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "velocity_variance": Mean variance of velocity magnitudes across nodes.
    """
    def compute_velocity(positions):
        # Compute velocity magnitudes at each node
        return np.linalg.norm(positions[-1] - positions[-2], axis=1)

    # Compute velocities for all ensemble realizations using multi-threading
    with ThreadPoolExecutor() as executor:
        velocities = list(executor.map(compute_velocity, ensemble_positions))

    velocities = np.array(velocities)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and take the mean across all nodes
    velocity_variance = np.mean(np.var(velocities, axis=0))

    return {
        "velocity_variance": float(velocity_variance)
    }

def compute_phase_variability(ensemble_phases):
    """
    Compute phase-based synchronization variability across multiple realizations.

    This function calculates the variance of phase values at each node across
    ensemble simulations and takes the mean variance across all nodes.

    Parameters:
        ensemble_phases (list of lists): 
            - List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "phase_variance": Mean variance of phase values across nodes.
    """
    phases = np.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and take the mean across all nodes
    phase_variance = np.mean(np.var(phases, axis=0))

    return {
        "phase_variance": float(phase_variance)
    }


def compute_laplacian_variability(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability across ensemble realizations.

    This function calculates the variance of Laplacian eigenvalues and eigenvectors 
    across multiple ensemble simulations to quantify the network’s synchronization stability.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_positions (list of lists): 
            - List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): 
            - List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "lambda_2_variance": Mean variance of the second smallest Laplacian eigenvalue.
            - "v2_variance": Mean variance of the second eigenvector components (Fiedler vector).
    """
    laplacian_matrices = np.array([nx.laplacian_matrix(G).toarray() for _ in ensemble_positions])

    eigenvalues_list = []
    eigenvectors_list = []

    # Compute eigenvalues and eigenvectors for each realization
    for L in laplacian_matrices:
        eigvals, eigvecs = eigh(L)
        eigenvalues_list.append(eigvals)
        eigenvectors_list.append(eigvecs)

    eigenvalues_list = np.array(eigenvalues_list)  # Shape: (ensemble_size, num_nodes)
    eigenvectors_list = np.array(eigenvectors_list)  # Shape: (ensemble_size, num_nodes, num_nodes)

    # Compute variance per realization and take the mean across the ensemble
    lambda_2_variance_per_ensemble = np.var(eigenvalues_list[:, 1], axis=0)
    v2_variance_per_ensemble = np.var(eigenvectors_list[:, :, 1], axis=0)

    lambda_2_variance_mean = np.mean(lambda_2_variance_per_ensemble)
    v2_variance_mean = np.mean(v2_variance_per_ensemble)

    return {
        "lambda_2_variance": lambda_2_variance_mean,
        "v2_variance": v2_variance_mean
    }


# ===========================
# 2. HONE-KURAMOTO MODEL SIMULATION
# ===========================

def HONE_worker_with_damped_kuramoto(adj_matrix, dim, iterations, tol, seed, dt, gamma, gamma_theta, K):
    """
    Simulate a network of harmonic oscillators with damped Kuramoto phase synchronization.

    This function models the evolution of positions and phases of nodes in a network 
    according to the damped Kuramoto model and harmonic oscillator interactions.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        dim (int): Embedding dimension.
        iterations (int): Maximum number of iterations.
        tol (float): Convergence threshold.
        seed (int): Random seed for reproducibility.
        dt (float): Time step.
        gamma (float): Damping coefficient for spatial movement.
        gamma_theta (float): Damping coefficient for phase synchronization.
        K (float): Coupling strength in the Kuramoto model.

    Returns:
        tuple: 
            - positions_history, phase_history, potential_energy, kinetic_energy, total_energy
    """
    np.random.seed(seed)
    num_nodes = adj_matrix.shape[0]

    # Initialize positions, velocities, phases, and intrinsic frequencies
    positions = np.random.rand(num_nodes, dim)
    velocities = np.zeros_like(positions)
    phases = np.random.uniform(0, 2 * np.pi, num_nodes)
    phase_velocities = np.zeros(num_nodes)
    intrinsic_frequencies = np.random.normal(0, 1, num_nodes)

    positions_history, phase_history = [positions.copy()], [phases.copy()]
    potential_energy_history, kinetic_energy_history, total_energy_history = [], [], []

    def calculate_forces(positions):
        """ Compute interaction forces between nodes based on adjacency matrix """
        forces = np.zeros_like(positions)
        for i in range(num_nodes):
            delta = positions - positions[i]
            distances = np.linalg.norm(delta, axis=1)
            mask = distances > 1e-6
            distances[~mask] = max(1e-6, np.min(distances[mask]))  
            forces[i] = np.sum(adj_matrix[i, mask][:, None] * (delta[mask] / distances[mask, None]), axis=0)
        return forces

    # Simulation loop
    for step in range(iterations):
        forces = calculate_forces(positions)
        velocities += forces * dt - gamma * velocities
        new_positions = positions + velocities * dt

        # Apply Periodic Boundary Conditions (PBC) to phase updates
        phase_diffs = np.array([
            np.sum(adj_matrix[i] * np.sin((phases - phases[i] + np.pi) % (2 * np.pi) - np.pi)) 
            for i in range(num_nodes)
        ])

        phase_velocities += (intrinsic_frequencies + K * phase_diffs - gamma_theta * phase_velocities) * dt
        new_phases = np.mod(phases + phase_velocities * dt, 2*np.pi)

        # Update positions and phases
        positions, phases = new_positions, new_phases
        positions_history.append(positions.copy())
        phase_history.append(phases.copy())

    return positions_history, phase_history, potential_energy_history, kinetic_energy_history, total_energy_history
