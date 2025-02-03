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
        HOND-Kuramoto Model: Synchronization Variability Analysis Framework
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

def HONE_worker_with_damped_kuramoto_cpu(adj_matrix, dim, iterations, tol, seed, dt, gamma, gamma_theta, K):
    """
    Simulate a network of harmonic oscillators with damped Kuramoto phase synchronization (CPU).

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

def compute_community_velocity_variability(G, ensemble_positions):
    """
    Compute velocity-based synchronization variability at the community level.

    This function calculates the variance of velocity magnitudes within each Louvain-detected 
    community and takes the mean variance across all communities.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "community_velocity_variance": Variance of velocity magnitudes within each community.
    """
    # Detect Louvain communities (with fixed seed for reproducibility)
    communities = sorted(list(louvain_partitions(G, seed=42))[-1], key=lambda x: min(x))
    community_velocity_variance = {}

    def compute_velocity(positions):
        return np.linalg.norm(positions[-1] - positions[-2], axis=1)

    # Compute velocities for all ensemble realizations using multi-threading
    with ThreadPoolExecutor() as executor:
        velocities = list(executor.map(compute_velocity, ensemble_positions))

    velocities = np.array(velocities)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per community
    for i, community in enumerate(communities):
        indices = [list(G.nodes).index(node) for node in community]
        velocity_variance_per_community = np.var(velocities[:, indices])
        community_velocity_variance[f"Community {i}"] = float(velocity_variance_per_community)

    return community_velocity_variance


def compute_community_phase_variability(G, ensemble_phases):
    """
    Compute phase-based synchronization variability at the community level.

    This function calculates the variance of phase values within each Louvain-detected 
    community and takes the mean variance across all communities.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "community_phase_variance": Variance of phase values within each community.
    """
    # Detect Louvain communities (with fixed seed for reproducibility)
    communities = sorted(list(louvain_partitions(G, seed=42))[-1], key=lambda x: min(x))
    community_phase_variance = {}

    phases = np.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per community
    for i, community in enumerate(communities):
        indices = [list(G.nodes).index(node) for node in community]
        phase_variance_per_community = np.var(phases[:, indices])
        community_phase_variance[f"Community {i}"] = float(phase_variance_per_community)

    return community_phase_variance


def compute_community_laplacian_variability(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability at the community level.

    This function evaluates synchronization stability by analyzing the variance of the 
    second smallest Laplacian eigenvalue (algebraic connectivity, λ₂) and its associated 
    eigenvector (Fiedler vector, v₂) within each community detected by the Louvain method.

    Unlike a global analysis, this function ensures that each community is treated 
    as an independent subgraph, allowing us to measure intra-community synchronization.

    Parameters:
        G (networkx.Graph): The input graph representing the network structure.
        ensemble_positions (list of lists): 
            - List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): 
            - List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "community_lambda_2_variance": Variance of the second smallest Laplacian eigenvalue within each community.
            - "community_v2_variance": Variance of the second eigenvector components within each community.
    """

    # 1. Detect Louvain communities (using fixed seed for reproducibility)
    communities = sorted(list(louvain_partitions(G, seed=42))[-1], key=lambda x: min(x))
    
    # 2. Initialize dictionaries to store community-based variances
    community_lambda_2_variance = {}
    community_v2_variance = {}

    # 3. Iterate over each detected community
    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)  # Extract subgraph for the community
        L_sub = nx.laplacian_matrix(subgraph).toarray()  # Compute Laplacian matrix of the community
        
        eigenvalues_sub_list = []
        eigenvectors_sub_list = []

        # 4. Compute Laplacian eigenvalues and eigenvectors for each ensemble realization
        for _ in ensemble_positions:
            eigvals, eigvecs = eigh(L_sub)  # Eigen decomposition
            eigenvalues_sub_list.append(eigvals[1])  # Extract only λ₂ (Algebraic connectivity)
            eigenvectors_sub_list.append(eigvecs[:, 1])  # Extract only v₂ (Fiedler vector)

        # Convert lists to NumPy arrays for variance computation
        eigenvalues_sub_list = np.array(eigenvalues_sub_list)  # Shape: (ensemble_size,)
        eigenvectors_sub_list = np.array(eigenvectors_sub_list)  # Shape: (ensemble_size, num_nodes_in_community)

        # 5. Compute variance **within the community** across all ensemble realizations
        lambda_2_variance_per_community = np.var(eigenvalues_sub_list)
        v2_variance_per_community = np.var(eigenvectors_sub_list)

        # Store the computed variances
        community_lambda_2_variance[f"Community {i}"] = float(lambda_2_variance_per_community)
        community_v2_variance[f"Community {i}"] = float(v2_variance_per_community)

    return {
        "community_lambda_2_variance": community_lambda_2_variance,
        "community_v2_variance": community_v2_variance
    }
