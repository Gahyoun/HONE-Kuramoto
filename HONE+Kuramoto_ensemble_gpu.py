import cupy as cp
import networkx as nx
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor  # Multi-threading for parallel execution

def compute_laplacian_variability_gpu(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability across ensemble realizations using GPU acceleration.

    This function calculates the variance of the Laplacian eigenvalues and eigenvectors 
    across multiple ensemble simulations, leveraging CuPy for GPU-based computations.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "lambda_2_variance": Mean variance of the second smallest Laplacian eigenvalue (algebraic connectivity).
            - "v2_variance": Mean variance of the second eigenvector components (Fiedler vector).
    """
    laplacian_matrix = cp.asarray(nx.laplacian_matrix(G).toarray())  # Move Laplacian matrix to GPU

    eigenvalues_list = []
    eigenvectors_list = []

    # Compute eigenvalues and eigenvectors for each realization
    for _ in ensemble_positions:
        eigvals, eigvecs = cp.linalg.eigh(laplacian_matrix)  # CuPy GPU-accelerated eigen decomposition
        eigenvalues_list.append(eigvals)
        eigenvectors_list.append(eigvecs)

    eigenvalues_list = cp.array(eigenvalues_list)  # Shape: (ensemble_size, num_nodes)
    eigenvectors_list = cp.array(eigenvectors_list)  # Shape: (ensemble_size, num_nodes, num_nodes)

    # Compute variance per realization and take the mean across the ensemble
    lambda_2_variance_per_ensemble = cp.var(eigenvalues_list[:, 1], axis=0)
    v2_variance_per_ensemble = cp.var(eigenvectors_list[:, :, 1], axis=0)

    lambda_2_variance_mean = cp.mean(lambda_2_variance_per_ensemble)
    v2_variance_mean = cp.mean(v2_variance_per_ensemble)

    return {
        "lambda_2_variance": float(lambda_2_variance_mean.get()),  # Move results back to CPU
        "v2_variance": float(v2_variance_mean.get())
    }

def compute_node_variability_gpu(ensemble_positions, ensemble_phases):
    """
    Compute node-based synchronization variability using GPU acceleration.

    This function calculates the variance of velocity magnitudes and phase values at each node
    across ensemble realizations and then takes the mean of these variances.

    Parameters:
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "velocity_variance": Mean variance of velocity magnitudes across nodes.
            - "phase_variance": Mean variance of phase values across nodes.
    """
    # Convert data to CuPy GPU arrays
    velocities = cp.array([
        cp.linalg.norm(cp.asarray(positions[-1]) - cp.asarray(positions[-2]), axis=1)
        for positions in ensemble_positions
    ])  # Shape: (ensemble_size, num_nodes)

    phases = cp.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and take the mean across all nodes
    velocity_variance_per_node = cp.var(velocities, axis=0)
    phase_variance_per_node = cp.var(phases, axis=0)

    velocity_variance_mean = cp.mean(velocity_variance_per_node)
    phase_variance_mean = cp.mean(phase_variance_per_node)

    return {
        "velocity_variance": float(velocity_variance_mean.get()),
        "phase_variance": float(phase_variance_mean.get())
    }

def compute_edge_variability_gpu(G, ensemble_positions, ensemble_phases):
    """
    Compute edge-based synchronization variability using GPU acceleration.

    This function calculates the variance of velocity differences and phase differences
    between connected nodes across ensemble realizations. The mean of these variances
    is computed to quantify synchronization fluctuations at the edge level.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "velocity_diff_variance": Mean variance of velocity differences between connected nodes.
            - "phase_diff_variance": Mean variance of phase differences between connected nodes.
    """
    edges = list(G.edges)
    velocity_diffs = []
    phase_diffs = []

    for positions, phases in zip(ensemble_positions, ensemble_phases):
        # Compute velocity magnitudes for each node using CuPy
        velocity = cp.linalg.norm(cp.asarray(positions[-1]) - cp.asarray(positions[-2]), axis=1)

        # Compute velocity and phase differences for each edge
        velocity_diff = cp.array([cp.abs(velocity[u] - velocity[v]) for u, v in edges])
        phase_diff = cp.array([cp.abs(cp.asarray(phases[-1])[u] - cp.asarray(phases[-1])[v]) for u, v in edges])

        velocity_diffs.append(velocity_diff)
        phase_diffs.append(phase_diff)

    velocity_diffs = cp.array(velocity_diffs)  # Shape: (ensemble_size, num_edges)
    phase_diffs = cp.array(phase_diffs)  # Shape: (ensemble_size, num_edges)

    # Compute variance per edge and take the mean across all edges
    velocity_diff_variance_per_edge = cp.var(velocity_diffs, axis=0)
    phase_diff_variance_per_edge = cp.var(phase_diffs, axis=0)

    velocity_diff_variance_mean = cp.mean(velocity_diff_variance_per_edge)
    phase_diff_variance_mean = cp.mean(phase_diff_variance_per_edge)

    return {
        "velocity_diff_variance": float(velocity_diff_variance_mean.get()),
        "phase_diff_variance": float(phase_diff_variance_mean.get())
    }
