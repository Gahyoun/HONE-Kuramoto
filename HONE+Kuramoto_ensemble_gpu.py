import cupy as cp
import networkx as nx
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor  # Multi-threading for parallel execution

import cupy as cp
from concurrent.futures import ThreadPoolExecutor  # For multi-threading across ensemble realizations

def compute_velocity_variability_gpu(ensemble_positions):
    """
    Compute velocity-based synchronization variability using GPU.

    This function calculates the variance of velocity magnitudes at each node
    across ensemble realizations and then takes the mean of these variances.

    Parameters:
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "velocity_variance": Mean variance of velocity magnitudes across nodes.
    """
    def compute_velocity(positions):
        # Calculate velocity magnitudes for each node
        return cp.linalg.norm(cp.asarray(positions[-1]) - cp.asarray(positions[-2]), axis=1)

    # Use multi-threading to compute velocities for all ensemble realizations
    with ThreadPoolExecutor() as executor:
        velocities = list(executor.map(compute_velocity, ensemble_positions))

    velocities = cp.array(velocities)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and take the mean across all nodes
    velocity_variance = cp.mean(cp.var(velocities, axis=0))

    return {
        "velocity_variance": float(velocity_variance.get())  # Move result back to CPU
    }

def compute_phase_variability_gpu(ensemble_phases):
    """
    Compute phase-based synchronization variability using GPU.

    This function calculates the variance of phase values at each node
    across ensemble realizations and then takes the mean of these variances.

    Parameters:
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "phase_variance": Mean variance of phase values across nodes.
    """
    phases = cp.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and take the mean across all nodes
    phase_variance = cp.mean(cp.var(phases, axis=0))

    return {
        "phase_variance": float(phase_variance.get())  # Move result back to CPU
    }

def compute_laplacian_variability_gpu(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability across ensemble realizations using GPU.

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
        eigvals, eigvecs = cp.linalg.eigh(laplacian_matrix)  # GPU-accelerated eigen decomposition
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
