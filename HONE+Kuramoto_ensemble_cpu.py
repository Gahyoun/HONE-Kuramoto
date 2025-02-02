import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor  # Explicitly using multi-threading for parallel execution

def compute_laplacian_variability(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability across ensemble realizations.

    This function calculates the variance of the Laplacian eigenvalues and eigenvectors 
    across multiple ensemble simulations to quantify the synchronization stability of the network.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "lambda_2_variance": Mean variance of the second smallest Laplacian eigenvalue (algebraic connectivity).
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

def compute_node_variability(ensemble_positions, ensemble_phases):
    """
    Compute node-based synchronization variability.

    This function calculates the variance of velocity magnitudes and phase values at each node
    across ensemble realizations and then takes the mean of these variances to quantify the
    synchronization fluctuations at the node level.

    Parameters:
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: A dictionary containing:
            - "velocity_variance": Mean variance of velocity magnitudes across nodes.
            - "phase_variance": Mean variance of phase values across nodes.
    """
    # Compute velocity magnitudes for each ensemble realization
    velocities = np.array([
        np.linalg.norm(positions[-1] - positions[-2], axis=1)
        for positions in ensemble_positions
    ])  # Shape: (ensemble_size, num_nodes)

    # Convert phase data into a NumPy array
    phases = np.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and take the mean across all nodes
    velocity_variance_per_node = np.var(velocities, axis=0)
    phase_variance_per_node = np.var(phases, axis=0)

    velocity_variance_mean = np.mean(velocity_variance_per_node)
    phase_variance_mean = np.mean(phase_variance_per_node)

    return {
        "velocity_variance": velocity_variance_mean,
        "phase_variance": phase_variance_mean
    }

def compute_edge_variability(G, ensemble_positions, ensemble_phases):
    """
    Compute edge-based synchronization variability.

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
        # Compute velocity magnitudes for each node
        velocity = np.linalg.norm(positions[-1] - positions[-2], axis=1)

        # Compute velocity differences for each edge
        velocity_diff = np.array([np.abs(velocity[u] - velocity[v]) for u, v in edges])
        phase_diff = np.array([np.abs(phases[-1][u] - phases[-1][v]) for u, v in edges])

        velocity_diffs.append(velocity_diff)
        phase_diffs.append(phase_diff)

    velocity_diffs = np.array(velocity_diffs)  # Shape: (ensemble_size, num_edges)
    phase_diffs = np.array(phase_diffs)  # Shape: (ensemble_size, num_edges)

    # Compute variance per edge and take the mean across all edges
    velocity_diff_variance_per_edge = np.var(velocity_diffs, axis=0)
    phase_diff_variance_per_edge = np.var(phase_diffs, axis=0)

    velocity_diff_variance_mean = np.mean(velocity_diff_variance_per_edge)
    phase_diff_variance_mean = np.mean(phase_diff_variance_per_edge)

    return {
        "velocity_diff_variance": velocity_diff_variance_mean,
        "phase_diff_variance": phase_diff_variance_mean
    }
