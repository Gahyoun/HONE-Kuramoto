import cupy as cp
import networkx as nx
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor  # Multi-threading for parallel execution

def compute_laplacian_variability_gpu(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability across ensemble realizations using GPU acceleration.

    Parameters:
        G (networkx.Graph): Input graph representing the network structure.
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: Variance of Laplacian eigenvalues and eigenvectors across ensemble realizations.
    """
    laplacian_matrix = cp.asarray(nx.laplacian_matrix(G).toarray())  # GPU-enabled Laplacian matrix

    eigenvalues_list = []
    eigenvectors_list = []

    # Compute eigenvalues and eigenvectors for each realization
    for _ in ensemble_positions:
        eigvals, eigvecs = cp.linalg.eigh(laplacian_matrix)  # GPU-accelerated eigen decomposition
        eigenvalues_list.append(eigvals)
        eigenvectors_list.append(eigvecs)

    eigenvalues_list = cp.array(eigenvalues_list)  # Shape: (ensemble_size, num_nodes)
    eigenvectors_list = cp.array(eigenvectors_list)  # Shape: (ensemble_size, num_nodes, num_nodes)

    # Compute variance and mean across realizations
    lambda_2_variance = cp.var(eigenvalues_list[:, 1])
    v2_variance = cp.var(eigenvectors_list[:, :, 1])

    return {
        "lambda_2_variance": float(lambda_2_variance.get()),  # Convert from GPU to CPU
        "v2_variance": float(v2_variance.get())
    }

def compute_velocity_variability_gpu(ensemble_positions):
    """
    Compute velocity synchronization variability across nodes using GPU acceleration.

    Parameters:
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.

    Returns:
        dict: Mean variance of velocity magnitudes across nodes.
    """
    velocities = cp.array([
        cp.linalg.norm(cp.asarray(positions[-1]) - cp.asarray(positions[-2]), axis=1)
        for positions in ensemble_positions
    ])  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and mean across all nodes
    velocity_variance_per_node = cp.var(velocities, axis=0)
    velocity_variance_mean = cp.mean(velocity_variance_per_node)

    return {"velocity_variance": float(velocity_variance_mean.get())}

def compute_phase_variability_gpu(ensemble_phases):
    """
    Compute phase synchronization variability across nodes using GPU acceleration.

    Parameters:
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: Mean variance of phase values across nodes.
    """
    phases = cp.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    # Compute variance per node and mean across all nodes
    phase_variance_per_node = cp.var(phases, axis=0)
    phase_variance_mean = cp.mean(phase_variance_per_node)

    return {"phase_variance": float(phase_variance_mean.get())}
