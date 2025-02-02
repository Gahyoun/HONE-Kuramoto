import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor

def compute_laplacian_variability(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability across ensemble realizations.

    Parameters:
        G (networkx.Graph): Input graph.
        ensemble_positions (list of lists): List of position histories for each ensemble realization.
        ensemble_phases (list of lists): List of phase histories for each ensemble realization.

    Returns:
        dict: Variance of Laplacian eigenvalues and eigenvectors across ensemble.
    """
    laplacian_matrices = np.array([nx.laplacian_matrix(G).toarray() for _ in ensemble_positions])
    eigenvalues_list = []
    eigenvectors_list = []

    for L in laplacian_matrices:
        eigvals, eigvecs = eigh(L)
        eigenvalues_list.append(eigvals)
        eigenvectors_list.append(eigvecs)

    eigenvalues_list = np.array(eigenvalues_list)
    eigenvectors_list = np.array(eigenvectors_list)

    lambda_2_variance = np.var(eigenvalues_list[:, 1])  # Variance of algebraic connectivity
    v2_variance = np.var(eigenvectors_list[:, :, 1], axis=0)  # Variance of the second eigenvector components

    return {"lambda_2_variance": lambda_2_variance, "v2_variance": v2_variance}

def compute_node_variability(ensemble_positions, ensemble_phases):
    """
    Compute node-based synchronization variability.

    Parameters:
        ensemble_positions (list of lists): List of position histories for each ensemble realization.
        ensemble_phases (list of lists): List of phase histories for each ensemble realization.

    Returns:
        dict: Variance of velocity magnitudes and phase values at each node.
    """
    velocities = np.array([[np.linalg.norm(positions[-1] - positions[-2], axis=1) for positions in ensemble_positions]])
    phases = np.array(ensemble_phases)

    velocity_variance = np.var(velocities, axis=0)
    phase_variance = np.var(phases, axis=0)

    return {"velocity_variance": velocity_variance, "phase_variance": phase_variance}

def compute_edge_variability(G, ensemble_positions, ensemble_phases):
    """
    Compute edge-based synchronization variability.

    Parameters:
        G (networkx.Graph): Input graph.
        ensemble_positions (list of lists): List of position histories for each ensemble realization.
        ensemble_phases (list of lists): List of phase histories for each ensemble realization.

    Returns:
        dict: Variance of velocity differences and phase differences between connected nodes.
    """
    edges = list(G.edges)
    velocity_diffs = []
    phase_diffs = []

    for positions, phases in zip(ensemble_positions, ensemble_phases):
        velocity = np.linalg.norm(positions[-1] - positions[-2], axis=1)

        velocity_diff = np.array([np.abs(velocity[u] - velocity[v]) for u, v in edges])
        phase_diff = np.array([np.abs(phases[-1][u] - phases[-1][v]) for u, v in edges])

        velocity_diffs.append(velocity_diff)
        phase_diffs.append(phase_diff)

    velocity_diff_variance = np.var(velocity_diffs, axis=0)
    phase_diff_variance = np.var(phase_diffs, axis=0)

    return {"velocity_diff_variance": velocity_diff_variance, "phase_diff_variance": phase_diff_variance}

def HONE_kuramoto_analysis(G, dim=2, iterations=100, ensemble_size=100, tol=1e-4, dt=0.01, gamma=1.0, gamma_theta=0.1, K=0.5):
    """
    Perform HONE with Kuramoto Model and analyze synchronization variability across ensemble simulations.

    Parameters:
        G (networkx.Graph): Input graph.
        dim (int): Number of dimensions for the embedding space.
        iterations (int): Maximum number of iterations for each simulation.
        ensemble_size (int): Number of ensemble realizations (seeds).
        tol (float): Convergence threshold.
        dt (float): Time step.
        gamma (float): Damping coefficient for spatial movement.
        gamma_theta (float): Damping coefficient for phase synchronization.
        K (float): Coupling strength in the Kuramoto model.

    Returns:
        dict: Synchronization variability analysis results.
    """
    # Convert the graph to an adjacency matrix
    adj_matrix = nx.to_numpy_array(G, weight="weight")
    
    results = [None] * ensemble_size

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                HONE_worker_with_damped_kuramoto,
                adj_matrix, dim, iterations, tol, seed, dt, gamma, gamma_theta, K
            )
            for seed in range(ensemble_size)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    # Extract histories
    ensemble_positions = [result[0] for result in results]
    ensemble_phases = [result[1] for result in results]

    # Compute variability metrics
    laplacian_variability = compute_laplacian_variability(G, ensemble_positions, ensemble_phases)
    node_variability = compute_node_variability(ensemble_positions, ensemble_phases)
    edge_variability = compute_edge_variability(G, ensemble_positions, ensemble_phases)

    return {
        "laplacian_variability": laplacian_variability,
        "node_variability": node_variability,
        "edge_variability": edge_variability
    }
