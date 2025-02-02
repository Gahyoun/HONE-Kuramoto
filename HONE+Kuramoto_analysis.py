import numpy as np
import networkx as nx
from scipy.linalg import eigh

def analyze_velocity_synchronization_with_laplacian(G, ensemble_positions, dt, resolution=1, weight="weight", seed=42):
    """
    Analyze velocity synchronization using the Laplacian matrix across ensemble realizations.

    This function evaluates synchronization stability by computing variance in velocity magnitudes,
    algebraic connectivity (λ₂), and synchronization patterns in detected communities.

    Parameters:
        G (networkx.Graph): Input network graph.
        ensemble_positions (list of lists): List of position histories for each ensemble realization.
        dt (float): Time step for numerical differentiation.
        resolution (float, optional): Resolution parameter for Louvain community detection. Default is 1.
        weight (str, optional): Edge weight attribute for community detection. Default is "weight".
        seed (int, optional): Random seed for Louvain partitioning. Default is 42.

    Returns:
        dict: Velocity synchronization analysis results.
    """
    num_nodes = G.number_of_nodes()
    
    # Compute velocities across ensemble realizations
    velocities = np.array([
        np.linalg.norm(positions[-1] - positions[-2], axis=1) / dt 
        for positions in ensemble_positions
    ])

    # Compute overall velocity variance
    velocity_variance = np.var(velocities, axis=0)

    # Compute Laplacian matrix and its eigenvalues/eigenvectors
    laplacian_matrix = nx.laplacian_matrix(G, weight=weight).toarray()
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # Compute algebraic connectivity (λ₂) and its eigenvector variance
    lambda_2 = eigenvalues[1]
    v2_variance = np.var(eigenvectors[:, 1])

    # Compute community-based synchronization
    partitions = list(nx.community.louvain_partitions(G, weight=weight, resolution=resolution, seed=seed))[-1]
    community_velocity_variances = {}

    for community_id, nodes in enumerate(partitions):
        indices = [list(G.nodes).index(node) for node in nodes]
        community_velocity_variances[community_id] = np.var(velocity_variance[indices])

    return {
        "Algebraic connectivity (lambda_2)": lambda_2,
        "Eigenvector variance (v_2)": v2_variance,
        "Overall velocity variance": np.mean(velocity_variance),
        "Community velocity variances": community_velocity_variances,
    }

def analyze_phase_synchronization_with_laplacian(G, ensemble_phases, resolution=1, weight="weight", seed=42):
    """
    Analyze phase synchronization using the Laplacian matrix across ensemble realizations.

    This function evaluates synchronization stability by computing phase variance,
    algebraic connectivity (λ₂), and synchronization patterns in detected communities.

    Parameters:
        G (networkx.Graph): Input network graph.
        ensemble_phases (list of lists): List of phase histories for each ensemble realization.
        resolution (float, optional): Resolution parameter for Louvain community detection. Default is 1.
        weight (str, optional): Edge weight attribute for community detection. Default is "weight".
        seed (int, optional): Random seed for Louvain partitioning. Default is 42.

    Returns:
        dict: Phase synchronization analysis results.
    """
    num_nodes = G.number_of_nodes()

    # Compute final phase variance across ensemble realizations
    final_phases = np.array([phases[-1] for phases in ensemble_phases])
    phase_variance = np.var(final_phases, axis=0)

    # Compute Laplacian matrix and its eigenvalues/eigenvectors
    laplacian_matrix = nx.laplacian_matrix(G, weight=weight).toarray()
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    # Compute algebraic connectivity (λ₂) and its eigenvector variance
    lambda_2 = eigenvalues[1]
    v2_variance = np.var(eigenvectors[:, 1])

    # Compute community-based synchronization
    partitions = list(nx.community.louvain_partitions(G, weight=weight, resolution=resolution, seed=seed))[-1]
    community_phase_variances = {}

    for community_id, nodes in enumerate(partitions):
        indices = [list(G.nodes).index(node) for node in nodes]
        community_phase_variances[community_id] = np.var(phase_variance[indices])

    return {
        "Algebraic connectivity (lambda_2)": lambda_2,
        "Eigenvector variance (v_2)": v2_variance,
        "Overall phase variance": np.mean(phase_variance),
        "Community phase variances": community_phase_variances,
    }
