import cupy as cp
import networkx as nx
from scipy.linalg import eigh
from concurrent.futures import ThreadPoolExecutor  # Multi-threading for parallel execution

def compute_velocity_variability_gpu(ensemble_positions):
    """
    Compute velocity-based synchronization variability using GPU acceleration (CuPy).

    Parameters:
        ensemble_positions (list of lists): List of node position histories for each ensemble realization.

    Returns:
        dict: {"velocity_variance": Mean variance of velocity magnitudes across nodes.}
    """
    def compute_velocity(positions):
        return cp.linalg.norm(cp.asarray(positions[-1]) - cp.asarray(positions[-2]), axis=1)

    with ThreadPoolExecutor() as executor:
        velocities = list(executor.map(compute_velocity, ensemble_positions))

    velocities = cp.array(velocities)  # Shape: (ensemble_size, num_nodes)

    velocity_variance = cp.mean(cp.var(velocities, axis=0))

    return {
        "velocity_variance": float(velocity_variance.get())
    }

def compute_phase_variability_gpu(ensemble_phases):
    """
    Compute phase-based synchronization variability using GPU acceleration.

    Parameters:
        ensemble_phases (list of lists): List of node phase histories for each ensemble realization.

    Returns:
        dict: {"phase_variance": Mean variance of phase values across nodes.}
    """
    phases = cp.array(ensemble_phases)  # Shape: (ensemble_size, num_nodes)

    phase_variance = cp.mean(cp.var(phases, axis=0))

    return {
        "phase_variance": float(phase_variance.get())
    }

def compute_laplacian_variability_gpu(G, ensemble_positions, ensemble_phases):
    """
    Compute Laplacian-based synchronization variability using GPU acceleration.

    Parameters:
        G (networkx.Graph): Network structure.
        ensemble_positions (list of lists): List of position histories.
        ensemble_phases (list of lists): List of phase histories.

    Returns:
        dict: 
            - "lambda_2_variance": Variance of the second smallest Laplacian eigenvalue.
            - "v2_variance": Variance of the second eigenvector components.
    """
    laplacian_matrix = cp.asarray(nx.laplacian_matrix(G).toarray())

    eigenvalues_list = []
    eigenvectors_list = []

    for _ in ensemble_positions:
        eigvals, eigvecs = cp.linalg.eigh(laplacian_matrix)
        eigenvalues_list.append(eigvals)
        eigenvectors_list.append(eigvecs)

    eigenvalues_list = cp.array(eigenvalues_list)
    eigenvectors_list = cp.array(eigenvectors_list)

    lambda_2_variance = cp.mean(cp.var(eigenvalues_list[:, 1], axis=0))
    v2_variance = cp.mean(cp.var(eigenvectors_list[:, :, 1], axis=0))

    return {
        "lambda_2_variance": float(lambda_2_variance.get()),
        "v2_variance": float(v2_variance.get())
    }

def HONE_worker_with_damped_kuramoto_gpu(adj_matrix, dim, iterations, tol, seed, dt, gamma, gamma_theta, K):
    """
    GPU-accelerated simulation of a network of harmonic oscillators with damped Kuramoto synchronization.

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
    cp.random.seed(seed)
    num_nodes = adj_matrix.shape[0]
    adj_matrix = cp.asarray(adj_matrix)  # Convert adjacency matrix to GPU

    # Initialize positions, velocities, phases, and intrinsic frequencies
    positions = cp.random.rand(num_nodes, dim)
    velocities = cp.zeros_like(positions)
    phases = cp.random.uniform(0, 2 * cp.pi, num_nodes)
    phase_velocities = cp.zeros(num_nodes)
    intrinsic_frequencies = cp.random.normal(0, 1, num_nodes)

    positions_history, phase_history = [positions.copy()], [phases.copy()]
    potential_energy_history, kinetic_energy_history, total_energy_history = [], [], []

    def calculate_forces(positions):
        """ Compute interaction forces using GPU acceleration """
        forces = cp.zeros_like(positions)
        for i in range(num_nodes):
            delta = positions - positions[i]
            distances = cp.linalg.norm(delta, axis=1)
            mask = distances > 1e-6
            distances[~mask] = 1e-6  
            forces[i] = cp.sum(adj_matrix[i, mask][:, None] * (delta[mask] / distances[mask, None]), axis=0)
        return forces

    # Simulation loop
    for step in range(iterations):
        forces = calculate_forces(positions)
        velocities += forces * dt - gamma * velocities
        new_positions = positions + velocities * dt

        # Apply Periodic Boundary Conditions (PBC) to phase updates
        phase_diffs = cp.array([
            cp.sum(adj_matrix[i] * cp.sin((phases - phases[i] + cp.pi) % (2 * cp.pi) - cp.pi)) 
            for i in range(num_nodes)
        ])

        phase_velocities += (intrinsic_frequencies + K * phase_diffs - gamma_theta * phase_velocities) * dt
        new_phases = cp.mod(phases + phase_velocities * dt, 2*cp.pi)

        # Update positions and phases
        positions, phases = new_positions, new_phases
        positions_history.append(positions.copy())
        phase_history.append(phases.copy())

    return positions_history, phase_history, potential_energy_history, kinetic_energy_history, total_energy_history

def HONE_kuramoto_ensemble_gpu(G, dim=2, iterations=100, ensemble_size=10, tol=1e-4, dt=0.01, gamma=1.0, gamma_theta=0.1, K=0.5):
    """
    Run an ensemble of GPU-accelerated HONE-Kuramoto simulations.

    Parameters:
        G (networkx.Graph): Network graph.
        dim (int): Embedding dimension.
        iterations (int): Number of simulation steps.
        ensemble_size (int): Number of ensemble realizations (seeds).
        tol (float): Convergence threshold.
        dt (float): Time step.
        gamma (float): Spatial damping coefficient.
        gamma_theta (float): Phase synchronization damping coefficient.
        K (float): Kuramoto coupling strength.

    Returns:
        tuple:
            - ensemble_positions, ensemble_phases, ensemble_potential_energies, 
              ensemble_kinetic_energies, ensemble_total_energies
    """
    adj_matrix = nx.to_numpy_array(G, weight="weight")
    results = [None] * ensemble_size

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                HONE_worker_with_damped_kuramoto_gpu,
                adj_matrix, dim, iterations, tol, seed, dt, gamma, gamma_theta, K
            )
            for seed in range(ensemble_size)
        ]
        for i, future in enumerate(futures):
            results[i] = future.result()

    ensemble_positions = [result[0] for result in results]
    ensemble_phases = [result[1] for result in results]
    return ensemble_positions, ensemble_phases


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
