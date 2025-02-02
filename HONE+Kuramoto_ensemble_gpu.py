import cupy as cp
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def HONE_worker_with_damped_kuramoto_gpu(adj_matrix, dim, iterations, tol, seed, dt, gamma, gamma_theta, K):
    """
    GPU-accelerated Harmonic Oscillator Network Embedding (HONE) with Damped Kuramoto Model.
    This function runs a single ensemble realization using CuPy for GPU computation.

    Parameters:
        adj_matrix (cp.ndarray): Adjacency matrix (stored in CuPy for GPU acceleration).
        dim (int): Dimensionality of the embedding space.
        iterations (int): Number of simulation steps.
        tol (float): Convergence threshold.
        seed (int): Random seed.
        dt (float): Time step.
        gamma (float): Damping coefficient for spatial movement.
        gamma_theta (float): Damping coefficient for phase synchronization.
        K (float): Coupling strength.

    Returns:
        tuple: (positions_history, phase_history, potential_energy_history, kinetic_energy_history, total_energy_history)
    """
    cp.random.seed(seed)  # Set GPU random seed
    num_nodes = adj_matrix.shape[0]  # Number of nodes

    # Initialize GPU arrays for positions and velocities
    positions = cp.random.rand(num_nodes, dim)
    velocities = cp.zeros_like(positions)

    # Initialize phases and phase velocities
    phases = cp.random.uniform(0, 2 * cp.pi, num_nodes)
    phase_velocities = cp.zeros(num_nodes)

    # Generate intrinsic frequencies from a normal distribution (mean = 0, variance = 1)
    intrinsic_frequencies = cp.random.normal(0, 1, num_nodes)

    # Lists to store simulation history (stored on CPU for better memory handling)
    positions_history = []
    phase_history = []
    potential_energy_history = []
    kinetic_energy_history = []
    total_energy_history = []

    def calculate_forces(positions):
        """ Compute harmonic oscillator forces using GPU """
        forces = cp.zeros_like(positions)
        for i in range(num_nodes):
            delta = positions - positions[i]
            distances = cp.linalg.norm(delta, axis=1)
            mask = distances > 1e-6
            distances[~mask] = cp.max(cp.array([1e-6, cp.min(distances[mask])]))  # Regularize small distances
            forces[i] = cp.sum(adj_matrix[i, mask][:, None] * (delta[mask] / distances[mask, None]), axis=0)
        return forces

    def compute_potential_energy(positions):
        """ Compute total potential energy using GPU """
        return 0.5 * cp.sum([
            adj_matrix[i, j] * cp.max(cp.linalg.norm(positions[i] - positions[j]), 1e-6)**2
            for i in range(num_nodes) for j in range(i)
        ])

    # Simulation loop
    for step in range(iterations):
        # Compute forces and update positions
        forces = calculate_forces(positions)
        velocities += forces * dt - gamma * velocities  # Apply damping
        new_positions = positions + velocities * dt

        # Compute phase dynamics using the Kuramoto model
        phase_diffs = cp.array([
            cp.sum(adj_matrix[i] * cp.sin(phases - phases[i])) for i in range(num_nodes)
        ])
        phase_accelerations = intrinsic_frequencies + K * phase_diffs - gamma_theta * phase_velocities
        phase_velocities += phase_accelerations * dt
        new_phases = phases + phase_velocities * dt

        # Compute kinetic and potential energy
        kinetic_energy = 0.5 * cp.sum(cp.linalg.norm(velocities, axis=1) ** 2)
        potential_energy = compute_potential_energy(new_positions)
        total_energy = kinetic_energy + potential_energy

        # Store history (transfer to CPU)
        positions_history.append(cp.asnumpy(new_positions.copy()))
        phase_history.append(cp.asnumpy(new_phases.copy()))
        kinetic_energy_history.append(float(kinetic_energy))
        potential_energy_history.append(float(potential_energy))
        total_energy_history.append(float(total_energy))

        # Check convergence
        total_movement = cp.sum(cp.linalg.norm(new_positions - positions, axis=1))
        if total_movement < tol:
            break

        positions = new_positions
        phases = new_phases

    return positions_history, phase_history, potential_energy_history, kinetic_energy_history, total_energy_history

def HONE_kuramoto_ensemble_gpu(G, dim=2, iterations=100, ensemble_size=100, tol=1e-4, dt=0.01, gamma=1.0, gamma_theta=0.1, K=0.5):
    """
    GPU-accelerated Harmonic Oscillator Network Embedding (HONE) for an ensemble of simulations.
    Runs multiple simulations in parallel using CuPy for GPU acceleration.

    Parameters:
        G (networkx.Graph): Input graph.
        dim (int): Number of dimensions.
        iterations (int): Number of iterations per simulation.
        ensemble_size (int): Number of ensemble realizations (seeds).
        tol (float): Convergence threshold.
        dt (float): Time step.
        gamma (float): Damping coefficient for positions.
        gamma_theta (float): Damping coefficient for phases.
        K (float): Coupling strength.

    Returns:
        tuple: (ensemble_positions, ensemble_phases, ensemble_potential_energies, ensemble_kinetic_energies, ensemble_total_energies)
    """
    # Convert graph to adjacency matrix and move to GPU
    if nx.is_weighted(G):
        adj_matrix = cp.asarray(nx.to_numpy_array(G, weight="weight"))
    else:
        adj_matrix = cp.asarray(nx.to_numpy_array(G))
        adj_matrix[adj_matrix > 0] = 1  # Convert to unweighted if needed

    results = [None] * ensemble_size

    # Use multi-threading for parallel execution (GPU threads)
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

    # Extract histories from results
    ensemble_positions = [result[0] for result in results]
    ensemble_phases = [result[1] for result in results]
    ensemble_potential_energies = [result[2] for result in results]
    ensemble_kinetic_energies = [result[3] for result in results]
    ensemble_total_energies = [result[4] for result in results]

    return ensemble_positions, ensemble_phases, ensemble_potential_energies, ensemble_kinetic_energies, ensemble_total_energies
