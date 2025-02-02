# 📌 Harmonic Oscillator Network Dynamics with Kuramoto model

This project implements **coupled harmonic oscillators with Kuramoto-type phase dynamics** embedded in a **weighted complex network**. The model incorporates **position-dependent forces (HONE)** and **phase synchronization effects (Kuramoto model with damping)**.

- **Supports multi-threaded CPU execution** 🚀  
- **Optimized for GPU acceleration with CuPy** ⚡  
- **Tracks energy evolution, phase synchronization, and network forces**  


## **0. Repository Structure**
```
📂 HONE-Kuramoto
│── 📜 README.md                   # Project documentation
│── 📜 HOND + Kuramoto Ensemble CPU.py  # Multi-threaded CPU implementation
│── 📜 HOND + Kuramoto Ensemble GPU.py  # GPU-accelerated implementation with CuPy
```
---

## **1. Weighted Damped Kuramoto Model (Phase Update)**
Each node has an **intrinsic frequency** \( \omega_i \) and interacts with adjacent nodes through **phase differences**, weighted by \( w_{ij} \). A damping term \( \gamma_\theta \) is included.

\[
\frac{d\theta_i}{dt} = \omega_i + K \sum_{j} w_{ij} \sin(\theta_j - \theta_i) - \gamma_\theta \dot{\theta}_i
\]

### **Discrete-Time Update**
\[
\dot{\theta}_i \leftarrow \dot{\theta}_i + \left( \omega_i + K \sum_{j} w_{ij} \sin(\theta_j - \theta_i) - \gamma_\theta \dot{\theta}_i \right) \cdot dt
\]

\[
\theta_i \leftarrow \theta_i + \dot{\theta}_i \cdot dt
\]

### **Parameters**
- \( \theta_i \): Phase of node \( i \)
- \( \dot{\theta}_i \): Phase velocity of node \( i \)
- \( \omega_i \): Intrinsic frequency (randomly initialized)
- \( K \): Phase coupling strength
- \( w_{ij} \): Network weight (strength of connection between nodes \( i \) and \( j \))
- \( \gamma_\theta \): Phase damping coefficient

---

## **2. Position & Energy Computation**
### **(1) Position Update (HONE Dynamics)**
Each node is influenced by **harmonic forces** from neighboring nodes.

\[
\mathbf{r}_i^{(t+1)} = \mathbf{r}_i^{(t)} + \Delta t \cdot \mathbf{v}_i^{(t)}
\]

\[
\mathbf{v}_i^{(t+1)} = \mathbf{v}_i^{(t)} + \Delta t \cdot \left( \mathbf{F}_i - \gamma \mathbf{v}_i^{(t)} \right)
\]

where the force \( \mathbf{F}_i \) is computed as:

\[
\mathbf{F}_i = \sum_{j} w_{ij} \frac{\mathbf{r}_j - \mathbf{r}_i}{||\mathbf{r}_j - \mathbf{r}_i||}
\]

### **(2) Energy Computation**
#### **(a) Potential Energy**
\[
U = \frac{1}{2} \sum_{i < j} w_{ij} ||\mathbf{r}_i - \mathbf{r}_j||^2
\]

#### **(b) Kinetic Energy**
\[
K = \frac{1}{2} \sum_{i} ||\mathbf{v}_i||^2
\]

#### **(c) Total Energy**
\[
E_{\text{total}} = K + U
\]

---

## **3. Algorithm Implementation**
### **(1) Single Simulation (HONE with Damped Kuramoto)**
Each simulation follows:
1. **Initialize node positions, velocities, phases, and intrinsic frequencies**
2. **For each time step**:
   - Compute **harmonic forces** \( \mathbf{F}_i \)
   - Update **positions and velocities** using HONE dynamics
   - Compute **phase evolution** using the Kuramoto model with damping
   - Compute **potential, kinetic, and total energy**
   - Check for **convergence** (if movement < \( \text{tol} \), stop early)

### **(2) Multi-threaded Ensemble Simulation**
- **CPU:** Uses `ThreadPoolExecutor` to run multiple simulations in parallel  
- **GPU:** Uses **CuPy** (`cp`) for GPU acceleration  
- **Each ensemble run stores:** Position, phase, and energy evolution  

---

## **4. Parameters**
| Parameter       | Description                                      |
|----------------|--------------------------------------------------|
| \( \gamma \)   | Damping coefficient for position update         |
| \( \gamma_\theta \) | Damping coefficient for phase synchronization |
| \( K \)        | Coupling strength of the Kuramoto model        |
| \( dt \)       | Time step for integration                      |
| \( w_{ij} \)   | Weight matrix (adjacency matrix)               |
| \( \omega_i \) | Intrinsic frequency of node \( i \)            |
| \( \mathbf{r}_i \) | Position of node \( i \)                   |
| \( \mathbf{v}_i \) | Velocity of node \( i \)                   |
| \( \theta_i \) | Phase of node \( i \)                          |
| \( \dot{\theta}_i \) | Phase velocity of node \( i \)            |

---
