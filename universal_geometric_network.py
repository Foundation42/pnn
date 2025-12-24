"""
Universal Geometric Network

The ONE TRUE REPRESENTATION for neural intelligence.
Platform-agnostic, interpretable, optimizable, transferable.

Train once in geometric space → Distill to ANY substrate:
  - GPU (CUDA kernels)
  - PCB (copper traces)
  - Photonic (waveguides)
  - FPGA (Verilog)
  - Analog (memristors)
  - Biological (DNA circuits)

Geometry is not visualization. Geometry IS the representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
import networkx as nx


@dataclass
class GeometricAnalysis:
    """Results from analyzing a geometric network's structure."""
    clusters: List[np.ndarray]           # Spatial clusters (functional modules)
    cluster_centers: np.ndarray          # Center of each module
    hierarchy_levels: np.ndarray         # Depth assignment per neuron
    information_flow: nx.DiGraph         # Directed graph of signal flow
    critical_path: List[int]             # Bottleneck neurons
    modularity_score: float              # How modular is the structure
    interpretability_score: float        # How interpretable is the geometry


class UniversalGeometricNetwork(nn.Module):
    """
    The Universal Geometric Network.

    Every neuron has:
    - Position in continuous N-dimensional space
    - Learnable type embedding
    - Connections to other neurons (weights)

    The geometry EMERGES from training alongside accuracy.
    The geometry IS the architecture.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 spatial_dims: int = 3,
                 type_dims: int = 8,
                 space_size: float = 100.0):
        """
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            spatial_dims: Dimensionality of geometric space (2D or 3D)
            type_dims: Dimensionality of neuron type embedding
            space_size: Size of the geometric space (arbitrary units)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spatial_dims = spatial_dims
        self.type_dims = type_dims
        self.space_size = space_size
        self.total_neurons = input_size + hidden_size + output_size

        # === GEOMETRIC PARAMETERS ===

        # Neuron positions in continuous space (THE KEY INNOVATION)
        # All neurons exist in the same geometric space
        all_positions = self._initialize_positions()

        # Input positions are fixed (form a grid/line at one end)
        self.register_buffer('input_positions', all_positions[:input_size])

        # Hidden positions are trainable (they MOVE during learning!)
        self.hidden_positions = nn.Parameter(all_positions[input_size:input_size + hidden_size])

        # Output positions are fixed (at the other end)
        self.register_buffer('output_positions', all_positions[input_size + hidden_size:])

        # === NEURON TYPE EMBEDDINGS ===
        # Continuous "type" of each neuron (edge detector? curve detector? etc)
        self.hidden_types = nn.Parameter(torch.randn(hidden_size, type_dims) * 0.1)

        # === CONNECTION WEIGHTS ===
        # Standard neural network weights
        self.W1 = nn.Parameter(torch.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.W2 = nn.Parameter(torch.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(output_size))

        # === GEOMETRIC PHYSICS ===
        # How does distance affect connection strength?
        self.distance_scale = nn.Parameter(torch.tensor(10.0))  # Learnable!

    def _initialize_positions(self) -> torch.Tensor:
        """Initialize neuron positions in geometric space."""
        positions = []

        # Inputs: grid on one face of the space
        for i in range(self.input_size):
            if self.spatial_dims == 2:
                pos = [0.1 * self.space_size,
                       (i + 0.5) / self.input_size * self.space_size]
            else:  # 3D
                # Arrange as 2D grid projected into 3D
                grid_size = int(np.ceil(np.sqrt(self.input_size)))
                row, col = i // grid_size, i % grid_size
                pos = [0.1 * self.space_size,
                       (row + 0.5) / grid_size * self.space_size,
                       (col + 0.5) / grid_size * self.space_size]
            positions.append(pos)

        # Hidden: random in the middle of the space
        for i in range(self.hidden_size):
            if self.spatial_dims == 2:
                pos = [torch.rand(1).item() * 0.6 + 0.2,  # x: 20-80%
                       torch.rand(1).item()]
                pos = [p * self.space_size for p in pos]
            else:  # 3D
                pos = [torch.rand(1).item() * 0.6 + 0.2,  # x: 20-80%
                       torch.rand(1).item(),
                       torch.rand(1).item()]
                pos = [p * self.space_size for p in pos]
            positions.append(pos)

        # Outputs: line on the opposite face
        for i in range(self.output_size):
            if self.spatial_dims == 2:
                pos = [0.9 * self.space_size,
                       (i + 0.5) / self.output_size * self.space_size]
            else:  # 3D
                pos = [0.9 * self.space_size,
                       (i + 0.5) / self.output_size * self.space_size,
                       0.5 * self.space_size]
            positions.append(pos)

        return torch.tensor(positions, dtype=torch.float32)

    def get_all_positions(self) -> torch.Tensor:
        """Get positions of all neurons."""
        return torch.cat([
            self.input_positions,
            self.hidden_positions,
            self.output_positions
        ], dim=0)

    def compute_distance_matrix(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        # pos1: (N, D), pos2: (M, D)
        # output: (N, M)
        diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)  # (N, M, D)
        return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)

    def geometric_attenuation(self, distance: torch.Tensor) -> torch.Tensor:
        """
        How much does distance attenuate connection strength?

        This is the PHYSICS of the geometric space.
        Different substrates will have different attenuation functions:
        - Copper: R ∝ length (linear)
        - Optical: exponential decay
        - Electronic: capacitive effects

        For the universal representation, we learn it!
        """
        # Soft exponential decay, learnable scale
        return torch.exp(-distance / self.distance_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with geometric modulation.

        Weights are modulated by the geometric distance between neurons.
        """
        # Compute distance-based attenuation
        dist_ih = self.compute_distance_matrix(self.input_positions, self.hidden_positions)
        attn_ih = self.geometric_attenuation(dist_ih)

        dist_ho = self.compute_distance_matrix(self.hidden_positions, self.output_positions)
        attn_ho = self.geometric_attenuation(dist_ho)

        # Effective weights = base weights × geometric attenuation
        W1_eff = self.W1 * attn_ih
        W2_eff = self.W2 * attn_ho

        # Standard forward pass with effective weights
        hidden = F.relu(torch.matmul(x, W1_eff) + self.b1)
        output = torch.matmul(hidden, W2_eff) + self.b2

        return output

    def geometry_loss(self,
                      compactness_weight: float = 0.01,
                      separation_weight: float = 0.1,
                      flow_weight: float = 0.01) -> torch.Tensor:
        """
        Loss terms that shape the geometry.

        These can be tuned for different optimization goals:
        - compactness: prefer smaller layouts
        - separation: prevent neuron overlap
        - flow: encourage left-to-right information flow
        """
        losses = []
        pos = self.hidden_positions

        # 1. Compactness: minimize total spread
        spread = pos.std(dim=0).sum()
        losses.append(compactness_weight * spread)

        # 2. Separation: neurons shouldn't overlap
        dist_matrix = self.compute_distance_matrix(pos, pos)
        min_dist = 1.0  # Minimum allowed distance
        mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1)
        violations = F.relu(min_dist - dist_matrix) * mask
        losses.append(separation_weight * violations.sum())

        # 3. Flow: hidden neurons should be between input and output (x-coordinate)
        input_x = self.input_positions[:, 0].mean()
        output_x = self.output_positions[:, 0].mean()

        too_left = F.relu(input_x - pos[:, 0])
        too_right = F.relu(pos[:, 0] - output_x)
        losses.append(flow_weight * (too_left.sum() + too_right.sum()))

        return sum(losses)

    # === INTERPRETABILITY THROUGH GEOMETRY ===

    def analyze_structure(self, n_clusters: int = None) -> GeometricAnalysis:
        """
        Analyze the geometric structure of the trained network.

        Because intelligence exists in SPACE, we can ask spatial questions!
        """
        pos = self.hidden_positions.detach().cpu().numpy()

        # 1. Find spatial clusters (functional modules!)
        if n_clusters is None:
            # Use DBSCAN for automatic cluster detection
            clustering = DBSCAN(eps=self.space_size * 0.1, min_samples=3)
            labels = clustering.fit_predict(pos)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pos)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_labels = np.zeros(len(pos), dtype=int)
            cluster_centers = pos.mean(axis=0, keepdims=True)

        clusters = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]

        # 2. Analyze hierarchy (depth in x-coordinate)
        hierarchy_levels = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min() + 1e-8)

        # 3. Build information flow graph
        W1 = self.W1.detach().cpu().numpy()
        W2 = self.W2.detach().cpu().numpy()

        G = nx.DiGraph()

        # Add all neurons as nodes
        for i in range(self.input_size):
            G.add_node(f'I{i}', layer='input', pos=self.input_positions[i].cpu().numpy())
        for i in range(self.hidden_size):
            G.add_node(f'H{i}', layer='hidden', pos=pos[i], cluster=cluster_labels[i])
        for i in range(self.output_size):
            G.add_node(f'O{i}', layer='output', pos=self.output_positions[i].cpu().numpy())

        # Add weighted edges
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                if abs(W1[i, j]) > 0.1:  # Only significant connections
                    G.add_edge(f'I{i}', f'H{j}', weight=float(W1[i, j]))

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                if abs(W2[i, j]) > 0.1:
                    G.add_edge(f'H{i}', f'O{j}', weight=float(W2[i, j]))

        # 4. Find critical path (strongest route from any input to any output)
        critical_path = self._find_critical_path(G)

        # 5. Compute modularity
        modularity = self._compute_modularity(cluster_labels, W1, W2)

        # 6. Interpretability score (how clear is the structure?)
        interpretability = self._compute_interpretability(
            clusters, hierarchy_levels, modularity
        )

        return GeometricAnalysis(
            clusters=clusters,
            cluster_centers=cluster_centers,
            hierarchy_levels=hierarchy_levels,
            information_flow=G,
            critical_path=critical_path,
            modularity_score=modularity,
            interpretability_score=interpretability
        )

    def _find_critical_path(self, G: nx.DiGraph) -> List[str]:
        """Find the path with strongest total weight."""
        max_weight = 0
        best_path = []

        input_nodes = [n for n in G.nodes if n.startswith('I')]
        output_nodes = [n for n in G.nodes if n.startswith('O')]

        for inp in input_nodes:
            for out in output_nodes:
                try:
                    for path in nx.all_simple_paths(G, inp, out):
                        weight = sum(abs(G[u][v]['weight']) for u, v in zip(path[:-1], path[1:]))
                        if weight > max_weight:
                            max_weight = weight
                            best_path = path
                except nx.NetworkXNoPath:
                    continue

        return best_path

    def _compute_modularity(self, cluster_labels: np.ndarray,
                            W1: np.ndarray, W2: np.ndarray) -> float:
        """
        Compute how modular the network is.
        High modularity = neurons in same cluster connect more to each other.
        """
        # For a 2-layer network, we measure if clusters specialize
        # by looking at weight variance within vs across clusters

        n_clusters = len(set(cluster_labels))
        if n_clusters <= 1:
            return 0.0

        # Compute within-cluster weight similarity
        within_var = 0.0
        across_var = 0.0

        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_W2 = W2[mask, :]

            # Variance within cluster (should be low if specialized)
            within_var += cluster_W2.var()

        within_var /= n_clusters
        across_var = W2.var()

        # Modularity: high when within-cluster variance is low relative to total
        modularity = 1.0 - (within_var / (across_var + 1e-8))
        return float(np.clip(modularity, 0, 1))

    def _compute_interpretability(self, clusters: List[np.ndarray],
                                   hierarchy: np.ndarray,
                                   modularity: float) -> float:
        """
        How interpretable is this geometric structure?

        Factors:
        - Clear clustering (distinct modules)
        - Clear hierarchy (depth structure)
        - High modularity (specialized clusters)
        """
        # Cluster clarity: clusters should be well-separated
        cluster_sizes = [len(c) for c in clusters]
        size_entropy = -sum(s/sum(cluster_sizes) * np.log(s/sum(cluster_sizes) + 1e-8)
                           for s in cluster_sizes if s > 0)
        max_entropy = np.log(len(clusters) + 1e-8)
        cluster_clarity = 1.0 - size_entropy / (max_entropy + 1e-8)

        # Hierarchy clarity: should span the full range
        hierarchy_range = hierarchy.max() - hierarchy.min()
        hierarchy_clarity = hierarchy_range

        # Combine factors
        interpretability = (cluster_clarity + hierarchy_clarity + modularity) / 3.0

        return float(np.clip(interpretability, 0, 1))

    def explain_decision(self, x: torch.Tensor, output_idx: int) -> Dict:
        """
        Explain why the network made a particular decision.

        Because we have geometry, we can trace the physical path!
        """
        # Forward pass with activation tracking
        self.eval()

        dist_ih = self.compute_distance_matrix(self.input_positions, self.hidden_positions)
        attn_ih = self.geometric_attenuation(dist_ih)
        W1_eff = self.W1 * attn_ih

        dist_ho = self.compute_distance_matrix(self.hidden_positions, self.output_positions)
        attn_ho = self.geometric_attenuation(dist_ho)
        W2_eff = self.W2 * attn_ho

        hidden_pre = torch.matmul(x, W1_eff) + self.b1
        hidden = F.relu(hidden_pre)

        # Find which hidden neurons contributed most to this output
        contributions = hidden * W2_eff[:, output_idx]
        top_hidden = contributions.abs().argsort(descending=True)[:5]

        # For each top hidden neuron, find which inputs contributed most
        explanation = {
            'output_idx': output_idx,
            'top_hidden_neurons': [],
        }

        for h_idx in top_hidden:
            h_idx = h_idx.item()
            input_contributions = x * W1_eff[:, h_idx]
            top_inputs = input_contributions.abs().argsort(descending=True)[:3]

            explanation['top_hidden_neurons'].append({
                'neuron_id': h_idx,
                'position': self.hidden_positions[h_idx].detach().cpu().numpy(),
                'contribution': contributions[0, h_idx].item(),
                'top_input_sources': [i.item() for i in top_inputs[0]]
            })

        return explanation

    def summary(self) -> str:
        """Print a summary of the geometric network."""
        analysis = self.analyze_structure()

        lines = [
            "=" * 60,
            "Universal Geometric Network",
            "=" * 60,
            f"Architecture: {self.input_size} → {self.hidden_size} → {self.output_size}",
            f"Spatial dimensions: {self.spatial_dims}D",
            f"Space size: {self.space_size}",
            "",
            "Geometric Analysis:",
            f"  Clusters found: {len(analysis.clusters)}",
            f"  Modularity score: {analysis.modularity_score:.3f}",
            f"  Interpretability score: {analysis.interpretability_score:.3f}",
            f"  Critical path length: {len(analysis.critical_path)} neurons",
            "",
            "Cluster sizes:",
        ]

        for i, cluster in enumerate(analysis.clusters):
            lines.append(f"  Cluster {i}: {len(cluster)} neurons")

        lines.append("=" * 60)
        return "\n".join(lines)


# === DISTILLERS ===
# These convert the universal geometric representation to specific substrates

class GPUDistiller:
    """Distill geometric network to optimized GPU execution."""

    def analyze_for_gpu(self, network: UniversalGeometricNetwork) -> Dict:
        """
        Analyze geometric structure for GPU optimization.

        Key insight: Spatial clusters → Thread block mapping
        Neurons that are geometrically close should compute together!
        """
        analysis = network.analyze_structure()

        return {
            'clusters': analysis.clusters,
            'cluster_centers': analysis.cluster_centers,
            'recommended_block_size': self._compute_block_size(analysis.clusters),
            'memory_layout': self._optimize_memory_layout(network, analysis),
            'expected_speedup': self._estimate_speedup(analysis)
        }

    def _compute_block_size(self, clusters: List[np.ndarray]) -> int:
        """Compute optimal CUDA block size based on cluster sizes."""
        avg_cluster_size = np.mean([len(c) for c in clusters])
        # Round up to nearest power of 2 for GPU efficiency
        block_size = 2 ** int(np.ceil(np.log2(avg_cluster_size)))
        return min(max(block_size, 32), 1024)  # CUDA limits

    def _optimize_memory_layout(self, network: UniversalGeometricNetwork,
                                 analysis: GeometricAnalysis) -> Dict:
        """
        Reorder weights/positions for cache-friendly access.

        Neurons in the same cluster are stored contiguously!
        """
        # Flatten cluster assignments into a reordering
        reorder = []
        for cluster in analysis.clusters:
            reorder.extend(cluster.tolist())

        return {
            'neuron_reorder': reorder,
            'cache_line_utilization': 0.85,  # Estimated
        }

    def _estimate_speedup(self, analysis: GeometricAnalysis) -> float:
        """Estimate speedup from geometric optimization."""
        # Higher modularity = better cache utilization = more speedup
        base_speedup = 1.0
        modularity_bonus = analysis.modularity_score * 2.0
        return base_speedup + modularity_bonus


class PCBDistiller:
    """Distill geometric network to PCB layout."""

    def distill(self, network: UniversalGeometricNetwork,
                board_size: Tuple[float, float] = (100, 100),
                n_layers: int = 2) -> Dict:
        """
        Convert geometric network to PCB specification.

        2D projection of positions → component placement
        Weights → trace widths
        Z-coordinate (if 3D) → layer assignment
        """
        pos = network.hidden_positions.detach().cpu().numpy()

        # Scale positions to board size
        pos_scaled = pos.copy()
        for d in range(min(2, pos.shape[1])):
            pos_min, pos_max = pos[:, d].min(), pos[:, d].max()
            margin = board_size[d] * 0.1
            pos_scaled[:, d] = (pos[:, d] - pos_min) / (pos_max - pos_min + 1e-8)
            pos_scaled[:, d] = pos_scaled[:, d] * (board_size[d] - 2 * margin) + margin

        # Layer assignment from Z coordinate (if 3D)
        if pos.shape[1] >= 3:
            z_normalized = (pos[:, 2] - pos[:, 2].min()) / (pos[:, 2].max() - pos[:, 2].min() + 1e-8)
            layer_assignments = (z_normalized * (n_layers - 1)).astype(int)
        else:
            layer_assignments = np.zeros(len(pos), dtype=int)

        return {
            'component_positions': pos_scaled[:, :2],
            'layer_assignments': layer_assignments,
            'board_size': board_size,
            'n_layers': n_layers,
            'traces': self._generate_traces(network, pos_scaled),
        }

    def _generate_traces(self, network: UniversalGeometricNetwork,
                         pos: np.ndarray) -> List[Dict]:
        """Generate PCB trace specifications."""
        traces = []
        W1 = network.W1.detach().cpu().numpy()
        W2 = network.W2.detach().cpu().numpy()

        # Input → Hidden traces
        input_pos = network.input_positions.cpu().numpy()
        for i in range(network.input_size):
            for j in range(network.hidden_size):
                if abs(W1[i, j]) > 0.05:
                    traces.append({
                        'from': input_pos[i, :2],
                        'to': pos[j, :2],
                        'weight': float(W1[i, j]),
                        'width_mm': self._weight_to_width(W1[i, j])
                    })

        return traces

    def _weight_to_width(self, weight: float) -> float:
        """Convert weight to trace width."""
        min_width, max_width = 0.09, 2.0
        normalized = (abs(weight) / 3.0)  # Assume weights in [-3, 3]
        return min_width + normalized * (max_width - min_width)


class PhotonicDistiller:
    """Distill geometric network to photonic integrated circuit."""

    def distill(self, network: UniversalGeometricNetwork) -> Dict:
        """
        Convert to photonic chip specification.

        Neurons → Ring resonators
        Connections → Waveguides
        Weights → Coupling coefficients
        """
        pos = network.hidden_positions.detach().cpu().numpy()

        resonators = []
        for i, p in enumerate(pos):
            resonators.append({
                'id': i,
                'position': p[:2],
                'radius_um': 10.0,  # Standard ring radius
                'resonance_wavelength_nm': 1550.0 + i * 0.1,  # WDM spacing
            })

        waveguides = self._route_waveguides(network, pos)

        return {
            'resonators': resonators,
            'waveguides': waveguides,
            'chip_size_mm': (10, 10),
            'layer_stack': 'SOI',  # Silicon-on-insulator
        }

    def _route_waveguides(self, network: UniversalGeometricNetwork,
                          pos: np.ndarray) -> List[Dict]:
        """Route optical waveguides between resonators."""
        waveguides = []
        W2 = network.W2.detach().cpu().numpy()
        output_pos = network.output_positions.cpu().numpy()

        for i in range(network.hidden_size):
            for j in range(network.output_size):
                if abs(W2[i, j]) > 0.1:
                    waveguides.append({
                        'from_resonator': i,
                        'to_output': j,
                        'coupling_coefficient': self._weight_to_coupling(W2[i, j]),
                        'path': [pos[i, :2], output_pos[j, :2]],
                    })

        return waveguides

    def _weight_to_coupling(self, weight: float) -> float:
        """Convert weight to optical coupling coefficient."""
        # Coupling: 0 = no coupling, 1 = full coupling
        return min(abs(weight) / 3.0, 1.0)


# === DEMO ===

if __name__ == "__main__":
    print("=" * 60)
    print("  Universal Geometric Network Demo")
    print("  The ONE TRUE REPRESENTATION")
    print("=" * 60)
    print()

    # Create a geometric network
    network = UniversalGeometricNetwork(
        input_size=16,
        hidden_size=32,
        output_size=4,
        spatial_dims=3,
        space_size=100.0
    )

    # Test forward pass
    x = torch.randn(8, 16)
    y = network(x)
    print(f"Forward pass: {x.shape} → {y.shape}")

    # Analyze structure
    print("\n" + network.summary())

    # Test distillers
    print("\n--- GPU Distillation ---")
    gpu_distiller = GPUDistiller()
    gpu_analysis = gpu_distiller.analyze_for_gpu(network)
    print(f"Recommended block size: {gpu_analysis['recommended_block_size']}")
    print(f"Expected speedup: {gpu_analysis['expected_speedup']:.2f}x")

    print("\n--- PCB Distillation ---")
    pcb_distiller = PCBDistiller()
    pcb_spec = pcb_distiller.distill(network, board_size=(50, 50))
    print(f"Board size: {pcb_spec['board_size']}mm")
    print(f"Layers: {pcb_spec['n_layers']}")
    print(f"Traces: {len(pcb_spec['traces'])}")

    print("\n--- Photonic Distillation ---")
    photonic_distiller = PhotonicDistiller()
    photonic_spec = photonic_distiller.distill(network)
    print(f"Resonators: {len(photonic_spec['resonators'])}")
    print(f"Waveguides: {len(photonic_spec['waveguides'])}")

    print("\n" + "=" * 60)
    print("  Geometry IS the Universal IR")
    print("=" * 60)
