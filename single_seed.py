import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class GeometricNeuralNetwork:
    def __init__(self, input_dim, output_dim, learning_rate=0.01, 
                 split_threshold=0.5, merge_threshold=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        
        # Start with single seed neuron at origin
        self.positions = np.zeros((1, input_dim))
        self.velocities = np.zeros((1, input_dim))
        self.weights = np.random.randn(1, output_dim) * 0.01
        self.gradients = np.zeros((1, input_dim))
        self.gradient_conflict = np.zeros(1)  # Track conflict per neuron
        
    def forward(self, X):
        # Calculate distances to all neurons
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - 
                                   self.positions[np.newaxis, :, :])**2, axis=2))
        
        # RBF activation with adaptive bandwidth
        bandwidth = np.median(distances) + 0.1
        activations = np.exp(-distances**2 / (2 * bandwidth**2))
        
        # Normalize activations
        activations = activations / (np.sum(activations, axis=1, keepdims=True) + 1e-10)
        
        # Weighted sum for output
        output = np.dot(activations, self.weights)
        
        return output, activations
    
    def compute_gradients(self, X, y, activations):
        """Compute geometric gradients for each neuron"""
        # Output error
        output, _ = self.forward(X)
        error = output - y

        # Weight gradients
        weight_grad = np.dot(activations.T, error) / len(X)

        # Position gradients (geometric)
        position_grad = np.zeros_like(self.positions)
        self.gradient_conflict = np.zeros(len(self.positions))  # NEW: track conflict

        for i in range(len(self.positions)):
            # Calculate how this neuron's position affects loss
            diff = X - self.positions[i]
            distances = np.sqrt(np.sum(diff**2, axis=1, keepdims=True)) + 1e-10

            # Per-sample gradients (don't average yet!)
            weighted_error = np.sum(error * activations[:, i:i+1], axis=1, keepdims=True)
            per_sample_grad = weighted_error * diff / distances

            # Mean gradient for movement
            position_grad[i] = np.mean(per_sample_grad, axis=0)

            # NEW: Measure gradient CONFLICT (variance in directions)
            # Normalize per-sample gradients to unit vectors
            norms = np.linalg.norm(per_sample_grad, axis=1, keepdims=True) + 1e-10
            directions = per_sample_grad / norms

            # Mean direction
            mean_dir = np.mean(directions, axis=0)
            mean_dir_norm = np.linalg.norm(mean_dir) + 1e-10

            # Conflict = 1 - alignment. If all point same way, alignment=1, conflict=0
            # If they cancel out (opposite directions), alignment=0, conflict=1
            self.gradient_conflict[i] = 1.0 - mean_dir_norm

        self.gradients = position_grad
        return weight_grad
    
    def should_split(self, neuron_idx):
        """Check if gradient conflict suggests need to split"""
        # Use CONFLICT (disagreement in gradient directions) not magnitude
        conflict = self.gradient_conflict[neuron_idx]
        return conflict > self.split_threshold
    
    def split_neuron(self, neuron_idx):
        """Split neuron along gradient direction"""
        print(f"✂️  Splitting neuron {neuron_idx} (conflict: {self.gradient_conflict[neuron_idx]:.3f})")
        
        # Normalize gradient direction
        direction = self.gradients[neuron_idx]
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Create two new neurons offset along gradient
        # Offset scales with input dimensionality
        offset = 1.0  # Much larger - need to actually separate in high-D space
        pos1 = self.positions[neuron_idx] + offset * direction
        pos2 = self.positions[neuron_idx] - offset * direction
        
        # Remove original, add two new
        self.positions = np.vstack([
            self.positions[:neuron_idx],
            pos1,
            pos2,
            self.positions[neuron_idx+1:]
        ])
        
        # Duplicate weights and velocities
        self.weights = np.vstack([
            self.weights[:neuron_idx],
            self.weights[neuron_idx:neuron_idx+1],
            self.weights[neuron_idx:neuron_idx+1],
            self.weights[neuron_idx+1:]
        ])
        
        self.velocities = np.vstack([
            self.velocities[:neuron_idx],
            self.velocities[neuron_idx:neuron_idx+1],
            self.velocities[neuron_idx:neuron_idx+1],
            self.velocities[neuron_idx+1:]
        ])
        
        self.gradients = np.vstack([
            self.gradients[:neuron_idx],
            self.gradients[neuron_idx:neuron_idx+1],
            self.gradients[neuron_idx:neuron_idx+1],
            self.gradients[neuron_idx+1:]
        ])

        self.gradient_conflict = np.concatenate([
            self.gradient_conflict[:neuron_idx],
            [0.0, 0.0],  # New neurons start with no conflict
            self.gradient_conflict[neuron_idx+1:]
        ])
    
    def should_merge(self, idx1, idx2):
        """Check if neurons are close enough to merge"""
        distance = np.linalg.norm(self.positions[idx1] - self.positions[idx2])
        return distance < self.merge_threshold
    
    def merge_neurons(self, idx1, idx2):
        """Merge two nearby neurons"""
        print(f"🔗 Merging neurons {idx1} and {idx2}")
        
        # Average position
        new_pos = (self.positions[idx1] + self.positions[idx2]) / 2
        new_weight = (self.weights[idx1] + self.weights[idx2]) / 2
        
        # Remove both, add merged
        keep_mask = np.ones(len(self.positions), dtype=bool)
        keep_mask[idx1] = False
        keep_mask[idx2] = False
        
        self.positions = np.vstack([self.positions[keep_mask], new_pos[np.newaxis, :]])
        self.weights = np.vstack([self.weights[keep_mask], new_weight[np.newaxis, :]])
        self.velocities = np.vstack([self.velocities[keep_mask],
                                     np.zeros((1, self.input_dim))])
        self.gradients = np.vstack([self.gradients[keep_mask],
                                   np.zeros((1, self.input_dim))])
        self.gradient_conflict = np.concatenate([self.gradient_conflict[keep_mask], [0.0]])
    
    def train_step(self, X, y):
        """Single training step with growth"""
        # Forward pass
        output, activations = self.forward(X)
        
        # Compute gradients
        weight_grad = self.compute_gradients(X, y, activations)
        
        # Update weights
        self.weights -= self.lr * weight_grad
        
        # Update positions along gradients - larger movement to differentiate
        self.positions -= self.lr * 1.0 * self.gradients
        
        # Debug: only print max conflict
        max_conflict = np.max(self.gradient_conflict)
        if max_conflict > 0.4:
            print(f"  Max conflict: {max_conflict:.3f}")
        
        # Check for splits - only split the MOST conflicted neuron per step
        # This gives neurons time to differentiate before more splitting
        if len(self.positions) > 0:
            most_conflicted = np.argmax(self.gradient_conflict)
            if self.should_split(most_conflicted):
                self.split_neuron(most_conflicted)
        
        # Check for merges
        merged = True
        while merged and len(self.positions) > 1:
            merged = False
            for i in range(len(self.positions)):
                for j in range(i + 1, len(self.positions)):
                    if self.should_merge(i, j):
                        self.merge_neurons(i, j)
                        merged = True
                        break
                if merged:
                    break
        
        # Calculate loss
        loss = np.mean((output - y)**2)
        
        return loss

# Load and prepare data
digits = load_digits()
X = digits.data[:100]  # Start small
y = np.eye(10)[digits.target[:100]]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create network - conflict threshold is 0-1 (0.5 = 50% disagreement)
net = GeometricNeuralNetwork(input_dim=64, output_dim=10,
                             split_threshold=0.5,  # Split when gradients 50% conflicting
                             merge_threshold=0.05)

# Training - more epochs for growth and learning
print("Starting with 1 seed neuron...")
for epoch in range(50):
    print(f"\n=== Epoch {epoch} ===")
    loss = net.train_step(X, y)
    accuracy = np.mean(np.argmax(net.forward(X)[0], axis=1) == digits.target[:100])
    print(f"Loss={loss:.4f}, Neurons={len(net.positions)}, Acc={accuracy:.3f}")

print(f"\nFinal: {len(net.positions)} neurons grown from 1 seed!")