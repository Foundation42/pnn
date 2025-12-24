# BRIEFING DOCUMENT: Physical Neural Network PCB Compiler

Christian Beaumont - 24th December 2025

**Project Goal:** Build a system that trains neural networks where the physical PCB layout is part of the optimization process. The network "condenses" into an optimal geometric form that can be directly fabricated.

**Deliverable:** Working code that trains an XOR neural network with optimized component placement, then exports manufacturing files for JLCPCB.

---

## Project Overview

### What We're Building

A neural network trainer where:
1. **Weights** are encoded as copper trace widths/lengths
2. **Neuron positions** are trainable parameters (alongside weights)
3. **Physical constraints** (trace resistance, board size, spacing) are part of the loss function
4. **Output** is ready-to-manufacture Gerber files

### Why This Matters

- **No discrete components needed** for weights (just copper geometry)
- **Inference time:** ~5-10 nanoseconds (speed of electrons)
- **Power consumption:** <50mW
- **Cost:** ~$2 per board from JLCPCB
- **Novel research:** Physical-aware training hasn't been done for analog PCB neural networks

---

## Technical Architecture

### Phase 1: Core Simulation (This Week)

**File Structure:**
```
physical-neural-net/
├── physics.py              # Copper trace physics calculations
├── model.py                # PyTorch network with trainable positions
├── train.py                # Training loop for XOR
├── visualize.py            # Animation of condensation process
├── export_kicad.py         # Generate KiCad PCB files
└── requirements.txt        # Dependencies
```

### Core Components

#### 1. Physics Engine (`physics.py`)

**Purpose:** Calculate electrical properties of copper PCB traces

**Key Functions:**
- `trace_resistance(length_mm, width_mm, temperature_c)` → Returns resistance in Ω
  - Formula: `R = ρ × L / A` where `A = width × thickness`
  - Copper resistivity: `ρ = 1.68×10⁻⁸ Ω·m`
  - Standard PCB copper: 35μm thick (1oz)
  
- `manhattan_distance(pos1, pos2)` → Distance for trace routing
  - PCB traces follow grid (not Euclidean)
  - Used to calculate trace lengths
  
- `current_capacity(width_mm)` → Max safe current
  - Based on IPC-2221 standards
  - Important for preventing thermal damage

**Constraints:**
- JLC minimum trace width: 0.09mm
- JLC maximum practical width: 5mm
- Standard FR4 board thickness: 1.6mm

#### 2. Physical Neural Network (`model.py`)

**Purpose:** PyTorch neural network where neuron positions are trainable

**Key Innovations:**

**Trainable Parameters:**
```python
# Standard NN parameters
self.W1 = Parameter(...)  # Input → Hidden weights
self.W2 = Parameter(...)  # Hidden → Output weights

# NEW: Physical positions (also optimized!)
self.hidden_positions = Parameter(...)  # (x, y) coordinates in mm
```

**Effective Weight Calculation:**
```python
def compute_effective_weight(weight, trace_length):
    """
    Physical trace length affects signal strength!
    Longer traces = higher resistance = weaker connection
    """
    width = weight_to_trace_width(weight)
    R_trace = physics.trace_resistance(trace_length, width)
    conductance = 1.0 / (R_trace + 1.0)
    effective_weight = weight * conductance
    return effective_weight
```

**Layout Loss Function:**
```python
def layout_loss():
    """
    Penalize layouts that are hard to manufacture or inefficient.
    """
    penalties = []
    
    # 1. Board area usage (stay within bounds)
    penalties.append(board_utilization_penalty)
    
    # 2. Minimum spacing (neurons too close = manufacturing violation)
    penalties.append(spacing_violations)
    
    # 3. Total trace length (minimize copper usage)
    penalties.append(total_wire_length)
    
    # 4. Neurons off board (hard constraint)
    penalties.append(out_of_bounds_penalty * 100)
    
    return sum(penalties)
```

**Total Loss:**
```python
total_loss = classification_loss + α * layout_loss
```
where `α` controls the tradeoff between accuracy and manufacturability.

#### 3. Training (`train.py`)

**XOR Dataset:**
```python
X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0],   [1],   [1],   [0]]
```

**Training Process:**
1. Initialize neuron positions randomly on board
2. For each epoch:
   - Forward pass (calculate with effective weights)
   - Compute classification loss (BCE)
   - Compute layout loss (physical constraints)
   - Backprop through BOTH weights and positions
   - Update parameters
3. Watch neurons move to optimal positions!

**Expected Behavior:**
- Early: Neurons scattered, high layout loss
- Mid: Neurons clustering near high-weight connections
- Late: Tight, efficient layout with short traces

**Hyperparameters:**
```python
epochs = 5000
learning_rate_weights = 0.01
learning_rate_positions = 0.5  # Positions can move faster
layout_weight_alpha = 0.1      # Balance accuracy vs layout
board_size = (50, 50)          # mm
hidden_neurons = 4
```

#### 4. Visualization (`visualize.py`)

**Purpose:** Animate the "condensation" process

**Key Visualizations:**

1. **Loss Curves Over Time**
   - Classification loss (should decrease)
   - Layout loss (should decrease)
   - Total loss
   
2. **Physical Layout Evolution**
   - Scatter plot of neuron positions
   - Trace connections (width = weight magnitude)
   - Board boundary
   - Color-coded by layer
   
3. **Animation**
   - Frame-by-frame showing neurons moving
   - Save as `condensation.gif`
   
**Implementation:**
```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_condensation(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def update(frame):
        # Left: Loss curves
        ax1.plot(history['total_loss'][:frame])
        
        # Right: Physical layout at this epoch
        positions = history['hidden_positions'][frame]
        ax2.scatter(positions[:, 0], positions[:, 1])
        # Draw traces...
        
    anim = FuncAnimation(fig, update, frames=len(history['total_loss']))
    return anim
```

#### 5. Export to KiCad (`export_kicad.py`)

**Purpose:** Convert trained network to manufacturable PCB

**Output Files:**
- `.kicad_pcb` - PCB layout file
- Gerber files (for JLCPCB)
- Drill files
- BOM (just passive components + diodes)

**Key Conversions:**

**Neuron Position → PCB Coordinate:**
```python
def mm_to_kicad(x_mm, y_mm):
    # KiCad uses nanometers internally
    return (int(x_mm * 1e6), int(y_mm * 1e6))
```

**Weight → Trace Specification:**
```python
def weight_to_trace_spec(weight, length_mm):
    """
    Generate PCB trace specification.
    """
    width_mm = weight_to_trace_width(weight)
    
    return {
        'width': width_mm,
        'length': length_mm,
        'layer': 'F.Cu',  # Top layer
        'net': f'weight_{id}'
    }
```

**Crossbar Array Layout:**
```
Layer 1 (F.Cu):   Horizontal traces (inputs)
Layer 2 (B.Cu):   Vertical traces (outputs)
Vias:             Connect layers at crossings (weights)
```

**Additional Components:**
- 2-4 SMT diodes (1N4148) for nonlinearity at hidden layer
- Input pads (test points or header pins)
- Output LED + resistor
- Power supply connections (5V, GND)

---

## Implementation Roadmap

### Milestone 1: Basic Simulation (Days 1-2)
- [ ] Implement `physics.py` with trace calculations
- [ ] Implement `model.py` with trainable positions
- [ ] Train XOR network, verify it learns
- [ ] Basic visualization of final layout

**Success Criteria:** XOR achieves >95% accuracy, neurons stay on board

### Milestone 2: Optimization (Days 3-4)
- [ ] Add layout loss components
- [ ] Tune hyperparameters (α, learning rates)
- [ ] Create condensation animation
- [ ] Validate trace geometries are manufacturable

**Success Criteria:** Layout loss converges, all spacing constraints satisfied

### Milestone 3: Export (Days 5-6)
- [ ] Generate KiCad PCB file
- [ ] Export Gerber files
- [ ] Create BOM for JLCPCB
- [ ] Validate design rules (DRC check)

**Success Criteria:** Files ready to upload to JLCPCB

### Milestone 4: Documentation (Day 7)
- [ ] Write README with theory
- [ ] Document training results
- [ ] Create comparison: abstract vs physical training
- [ ] Prepare blog post for entrained.ai

---

## Key Design Decisions

### Why XOR First?
- Simplest nonlinear problem
- Only needs 4 hidden neurons
- Fits on small board (50mm × 50mm)
- Easy to validate (4 test cases)
- Proves concept before scaling

### Trace Width Mapping Strategy
```python
def weight_to_trace_width(weight):
    """
    Map weight [-3, +3] to trace width [0.09mm, 5mm]
    
    Positive weights → wider traces → lower R → stronger connection
    Negative weights → thinner traces → higher R → weaker connection
    """
    w_normalized = (clamp(weight, -3, 3) + 3) / 6  # → [0, 1]
    width = 0.09 + w_normalized * (5.0 - 0.09)
    return width
```

### Position Initialization
- **Inputs:** Fixed at left edge, evenly spaced
- **Hidden:** Random uniform across board
- **Outputs:** Fixed at right edge, evenly spaced

This creates natural left-to-right signal flow.

### Layer Assignment
**2-layer board (sufficient for XOR):**
- Top (F.Cu): Input → Hidden traces
- Bottom (B.Cu): Hidden → Output traces
- Vias connect at neuron positions

### Nonlinearity Implementation
**Option 1:** SMT diodes at hidden layer
- Provides sigmoid-like response
- Need ~2-4 diodes
- Part: 1N4148 (cheap, stock at JLC)

**Option 2:** Resistor nonlinearity
- Exploit trace heating (I²R)
- Zero additional components!
- Less precise, but interesting

Start with diodes for reliability.

---

## Testing Strategy

### Unit Tests
```python
# test_physics.py
def test_trace_resistance():
    """Known geometry should give known resistance."""
    physics = CopperPhysics()
    R = physics.trace_resistance(10, 1.0)  # 10mm × 1mm
    expected = 0.48  # Ω (calculated)
    assert abs(R - expected) < 0.01

def test_weight_mapping():
    """Weight range should map to valid trace widths."""
    for weight in [-3, 0, 3]:
        width = weight_to_trace_width(weight)
        assert 0.09 <= width <= 5.0
```

### Integration Tests
```python
# test_training.py
def test_xor_convergence():
    """Network should learn XOR to >95% accuracy."""
    model, history = train_xor(epochs=5000)
    final_accuracy = history['accuracy'][-1]
    assert final_accuracy > 0.95

def test_layout_constraints():
    """Trained layout should satisfy manufacturing rules."""
    model, history = train_xor(epochs=5000)
    
    # Check spacing
    positions = model.hidden_positions.detach()
    min_spacing = compute_min_spacing(positions)
    assert min_spacing > 2.0  # mm
    
    # Check bounds
    assert positions[:, 0].max() < 50  # Within board
    assert positions[:, 1].max() < 50
```

### Manufacturing Validation
1. Export Gerbers
2. Upload to JLCPCB
3. Run their DRC (Design Rule Check)
4. Should pass with zero violations

---

## Expected Results

### Training Convergence
- **Classification loss:** Should reach ~0.01 by epoch 3000
- **Layout loss:** Should reach ~5-10 (depending on α)
- **Accuracy:** >99% on XOR test set
- **Training time:** ~2-5 minutes on CPU

### Final Layout Characteristics
- **Board utilization:** ~60-80% of 50mm × 50mm
- **Total trace length:** ~200-300mm
- **Neuron clustering:** Strongly-connected pairs should be close
- **Symmetry:** Expect some symmetry due to XOR symmetry

### Physical Performance (Post-Fabrication)
- **Inference time:** ~5ns (calculated from trace RC)
- **Power consumption:** ~30mW @ 5V
- **Accuracy:** Should match simulation (±2% for manufacturing tolerances)

---

## Dependencies

```txt
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
pcbnew  # KiCad Python API (optional, for advanced export)
```

---

## Next Steps After XOR

Once XOR works, we can:

1. **Scale up:** MNIST classifier (784→128→10)
2. **Multi-layer boards:** 4-layer for complex networks
3. **Generalize framework:** Abstract `PhysicalNeuralNetwork` for other physics
4. **Add more physics:**
   - Thermal simulation (trace heating)
   - Signal propagation delay (RC time constants)
   - Crosstalk between parallel traces
5. **Manufacturing GAN:** Learn layout wisdom from real PCBs

---

## Success Metrics

**Minimum Viable Product:**
- ✓ XOR network trains to >95% accuracy
- ✓ Neurons stay within board bounds
- ✓ Spacing constraints satisfied
- ✓ Exports valid Gerber files

**Stretch Goals:**
- Condensation animation looks cool
- Layout is obviously better than random placement
- Total trace length reduced by >30% vs. random
- Publishable results (blog post ready)

---

## Questions for Claude Code

1. Should we use `pcbnew` Python API or generate Gerbers manually?
2. Best way to visualize trace widths in matplotlib (line width scaling)?
3. Should we add momentum to position updates (like particle physics)?
4. How to handle negative weights? (reverse polarity? just use abs value?)

---

## References

- IPC-2221: PCB design standards
- JLCPCB capabilities: https://jlcpcb.com/capabilities/pcb-capabilities
- KiCad file formats: https://docs.kicad.org/
- Backprop through physics: See diffusion models, differentiable rendering

---

**END OF BRIEFING**
