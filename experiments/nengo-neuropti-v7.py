# %%
"""
===============================================================================
Neuromorphic Optimizer v7 - Basal Ganglia Operator Selection
===============================================================================

ARCHITECTURE OVERVIEW
---------------------
This optimizer uses neural basal ganglia circuits to select between multiple
optimization operators in real-time based on the current search state. Each
operator generates a population of candidates in parallel (LAMBDA = 50 per
timestep), enabling massive parallelism on neuromorphic hardware.

KEY FEATURES
------------
1. **Basal Ganglia Action Selection**: Winner-take-all circuit selects the
   best operator based on state-dependent utility functions

2. **Multiple Operators**: Currently implements four fundamentally different strategies:
   - LF: Lévy flight exploration (heavy-tailed jumps for escaping local minima)
   - DM: Differential Evolution (uses memory diversity for directed exploration)
   - PS: Particle Swarm Optimization (velocity-based exploitation with attraction)
   - SP: Spiral optimization (logarithmic convergence towards best solution)

3. **Population-Based Evaluation**: Each timestep evaluates LAMBDA candidates
   (50× more efficient than single-candidate approaches)

4. **Competitive Memory**: Fixed-size memory (MU=25) maintains best solutions
   through competitive updates

5. **State Features**: Three-dimensional state representation:
   - Diversity: Average pairwise distance in solution space (0=identical, 1=maximally spread)
   - Improvement rate: Fraction of recent timesteps that improved best-so-far (0=stagnant, 1=always improving)
   - Convergence: Fitness homogeneity indicator (0=diverse fitness, 1=similar fitness)

NEUROMORPHIC COMPONENTS
-----------------------
- State Ensemble (300 neurons): Encodes [diversity, improvement, convergence]
- Utility Ensembles (3×100 neurons): Compute operator-specific utilities
- Basal Ganglia (300 neurons): Winner-take-all selection across operators
- Thalamus (300 neurons): Action gating with mutual inhibition
- Selected Operator (300 neurons): One-hot encoding of active operator

OPERATOR SELECTION LOGIC
-------------------------
LF operator selected when:
  utility = 2.0 * (1 - improvement) + (1 - convergence) - 0.3 * diversity
  → High when stuck (not improving) and not converged (need global escape)

DM operator selected when:
  utility = 1.5 * diversity + (1 - convergence) + 0.5 * improvement
  → High when memory has diversity and still exploring

PS operator selected when:
  utility = 2.0 * improvement + 0.8 * convergence - 0.5 * diversity
  → High when improving and converging (exploitation phase)

SP operator selected when:
  utility = 2.0 * convergence + improvement - diversity
  → High when highly converged and still improving (fine-tuning)

EXTENSIBILITY
-------------
To add a new operator:
1. Define operator class inheriting from Operator base class
2. Add to OPERATORS dictionary
3. Add utility function in basal ganglia section
4. Increase n_operators count

PERFORMANCE
-----------
- Evaluations per second: ~50,000 (LAMBDA × dt^-1)
- Total evaluations (20s): ~1,000,000
- Memory footprint: 25 solutions × (D+1) floats
- Neuron count: ~1,200 total

BIOLOGICAL ANALOGY
------------------
The basal ganglia architecture mirrors cortico-basal ganglia-thalamic loops
in the mammalian brain, where:
- Cortex (state ensemble) → encodes sensory/internal state
- Striatum (utility ensembles) → evaluates action values
- Globus pallidus (basal ganglia) → selects best action via lateral inhibition
- Thalamus → gates selected action to motor cortex
- Motor cortex (population generator) → executes selected action

CITATION
--------
If you use this code, please cite the Nengo framework:
  Bekolay et al. (2014). Nengo: a Python tool for building large-scale
  functional brain models. Frontiers in Neuroinformatics, 7, 48.

===============================================================================
"""

import matplotlib.pyplot as plt
import nengo
import numpy as np
import math
from ioh import get_problem
import seaborn as sns

from neuroptimiser.utils import trs2o

# %%
# ==============================================================================
# PROBLEM CONFIGURATION
# ==============================================================================

PROBLEM_ID = 20
PROBLEM_INS = 1
NUM_DIMS = 2
SIMULATION_TIME = 20.0

problem = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMS)
problem.reset()
print(f"Problem: {problem}")

X_LOWER_BOUND = problem.bounds.lb
X_UPPER_BOUND = problem.bounds.ub

V_INITIAL_GUESS = np.random.uniform(-1.0, 1.0, NUM_DIMS)
X_INITIAL_GUESS = trs2o(V_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
F_INITIAL_GUESS = problem(X_INITIAL_GUESS)

EPS = 1e-12
# F_DEFAULT_WORST must be larger than any achievable fitness
# For minimization: use a large positive number regardless of problem scale
F_DEFAULT_WORST = 1e10


def eval_obj_func(v):
    """Evaluate objective function from v-space ([-1,1]^d)"""
    x = trs2o(v, X_LOWER_BOUND, X_UPPER_BOUND)
    return problem(x)


# ==============================================================================
# OPTIMIZER CONFIGURATION
# ==============================================================================

LAMBDA = 50  # Population size per operator per timestep
MU = 25  # Memory size

# Neuron configuration
NEURONS_PER_DIM = 100
NEURONS_BG = 100  # Neurons per action in basal ganglia

# Operator parameters
SIGMA_GLOBAL = 0.5  # Large steps for global exploration
SIGMA_LOCAL = 0.05  # Small steps for local refinement
SIGMA_ADAPTIVE = None  # Will be computed from memory variance

INIT_TIME = 0.001

# ==============================================================================
# GLOBAL STATE
# ==============================================================================

state = {
    "best_v": None,
    "best_f": None,
    "memory_vectors": np.zeros((MU, NUM_DIMS)),
    "memory_fitness": np.full(MU, F_DEFAULT_WORST),
    "memory_age": np.zeros(MU),
    "current_operator": "LF",
    "operator_counts": {"LF": 0, "DM": 0, "PS": 0, "SP": 0},
    "total_evals": 0,
    "improvement_history": [],
    # Adaptive utility weights - learn which operators work best
    "utility_weights": {"LF": 1.0, "DM": 1.0, "PS": 1.0, "SP": 1.0},
    "operator_rewards": {"LF": [], "DM": [], "PS": [], "SP": []},
    "last_operator": None,
    "last_best_f": None,
}

# ==============================================================================
# OPERATOR DEFINITIONS
# ==============================================================================


class Operator:
    """Base class for optimization operators"""

    def __init__(self, name):
        self.name = name

    def generate_population(self, centre):
        """Generate LAMBDA candidates - must be implemented by subclass"""
        raise NotImplementedError


class LevyFlight(Operator):
    """EXPLORATION: Heavy-tailed Lévy flight for escaping local minima"""

    def __init__(self):
        super().__init__("LF")
        self.alpha = 1.5  # Lévy exponent

    def generate_population(self, centre):
        """Generate candidates using Lévy flight"""
        # Mantegna's algorithm for Lévy flight
        global_best = state["best_v"] if state["best_v"] is not None else centre

        _alpha  = 0.3
        beta    = self.alpha
        sigma_u = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))
        ) ** (1 / beta)

        candidates = []
        for _ in range(LAMBDA):
            u       = np.random.normal(0, sigma_u, NUM_DIMS)
            v       = np.random.normal(0, 1, NUM_DIMS)
            step    = u / (np.abs(v) ** (1 / beta))

            direction = global_best - centre

            # Scale step and add to centre
            candidate = centre + _alpha * step + 0.1 * direction
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class DifferentialEvolution(Operator):
    """EXPLORATION: DM/rand/1 mutation using memory diversity"""

    def __init__(self):
        super().__init__("DM")
        self.F = 0.8  # Mutation factor

    def generate_population(self, centre):
        """Generate candidates using differential evolution"""
        valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
        valid_vectors = state["memory_vectors"][valid_mask]

        if len(valid_vectors) < 3:
            # Fallback to random exploration
            noise = np.random.randn(LAMBDA, NUM_DIMS)
            return np.clip(centre + 0.5 * noise, -1.0, 1.0)

        candidates = []
        for _ in range(LAMBDA):
            # Select 3 random distinct solutions
            idx = np.random.choice(len(valid_vectors), 3, replace=False)
            a, b, c = valid_vectors[idx]

            # DM mutation: a + F * (b - c)
            mutant = a + self.F * (b - c)
            candidates.append(np.clip(mutant, -1.0, 1.0))

        return np.array(candidates)


class ParticleSwarm(Operator):
    """EXPLOITATION: PS with personal and global best attraction"""

    def __init__(self):
        super().__init__("PS")
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        # Initialize velocity for each potential particle
        self.velocities = {}

    def generate_population(self, centre):
        """Generate candidates using PS dynamics"""
        global_best = state["best_v"] if state["best_v"] is not None else centre

        candidates = []
        for i in range(LAMBDA):
            # Get or initialize velocity for this particle
            if i not in self.velocities:
                self.velocities[i] = np.random.randn(NUM_DIMS) * 0.1

            # Current position (sample around centre with small noise)
            current = centre + np.random.randn(NUM_DIMS) * 0.05

            # PS velocity update
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (centre - current)  # Personal best = centre
            social = self.c2 * r2 * (global_best - current)  # Global best

            self.velocities[i] = (
                self.w * self.velocities[i] +
                cognitive +
                social
            )

            # Clip velocity
            self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)

            # Update position
            new_position = current + self.velocities[i]
            candidates.append(np.clip(new_position, -1.0, 1.0))

        return np.array(candidates)


class SpiralOptimization(Operator):
    """EXPLOITATION: Anisotropic Spiral Optimisation with per-plane randomisation"""

    def __init__(self):
        super().__init__("SP")
        self.r_base = 0.95  # Base convergence rate
        self.min_theta = 1e-3
        self.max_theta = 2 * np.pi

    def generate_population(self, centre):
        """Generate candidates using anisotropic spiral with per-plane randomisation"""
        global_best = state["best_v"] if state["best_v"] is not None else centre

        candidates = []
        n_planes = NUM_DIMS // 2  # Number of 2D planes

        for i in range(LAMBDA):
            x = centre.copy()

            # INDEPENDENT rotation and convergence for each 2D plane
            for plane_idx in range(n_planes):
                d = plane_idx * 2  # Dimensional indices for this plane

                # Per-plane random rotation angle
                theta_plane = np.random.uniform(self.min_theta, self.max_theta)

                # Per-plane random convergence rate
                r_variation_plane = np.random.uniform(0.9, 1.1)
                r_plane = np.clip(self.r_base * r_variation_plane, 0.85, 0.999)

                # Extract 2D slice for this plane
                x_d = x[d:d+2] - global_best[d:d+2]

                # Spiral transformation with plane-specific parameters
                r_theta = r_plane ** theta_plane

                cos_t = np.cos(theta_plane)
                sin_t = np.sin(theta_plane)

                x_rotated = r_theta * np.array([
                    cos_t * x_d[0] - sin_t * x_d[1],
                    sin_t * x_d[0] + cos_t * x_d[1]
                ])

                x[d:d+2] = global_best[d:d+2] + x_rotated

            # Handle odd dimension independently
            if NUM_DIMS % 2 == 1:
                theta_odd = np.random.uniform(self.min_theta, self.max_theta)
                r_odd = np.clip(self.r_base * np.random.uniform(0.9, 1.1), 0.85, 0.999)
                x[-1] = global_best[-1] + r_odd ** theta_odd * (x[-1] - global_best[-1])

            candidates.append(np.clip(x, -1.0, 1.0))

        return np.array(candidates)



# Instantiate operators
OPERATORS = {
    "LF": LevyFlight(),
    "DM": DifferentialEvolution(),
    "PS": ParticleSwarm(),
    "SP": SpiralOptimization(),
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def update_memory(candidates, fitness):
    """Competitive memory update with population"""
    for i in range(len(candidates)):
        worst_idx = np.argmax(state["memory_fitness"])
        if fitness[i] < state["memory_fitness"][worst_idx]:
            state["memory_vectors"][worst_idx] = candidates[i]
            state["memory_fitness"][worst_idx] = fitness[i]
            state["memory_age"][worst_idx] = 0.0

    state["memory_age"] += 1.0


def get_centre():
    """Get fitness-weighted centroid from memory"""
    valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
    if not np.any(valid_mask):
        return V_INITIAL_GUESS.copy()

    valid_fitness = state["memory_fitness"][valid_mask]
    valid_vectors = state["memory_vectors"][valid_mask]

    # Rank-based weighting
    ranks = np.argsort(np.argsort(valid_fitness))
    weights = 1.0 / (ranks + 1.0)
    weights /= np.sum(weights)

    return np.average(valid_vectors, axis=0, weights=weights)


def compute_state_features():
    """Compute features describing current optimization state"""
    features = {
        "diversity": 0.5,  # Default to medium
        "improvement_rate": 0.5,  # Default to medium
        "convergence": 0.0,  # Default to not converged
    }

    valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
    n_valid = np.sum(valid_mask)

    if n_valid < 2:
        return features

    valid_fitness = state["memory_fitness"][valid_mask]
    valid_vectors = state["memory_vectors"][valid_mask]

    # 1. DIVERSITY: Spread of solutions in search space
    # Use standard deviation across all dimensions, then average
    std_per_dim = np.std(valid_vectors, axis=0)
    avg_std = np.mean(std_per_dim)
    # In [-1,1] space, std can range from 0 (identical) to ~0.577 (uniform)
    # Map to [0,1] range
    diversity = np.clip(avg_std / 0.6, 0.0, 1.0)
    features["diversity"] = float(diversity)

    # 2. IMPROVEMENT RATE: How often we're finding better solutions
    if len(state["improvement_history"]) >= 10:
        # Look at last 50 timesteps
        recent = state["improvement_history"][-50:]
        # Improvement rate = fraction that improved
        rate = np.mean(recent)
        features["improvement_rate"] = float(rate)

    # 3. CONVERGENCE: How close fitness values are
    # Use relative fitness range
    f_min = np.min(valid_fitness)
    f_max = np.max(valid_fitness)
    f_range = f_max - f_min

    # Convergence is high when range is small relative to absolute values
    # Use absolute value of min fitness as reference
    if abs(f_min) > EPS:
        relative_range = f_range / abs(f_min)
        # Map: small range = high convergence
        # relative_range: 0 → conv=1, 0.1 → conv=0.5, 1.0+ → conv=0
        convergence = 1.0 / (1.0 + 10.0 * relative_range)
        features["convergence"] = float(convergence)
    else:
        # Near-zero fitness, use absolute range
        # Small absolute range = converged
        convergence = 1.0 - np.clip(f_range / 100.0, 0.0, 1.0)
        features["convergence"] = float(convergence)

    return features


# ==============================================================================
# NENGO MODEL WITH BASAL GANGLIA OPERATOR SELECTION
# ==============================================================================

model = nengo.Network(label="nNeurOpti v7 - Basal Ganglia Operator Selection", seed=69)

with model:
    # ==========================================================================
    # STATE FEATURE COMPUTATION
    # ==========================================================================

    def state_features_func(t):
        """Compute 3D state vector: [diversity, improvement_rate, convergence]"""
        if t < INIT_TIME:
            return np.array([0.5, 0.5, 0.0])  # Neutral initial state

        features = compute_state_features()


        return np.array([
            features["diversity"],
            features["improvement_rate"],
            features["convergence"]
        ])

    state_features_node = nengo.Node(state_features_func, label="State Features")

    # State representation ensemble
    state_ensemble = nengo.Ensemble(
        n_neurons=300,
        dimensions=3,
        radius=1.5,
        label="State Ensemble"
    )

    nengo.Connection(state_features_node, state_ensemble, synapse=None)

    # ==========================================================================
    # BASAL GANGLIA OPERATOR SELECTION
    # ==========================================================================

    # Define utility functions for each operator
    # Input: [diversity, improvement_rate, convergence]

    def utility_levy(x):
        """LF: Use when stuck (low improvement) and not converged (need global exploration)"""
        diversity, improvement, convergence = x
        # Reduced base utility - only use when really stuck
        base = (1.0 - improvement) * 0.5 + (1.0 - convergence) * 0.3
        # Modulate by learned weight
        return base * state["utility_weights"]["LF"]

    def utility_de(x):
        """DM: Use when have diversity in memory and exploring"""
        diversity, improvement, convergence = x
        # Favor when diversity exists
        base = diversity * 0.8 + (1.0 - convergence) * 0.3 + 0.2  # Added small bias
        # Modulate by learned weight
        return base * state["utility_weights"]["DM"]

    def utility_pso(x):
        """PS: Use when improving and starting to converge (exploitation phase)"""
        diversity, improvement, convergence = x
        # Favor improvement and convergence
        base = improvement * 0.8 + convergence * 0.4 + 0.2  # Added small bias
        # Modulate by learned weight
        return base * state["utility_weights"]["PS"]

    def utility_spiral(x):
        """SP: Use when highly converged and still improving (fine-tuning)"""
        diversity, improvement, convergence = x
        # Favor when very converged
        base = convergence * 0.8 + improvement * 0.4 + 0.1  # Small bias for fine-tuning
        # Modulate by learned weight
        return base * state["utility_weights"]["SP"]

    # Number of operators
    n_operators = 4

    # Utility ensemble for each operator
    utility_levy_ens = nengo.Ensemble(
        n_neurons=NEURONS_BG,
        dimensions=1,
        radius=3.0,
        label="Utility LF"
    )

    utility_de_ens = nengo.Ensemble(
        n_neurons=NEURONS_BG,
        dimensions=1,
        radius=3.0,
        label="Utility DM"
    )

    utility_pso_ens = nengo.Ensemble(
        n_neurons=NEURONS_BG,
        dimensions=1,
        radius=3.0,
        label="Utility PS"
    )

    utility_spiral_ens = nengo.Ensemble(
        n_neurons=NEURONS_BG,
        dimensions=1,
        radius=3.0,
        label="Utility SP"
    )

    # Connect state to utilities
    nengo.Connection(state_ensemble, utility_levy_ens, function=utility_levy, synapse=0.01)
    nengo.Connection(state_ensemble, utility_de_ens, function=utility_de, synapse=0.01)
    nengo.Connection(state_ensemble, utility_pso_ens, function=utility_pso, synapse=0.01)
    nengo.Connection(state_ensemble, utility_spiral_ens, function=utility_spiral, synapse=0.01)

    # Basal ganglia network (winner-take-all)
    bg = nengo.networks.BasalGanglia(n_operators, n_neurons_per_ensemble=NEURONS_BG)

    # Connect utilities to basal ganglia
    nengo.Connection(utility_levy_ens, bg.input[0], synapse=None)
    nengo.Connection(utility_de_ens, bg.input[1], synapse=None)
    nengo.Connection(utility_pso_ens, bg.input[2], synapse=None)
    nengo.Connection(utility_spiral_ens, bg.input[3], synapse=None)

    # Thalamus (action selection with inhibition)
    thalamus = nengo.networks.Thalamus(n_operators, n_neurons_per_ensemble=NEURONS_BG)
    nengo.Connection(bg.output, thalamus.input, synapse=None)

    # Selected operator ensemble (one-hot encoded: [is_levy, is_de, is_pso, is_spiral])
    selected_operator_ens = nengo.Ensemble(
        n_neurons=NEURONS_BG * n_operators,
        dimensions=n_operators,
        radius=1.5,
        label="Selected Operator"
    )

    nengo.Connection(thalamus.output, selected_operator_ens, synapse=0.01)

    # ==========================================================================
    # POPULATION GENERATOR
    # ==========================================================================

    def population_generator_func(t, operator_selection):
        """Generate population using selected operator"""
        if t < INIT_TIME:
            state["total_evals"] += LAMBDA
            return np.zeros(3)  # [best_f, mean_f, operator_id]

        # EPSILON-GREEDY: 10% random operator selection for exploration
        epsilon = 0.10
        if np.random.rand() < epsilon:
            # Random exploration
            operator_idx = np.random.randint(n_operators)
        else:
            # Use basal ganglia selection
            operator_idx = np.argmax(operator_selection)

        operator_name = list(OPERATORS.keys())[operator_idx]

        # LEARNING RULE: Update utility weights based on last operator's performance
        # ...existing code...
        if state["last_operator"] is not None and state["last_best_f"] is not None:
            # Compute reward: did the last operator improve the best fitness?
            if state["best_f"] is not None and state["best_f"] < state["last_best_f"]:
                # Improvement! Give positive reward
                reward = (state["last_best_f"] - state["best_f"]) / (abs(state["last_best_f"]) + EPS)
                # Increase weight for successful operator
                learning_rate = 0.4
                state["utility_weights"][state["last_operator"]] += learning_rate * reward
                # Keep weights in reasonable range
                state["utility_weights"][state["last_operator"]] = np.clip(
                    state["utility_weights"][state["last_operator"]], 0.1, 5.0
                )
            else:
                # No improvement - small penalty to encourage exploration
                penalty = 0.01
                state["utility_weights"][state["last_operator"]] -= penalty
                state["utility_weights"][state["last_operator"]] = max(
                    0.1, state["utility_weights"][state["last_operator"]]
                )

        # Update tracking for next iteration
        state["last_operator"] = operator_name
        state["last_best_f"] = state["best_f"]
        state["current_operator"] = operator_name

        # Generate centre
        centre = get_centre()

        # Generate population with selected operator
        operator = OPERATORS[operator_name]
        candidates = operator.generate_population(centre)

        # Evaluate population
        fitness = np.array([eval_obj_func(v) for v in candidates])

        # Update memory
        update_memory(candidates, fitness)

        # Update best-so-far and track improvement
        best_idx = np.argmin(fitness)
        prev_best = state["best_f"] if state["best_f"] is not None else F_DEFAULT_WORST

        if state["best_f"] is None or fitness[best_idx] < state["best_f"]:
            # Improvement happened
            improvement = 1.0
            state["best_v"] = candidates[best_idx]
            state["best_f"] = fitness[best_idx]
        else:
            # No improvement
            improvement = 0.0

        # Track improvement every timestep
        state["improvement_history"].append(improvement)
        if len(state["improvement_history"]) > 100:
            state["improvement_history"].pop(0)

        # Track operator usage
        state["operator_counts"][operator_name] += 1
        state["total_evals"] += LAMBDA

        # Return statistics
        mean_f = np.mean(fitness)

        return np.array([state["best_f"], mean_f, float(operator_idx)])

    population_node = nengo.Node(
        population_generator_func,
        size_in=n_operators,
        size_out=3,
        label="Population Generator"
    )

    nengo.Connection(selected_operator_ens, population_node, synapse=0.05)

    # ==========================================================================
    # PROBES
    # ==========================================================================

    # Probe the node directly to get raw feature values (no neural filtering)
    state_features_probe = nengo.Probe(state_features_node, synapse=None, label="Raw Features")
    # Also probe the ensemble to see neural representation
    state_probe = nengo.Probe(state_ensemble, synapse=0.01)
    utilities_probe = nengo.Probe(bg.input, synapse=0.01)
    operator_probe = nengo.Probe(selected_operator_ens, synapse=0.01)
    stats_probe = nengo.Probe(population_node, synapse=None)

# %%
# ==============================================================================
# RUN SIMULATION
# ==============================================================================

print(f"\nStarting simulation for {SIMULATION_TIME}s...")
print(f"Population size: {LAMBDA} per operator per timestep")
print(f"Total expected evaluations: ~{int(SIMULATION_TIME * 1000 * LAMBDA)}\n")

with nengo.Simulator(model, dt=0.001) as sim:
    sim.run(SIMULATION_TIME)

# %%
# ==============================================================================
# RESULTS
# ==============================================================================

print("=" * 70)
print("OPTIMIZATION RESULTS")
print("=" * 70)

total_evals = state["total_evals"]
print(f"\nTotal evaluations: {total_evals:,}")
print(f"Best fitness: {state['best_f']:.6f}")
print(f"Optimal fitness: {problem.optimum.y:.6f}")
print(f"Error: {state['best_f'] - problem.optimum.y:.6f}")

print(f"\nOperator usage:")
for op_name, count in state["operator_counts"].items():
    pct = 100 * count / sum(state["operator_counts"].values())
    weight = state["utility_weights"][op_name]
    print(f"  {op_name:10s}: {count:6d} calls ({pct:5.1f}%)  [weight: {weight:.3f}]")

if state["best_v"] is not None:
    xbest = trs2o(state["best_v"], X_LOWER_BOUND, X_UPPER_BOUND)
    print(f"\n{'Dim':3s}  {'Best':>10s}  {'Optimal':>10s}  {'Error':>10s}")
    print("-" * 40)
    for i in range(NUM_DIMS):
        error = xbest[i] - problem.optimum.x[i]
        print(f"{i + 1:3d}  {xbest[i]:10.4f}  {problem.optimum.x[i]:10.4f}  {error:10.4f}")

# %%
# ==============================================================================
# VISUALIZATION
# ==============================================================================
fontsize = 14
plt.rcParams.update({
    # LaTeX rendering
    'text.usetex': True,
    # 'text.latex.preamble': r'\usepackage{amsmath}',

    # Fonts
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize,

    # Figure
    'figure.dpi': 300,
    'figure.figsize': (14, 8),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # Lines and markers
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.linewidth': 0.8,

    # Grid
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'grid.color': '0.9',
    'axes.grid': True,

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',

    # Ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

# Extract data
stats_data = sim.data[stats_probe]
best_f_trace = stats_data[:, 0]
mean_f_trace = stats_data[:, 1]
operator_trace = stats_data[:, 2]

# Plot 1: Fitness evolution
ax1 = axes[0]
valid_mask = best_f_trace < F_DEFAULT_WORST
t_valid = sim.trange()[valid_mask]
best_valid = best_f_trace[valid_mask]
mean_valid = mean_f_trace[valid_mask]

ax1.plot(t_valid, best_valid - problem.optimum.y, 'b-', linewidth=2, label='Best-so-far error')
ax1.plot(t_valid, mean_valid - problem.optimum.y, 'gray', alpha=0.3, linewidth=0.5, label='Population mean error')
ax1.set_ylabel('Fitness Error')
ax1.set_yscale('log')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Operator selection
# ax2 = axes[1]
# operator_names = list(OPERATORS.keys())
# colors = {'LF': 'red', 'DM': 'orange', 'PS': 'green', 'SP': 'blue'}
#
# for i, op_name in enumerate(operator_names):
#     mask = operator_trace == i
#     if np.any(mask):
#         ax2.scatter(sim.trange()[mask], np.ones(np.sum(mask)) * i,
#                     c=colors.get(op_name, 'gray'), s=1, alpha=0.5, label=op_name)
#
# ax2.set_ylabel('Active Operator')
# ax2.set_yticks(range(len(operator_names)))
# ax2.set_yticklabels(operator_names)
# ax2.legend(loc='upper right')
# ax2.grid(True, alpha=0.3)

# Plot 2: Operator selection with horizontal violin plots
ax2 = axes[1]
operator_names = list(OPERATORS.keys())
colors = {'LF': 'red', 'DM': 'orange', 'PS': 'green', 'SP': 'blue'}

# Histogram-style visualization
for i, op_name in enumerate(operator_names):
    mask = operator_trace == i
    times = sim.trange()[mask]

    if len(times) > 0:
        # Plot vertical lines at each activation
        ax2.vlines(times, i - 0.3, i + 0.3,
                   colors=colors[op_name], alpha=0.3, linewidths=0.5)
        # Add density line
        hist, edges = np.histogram(times, bins=50)
        centers = (edges[:-1] + edges[1:]) / 2
        density = hist / hist.max() * 0.4  # Normalize to ±0.4 height
        ax2.fill_between(centers, i - density, i + density,
                         color=colors[op_name], alpha=0.5, label=op_name)

ax2.set_ylabel('Active Operator')
ax2.set_yticks(range(len(operator_names)))
ax2.set_yticklabels(operator_names)
# ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_ylim(-0.5, len(operator_names) - 0.5)

# Plot 3: State features (use RAW features from node, not filtered ensemble)
ax3 = axes[2]
# Use state_features_probe (raw) instead of state_probe (neural filtered)
raw_features = sim.data[state_features_probe]  # Shape: (timesteps, 3)
neural_features = sim.data[state_probe]  # For comparison

# Downsample for visibility
downsample = 10
time_ds = sim.trange()[::downsample]
diversity_raw = raw_features[::downsample, 0]
improvement_raw = raw_features[::downsample, 1]
convergence_raw = raw_features[::downsample, 2]

# Plot raw features (solid lines)
ax3.plot(time_ds, diversity_raw, 'r-', label=r'Diversity ($\phi_s$)', alpha=0.8, linewidth=1.5)
ax3.plot(time_ds, improvement_raw, 'g-', label=r'Improvement Rate ($\phi_i$)', alpha=0.8, linewidth=1.5)
ax3.plot(time_ds, convergence_raw, 'b-', label=r'Convergence ($\phi_c$)', alpha=0.8, linewidth=1.5)

# Optionally plot neural filtered versions (dashed, for comparison)
diversity_neural = neural_features[::downsample, 0]
improvement_neural = neural_features[::downsample, 1]
convergence_neural = neural_features[::downsample, 2]

ax3.plot(time_ds, diversity_neural, 'r--', alpha=0.3, linewidth=1.0, label='_diversity_neural')
ax3.plot(time_ds, improvement_neural, 'g--', alpha=0.3, linewidth=1.0, label='_improvement_neural')
ax3.plot(time_ds, convergence_neural, 'b--', alpha=0.3, linewidth=1.0, label='_convergence_neural')

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Feature Value')
ax3.set_ylim(-0.1, 1.1)
ax3.legend(loc='center right')
# ax3.grid(True, alpha=0.3)

plt.suptitle(f'{problem.meta_data.name} (f{problem.meta_data.problem_id:02d}) {NUM_DIMS}D; '
             # f'Evals: {total_evals:,}; '
             f'Best: {state["best_f"]:.2f}; '
             f'Error: {state["best_f"] - problem.optimum.y:.2g}')

plt.tight_layout()

FOLDER = "./simple_plots/"

plt.savefig(FOLDER +
    f'{problem.meta_data.problem_id:02d}_{NUM_DIMS}D_neuroptibg.png')

plt.show()

# %%
print("\n" + "=" * 70)
print("Simulation complete!")
print("=" * 70)

