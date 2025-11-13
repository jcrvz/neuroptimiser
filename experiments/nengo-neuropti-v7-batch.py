#!/usr/bin/env python3
"""
===============================================================================
Neuromorphic Optimizer v7 - Batch Mode
===============================================================================
This is a batch-compatible version that accepts command-line arguments and
saves results to JSON files without generating plots.

Usage:
    python nengo-neuropti-v7-batch.py <fid> <instance> <dims> <output_file>

Example:
    python nengo-neuropti-v7-batch.py 1 1 10 result.json
===============================================================================
"""

import sys
import json
import numpy as np
import math
import nengo
from ioh import get_problem
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from neuroptimiser.utils import trs2o

# ==============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ==============================================================================

if len(sys.argv) != 5:
    print("Usage: python nengo-neuropti-v7-batch.py <fid> <instance> <dims> <output_file>")
    sys.exit(1)

PROBLEM_ID = int(sys.argv[1])
PROBLEM_INS = int(sys.argv[2])
NUM_DIMS = int(sys.argv[3])
OUTPUT_FILE = sys.argv[4]

print(f"Configuration: FID={PROBLEM_ID}, Instance={PROBLEM_INS}, Dims={NUM_DIMS}")

# ==============================================================================
# PROBLEM CONFIGURATION
# ==============================================================================

SIMULATION_TIME = 20.0

problem = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMS)
problem.reset()

X_LOWER_BOUND = problem.bounds.lb
X_UPPER_BOUND = problem.bounds.ub

V_INITIAL_GUESS = np.random.uniform(-1.0, 1.0, NUM_DIMS)
X_INITIAL_GUESS = trs2o(V_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
F_INITIAL_GUESS = problem(X_INITIAL_GUESS)

EPS = 1e-12
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
        self.alpha = 1.5

    def generate_population(self, centre):
        """Generate candidates using Lévy flight with weak directional bias"""
        global_best = state["best_v"] if state["best_v"] is not None else centre

        _alpha = 0.3
        beta = self.alpha
        sigma_u = (
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))
        ) ** (1 / beta)

        candidates = []
        for _ in range(LAMBDA):
            u = np.random.normal(0, sigma_u, NUM_DIMS)
            v = np.random.normal(0, 1, NUM_DIMS)
            step = u / (np.abs(v) ** (1 / beta))

            direction = global_best - centre
            candidate = centre + _alpha * step + 0.1 * direction
            candidates.append(np.clip(candidate, -1.0, 1.0))

        return np.array(candidates)


class DifferentialEvolution(Operator):
    """EXPLORATION: DE/rand/1 mutation using memory diversity"""

    def __init__(self):
        super().__init__("DM")
        self.F = 0.8

    def generate_population(self, centre):
        """Generate candidates using differential evolution"""
        valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
        valid_vectors = state["memory_vectors"][valid_mask]

        if len(valid_vectors) < 3:
            noise = np.random.randn(LAMBDA, NUM_DIMS)
            return np.clip(centre + 0.5 * noise, -1.0, 1.0)

        candidates = []
        for _ in range(LAMBDA):
            idx = np.random.choice(len(valid_vectors), 3, replace=False)
            a, b, c = valid_vectors[idx]
            mutant = a + self.F * (b - c)
            candidates.append(np.clip(mutant, -1.0, 1.0))

        return np.array(candidates)


class ParticleSwarm(Operator):
    """EXPLOITATION: PSO with velocity-based movement"""

    def __init__(self):
        super().__init__("PS")
        self.omega = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocities = np.zeros((LAMBDA, NUM_DIMS))

    def generate_population(self, centre):
        """Generate candidates using particle swarm"""
        global_best = state["best_v"] if state["best_v"] is not None else centre

        candidates = []
        for i in range(LAMBDA):
            particle = centre + np.random.randn(NUM_DIMS) * 0.1

            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (
                self.omega * self.velocities[i] +
                self.c1 * r1 * (centre - particle) +
                self.c2 * r2 * (global_best - particle)
            )

            self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)
            particle = particle + self.velocities[i]
            candidates.append(np.clip(particle, -1.0, 1.0))

        return np.array(candidates)


class SpiralOptimization(Operator):
    """EXPLOITATION: Anisotropic Spiral Optimisation"""

    def __init__(self):
        super().__init__("SP")
        self.r_base = 0.95
        self.min_theta = 1e-3
        self.max_theta = 2 * np.pi

    def generate_population(self, centre):
        """Generate candidates using anisotropic spiral"""
        global_best = state["best_v"] if state["best_v"] is not None else centre

        candidates = []
        n_planes = NUM_DIMS // 2

        for i in range(LAMBDA):
            x = centre.copy()

            for plane_idx in range(n_planes):
                d = plane_idx * 2
                theta_plane = np.random.uniform(self.min_theta, self.max_theta)
                r_variation_plane = np.random.uniform(0.9, 1.1)
                r_plane = np.clip(self.r_base * r_variation_plane, 0.85, 0.999)

                x_d = x[d:d+2] - global_best[d:d+2]
                r_theta = r_plane ** theta_plane

                cos_t = np.cos(theta_plane)
                sin_t = np.sin(theta_plane)

                x_rotated = r_theta * np.array([
                    cos_t * x_d[0] - sin_t * x_d[1],
                    sin_t * x_d[0] + cos_t * x_d[1]
                ])

                x[d:d+2] = global_best[d:d+2] + x_rotated

            if NUM_DIMS % 2 == 1:
                theta_odd = np.random.uniform(self.min_theta, self.max_theta)
                r_odd = np.clip(self.r_base * np.random.uniform(0.9, 1.1), 0.85, 0.999)
                x[-1] = global_best[-1] + r_odd ** theta_odd * (x[-1] - global_best[-1])

            candidates.append(np.clip(x, -1.0, 1.0))

        return np.array(candidates)


OPERATORS = {
    "LF": LevyFlight(),
    "DM": DifferentialEvolution(),
    "PS": ParticleSwarm(),
    "SP": SpiralOptimization()
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def update_memory(candidates, fitness):
    """Competitive memory update"""
    for i in range(len(candidates)):
        worst_idx = np.argmax(state["memory_fitness"])
        if fitness[i] < state["memory_fitness"][worst_idx]:
            state["memory_vectors"][worst_idx] = candidates[i]
            state["memory_fitness"][worst_idx] = fitness[i]
            state["memory_age"][worst_idx] = 0
        else:
            state["memory_age"] += 0.001


def get_centre():
    """Get fitness-weighted centre"""
    valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
    valid_fitness = state["memory_fitness"][valid_mask]
    valid_vectors = state["memory_vectors"][valid_mask]

    if len(valid_fitness) == 0:
        return np.random.uniform(-1.0, 1.0, NUM_DIMS)

    ranks = np.argsort(np.argsort(valid_fitness))
    weights = 1.0 / (ranks + 1.0)
    weights = weights / np.sum(weights)

    return np.average(valid_vectors, weights=weights, axis=0)


def compute_state_features():
    """Compute diversity, improvement rate, and convergence"""
    valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
    valid_vectors = state["memory_vectors"][valid_mask]
    valid_fitness = state["memory_fitness"][valid_mask]

    if len(valid_vectors) < 2:
        return {"diversity": 1.0, "improvement": 0.0, "convergence": 0.0}

    # Diversity
    std_per_dim = np.std(valid_vectors, axis=0)
    diversity = np.mean(std_per_dim) / (1.0 / np.sqrt(3))
    diversity = np.clip(diversity, 0.0, 1.0)

    # Improvement rate
    if len(state["improvement_history"]) > 0:
        recent = state["improvement_history"][-50:]
        improvement_rate = np.mean(recent)
    else:
        improvement_rate = 0.0

    # Convergence
    f_range = np.max(valid_fitness) - np.min(valid_fitness)
    best_f = np.min(valid_fitness)
    rho = f_range / (abs(best_f) + 0.01 * f_range + EPS)
    convergence = 1.0 / (1.0 + 10.0 * rho)
    convergence = np.clip(convergence, 0.0, 1.0)

    return {
        "diversity": diversity,
        "improvement": improvement_rate,
        "convergence": convergence
    }


# ==============================================================================
# BUILD NENGO MODEL
# ==============================================================================

model = nengo.Network(label="Neuromorphic Optimizer v7")

with model:
    n_operators = len(OPERATORS)

    # State features node
    def state_features_func(t):
        if t < INIT_TIME:
            return np.array([1.0, 0.0, 0.0])
        features = compute_state_features()
        return np.array([features["diversity"], features["improvement"], features["convergence"]])

    state_features_node = nengo.Node(state_features_func, size_out=3, label="State Features")

    # State ensemble
    state_ensemble = nengo.Ensemble(
        n_neurons=300,
        dimensions=3,
        radius=1.5,
        label="State (Diversity, Improvement, Convergence)"
    )
    nengo.Connection(state_features_node, state_ensemble, synapse=None)

    # Utility functions
    def utility_levy(s):
        w = state["utility_weights"]["LF"]
        return w * (0.0 * s[0] - 0.5 * s[1] - 0.3 * s[2] + 0.0)

    def utility_de(s):
        w = state["utility_weights"]["DM"]
        return w * (0.8 * s[0] + 0.0 * s[1] - 0.3 * s[2] + 0.2)

    def utility_pso(s):
        w = state["utility_weights"]["PS"]
        return w * (0.0 * s[0] + 0.8 * s[1] + 0.4 * s[2] + 0.2)

    def utility_spiral(s):
        w = state["utility_weights"]["SP"]
        return w * (0.0 * s[0] + 0.4 * s[1] + 0.8 * s[2] + 0.1)

    # Utility ensembles
    utility_levy_ens = nengo.Ensemble(n_neurons=NEURONS_BG, dimensions=1, radius=3.0, label="Utility LF")
    utility_de_ens = nengo.Ensemble(n_neurons=NEURONS_BG, dimensions=1, radius=3.0, label="Utility DM")
    utility_pso_ens = nengo.Ensemble(n_neurons=NEURONS_BG, dimensions=1, radius=3.0, label="Utility PS")
    utility_spiral_ens = nengo.Ensemble(n_neurons=NEURONS_BG, dimensions=1, radius=3.0, label="Utility SP")

    nengo.Connection(state_ensemble, utility_levy_ens, function=utility_levy, synapse=0.01)
    nengo.Connection(state_ensemble, utility_de_ens, function=utility_de, synapse=0.01)
    nengo.Connection(state_ensemble, utility_pso_ens, function=utility_pso, synapse=0.01)
    nengo.Connection(state_ensemble, utility_spiral_ens, function=utility_spiral, synapse=0.01)

    # Basal ganglia
    bg = nengo.networks.BasalGanglia(n_operators, n_neurons_per_ensemble=NEURONS_BG)
    nengo.Connection(utility_levy_ens, bg.input[0], synapse=None)
    nengo.Connection(utility_de_ens, bg.input[1], synapse=None)
    nengo.Connection(utility_pso_ens, bg.input[2], synapse=None)
    nengo.Connection(utility_spiral_ens, bg.input[3], synapse=None)

    # Thalamus
    thalamus = nengo.networks.Thalamus(n_operators, n_neurons_per_ensemble=NEURONS_BG)
    nengo.Connection(bg.output, thalamus.input, synapse=None)

    # Selected operator ensemble
    selected_operator_ens = nengo.Ensemble(
        n_neurons=NEURONS_BG * n_operators,
        dimensions=n_operators,
        radius=1.5,
        label="Selected Operator"
    )
    nengo.Connection(thalamus.output, selected_operator_ens, synapse=0.01)

    # Population generator
    def population_generator_func(t, operator_selection):
        if t < INIT_TIME:
            state["total_evals"] += LAMBDA
            return np.zeros(3)

        epsilon = 0.10
        if np.random.rand() < epsilon:
            operator_idx = np.random.randint(n_operators)
        else:
            operator_idx = np.argmax(operator_selection)

        operator_name = list(OPERATORS.keys())[operator_idx]

        # Learning rule
        if state["last_operator"] is not None and state["last_best_f"] is not None:
            if state["best_f"] is not None and state["best_f"] < state["last_best_f"]:
                reward = (state["last_best_f"] - state["best_f"]) / (abs(state["last_best_f"]) + EPS)
                learning_rate = 0.4
                state["utility_weights"][state["last_operator"]] += learning_rate * reward
                state["utility_weights"][state["last_operator"]] = np.clip(
                    state["utility_weights"][state["last_operator"]], 0.1, 5.0
                )
            else:
                penalty = 0.01
                state["utility_weights"][state["last_operator"]] -= penalty
                state["utility_weights"][state["last_operator"]] = max(
                    0.1, state["utility_weights"][state["last_operator"]]
                )

        state["last_operator"] = operator_name
        state["last_best_f"] = state["best_f"]
        state["current_operator"] = operator_name

        centre = get_centre()
        operator = OPERATORS[operator_name]
        candidates = operator.generate_population(centre)
        fitness = np.array([eval_obj_func(v) for v in candidates])

        update_memory(candidates, fitness)

        best_idx = np.argmin(fitness)
        if state["best_f"] is None or fitness[best_idx] < state["best_f"]:
            improvement = 1.0
            state["best_v"] = candidates[best_idx]
            state["best_f"] = fitness[best_idx]
        else:
            improvement = 0.0

        state["improvement_history"].append(improvement)
        if len(state["improvement_history"]) > 100:
            state["improvement_history"].pop(0)

        state["operator_counts"][operator_name] += 1
        state["total_evals"] += LAMBDA

        mean_f = np.mean(fitness)
        return np.array([state["best_f"], mean_f, float(operator_idx)])

    population_node = nengo.Node(
        population_generator_func,
        size_in=n_operators,
        size_out=3,
        label="Population Generator"
    )
    nengo.Connection(selected_operator_ens, population_node, synapse=0.05)

# ==============================================================================
# RUN SIMULATION
# ==============================================================================

print(f"Starting simulation for {SIMULATION_TIME}s...")

with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
    sim.run(SIMULATION_TIME)

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

total_ops = sum(state["operator_counts"].values())
operator_percentages = {
    op: 100.0 * count / total_ops if total_ops > 0 else 0.0
    for op, count in state["operator_counts"].items()
}

results = {
    "configuration": {
        "simulation_time": SIMULATION_TIME,
        "lambda": LAMBDA,
        "mu": MU,
        "neurons_per_dim": NEURONS_PER_DIM,
        "neurons_bg": NEURONS_BG,
    },
    "problem": {
        "function_id": PROBLEM_ID,
        "instance": PROBLEM_INS,
        "dimension": NUM_DIMS,
        "lower_bound": X_LOWER_BOUND.tolist(),
        "upper_bound": X_UPPER_BOUND.tolist(),
    },
    "performance": {
        "total_evaluations": state["total_evals"],
        "best_fitness": float(state["best_f"]) if state["best_f"] is not None else None,
        "optimal_fitness": float(problem.optimum.y),
        "error": float(state["best_f"] - problem.optimum.y) if state["best_f"] is not None else None,
    },
    "operator_usage": {op: int(count) for op, count in state["operator_counts"].items()},
    "operator_percentages": operator_percentages,
    "operator_weights": {op: float(w) for op, w in state["utility_weights"].items()},
    "best_solution": {
        "v_space": state["best_v"].tolist() if state["best_v"] is not None else None,
        "x_space": trs2o(state["best_v"], X_LOWER_BOUND, X_UPPER_BOUND).tolist() if state["best_v"] is not None else None,
    },
    "optimal_solution": {
        "x_space": problem.optimum.x.tolist(),
    }
}

# Save to JSON
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"Best fitness: {state['best_f']:.6e}")
print(f"Error: {state['best_f'] - problem.optimum.y:.6e}")
print(f"Total evaluations: {state['total_evals']:,}")
print(f"{'='*70}\n")

