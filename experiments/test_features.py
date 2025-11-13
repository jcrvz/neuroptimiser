"""Quick test to verify state features are changing"""
import numpy as np
import sys
sys.path.insert(0, '/Users/jcrvz/PycharmProjects/neuroptimiser')

from ioh import get_problem
from neuroptimiser.utils import trs2o

# Minimal setup
PROBLEM_ID = 3
NUM_DIMS = 10
LAMBDA = 50
MU = 25

problem = get_problem(fid=PROBLEM_ID, instance=1, dimension=NUM_DIMS)
problem.reset()

X_LOWER_BOUND = problem.bounds.lb
X_UPPER_BOUND = problem.bounds.ub
V_INITIAL_GUESS = np.zeros(NUM_DIMS)
F_INITIAL_GUESS = problem(trs2o(V_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND))
EPS = 1e-12
F_DEFAULT_WORST = 10 ** np.ceil(np.log10(F_INITIAL_GUESS + EPS))

def eval_obj_func(v):
    x = trs2o(v, X_LOWER_BOUND, X_UPPER_BOUND)
    return problem(x)

state = {
    "best_v": None,
    "best_f": None,
    "memory_vectors": np.zeros((MU, NUM_DIMS)),
    "memory_fitness": np.full(MU, F_DEFAULT_WORST),
    "improvement_history": [],
}

def update_memory(candidates, fitness):
    for i in range(len(candidates)):
        worst_idx = np.argmax(state["memory_fitness"])
        if fitness[i] < state["memory_fitness"][worst_idx]:
            state["memory_vectors"][worst_idx] = candidates[i]
            state["memory_fitness"][worst_idx] = fitness[i]

def compute_state_features():
    features = {"diversity": 0.0, "improvement_rate": 0.0, "convergence": 0.0}

    valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
    print(f"  DEBUG: valid_mask sum = {np.sum(valid_mask)}, F_DEFAULT_WORST = {F_DEFAULT_WORST}")

    if not np.any(valid_mask):
        print("  DEBUG: No valid solutions!")
        return features

    valid_fitness = state["memory_fitness"][valid_mask]
    valid_vectors = state["memory_vectors"][valid_mask]

    print(f"  DEBUG: len(valid_vectors) = {len(valid_vectors)}, len(valid_fitness) = {len(valid_fitness)}")

    # Diversity
    if len(valid_vectors) >= 2:
        pairwise_dists = []
        for i in range(min(10, len(valid_vectors))):
            for j in range(i + 1, min(10, len(valid_vectors))):
                dist = np.linalg.norm(valid_vectors[i] - valid_vectors[j])
                pairwise_dists.append(dist)
        avg_dist = np.mean(pairwise_dists)
        max_dist = 2.0 * np.sqrt(NUM_DIMS)
        diversity = avg_dist / max_dist
        print(f"  DEBUG: avg_dist = {avg_dist:.4f}, max_dist = {max_dist:.4f}, diversity = {diversity:.4f}")
        features["diversity"] = float(np.clip(diversity, 0.0, 1.0))

    # Improvement rate
    if len(state["improvement_history"]) >= 5:
        recent = state["improvement_history"][-20:]
        improvements = sum(1 for x in recent if x > 0.001)
        improvement_rate = improvements / len(recent)
        print(f"  DEBUG: improvements = {improvements}, len(recent) = {len(recent)}, rate = {improvement_rate:.4f}")
        features["improvement_rate"] = float(improvement_rate)

    # Convergence
    if len(valid_fitness) >= 2:
        f_range = np.max(valid_fitness) - np.min(valid_fitness)
        f_mean = np.mean(valid_fitness)
        print(f"  DEBUG: f_range = {f_range:.4f}, f_mean = {f_mean:.4f}")
        if f_mean > EPS:
            cv = f_range / f_mean
            convergence = 1.0 - np.clip(cv, 0.0, 1.0)
            print(f"  DEBUG: cv = {cv:.4f}, convergence = {convergence:.4f}")
            features["convergence"] = float(convergence)

    return features

# Simulate optimization
print("Timestep | Diversity | Improv Rate | Convergence | Best F")
print("-" * 60)

for t in range(100):
    # Generate candidates
    if state["best_v"] is None:
        centre = np.zeros(NUM_DIMS)
    else:
        centre = state["best_v"]

    noise = np.random.randn(LAMBDA, NUM_DIMS)
    candidates = np.clip(centre + 0.5 * noise, -1.0, 1.0)
    fitness = np.array([eval_obj_func(v) for v in candidates])

    # Update memory
    update_memory(candidates, fitness)

    # Track improvement
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

    # Compute features
    if t % 10 == 0:
        features = compute_state_features()
        print(f"{t:8d} | {features['diversity']:9.3f} | {features['improvement_rate']:11.3f} | {features['convergence']:11.3f} | {state['best_f']:7.1f}")

print("\nFeatures ARE changing!" if features['diversity'] > 0.01 else "\nFeatures NOT changing - BUG!")

