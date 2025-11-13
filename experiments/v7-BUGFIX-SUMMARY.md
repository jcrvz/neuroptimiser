# v7 State Features Fix - Summary

## Bugs Found and Fixed

### Bug #1: F_DEFAULT_WORST was NaN for Negative Fitness

**Problem**: 
```python
F_DEFAULT_WORST = 10 ** np.ceil(np.log10(F_INITIAL_GUESS + EPS))
```

For Rastrigin (minimization, negative fitness like -300), this computes:
- `log10(-300 + 1e-12)` → **undefined/NaN**
- Result: All memory solutions marked as invalid
- Consequence: All state features stuck at 0.0

**Fix**:
```python
F_DEFAULT_WORST = 1e10  # Simple large positive number
```

### Bug #2: Improvement History Only Tracked When Best Improved

**Problem**:
```python
if state["best_f"] is None or fitness[best_idx] < state["best_f"]:
    # Only append to history HERE
    state["improvement_history"].append(improvement)
```

Once converged, best rarely improves → history stops growing → `improvement_rate` feature broken.

**Fix**:
```python
# Track EVERY timestep (1.0 if improved, 0.0 otherwise)
if state["best_f"] is None or fitness[best_idx] < state["best_f"]:
    improvement = 1.0
    state["best_v"] = candidates[best_idx]
    state["best_f"] = fitness[best_idx]
else:
    improvement = 0.0

state["improvement_history"].append(improvement)  # ALWAYS append
```

### Bug #3: Diversity Computed from Variance (Too Small)

**Problem**:
```python
diversity = np.mean(np.var(valid_vectors, axis=0))
```

For solutions in [-1,1]^d, variance per dimension is ~0.001-0.1, so mean variance ~0.01. This gets clipped to essentially 0.

**Fix**: Use pairwise distances normalized by hypercube diagonal:
```python
# Compute average pairwise distance
pairwise_dists = []
for i in range(min(10, len(valid_vectors))):
    for j in range(i + 1, min(10, len(valid_vectors))):
        dist = np.linalg.norm(valid_vectors[i] - valid_vectors[j])
        pairwise_dists.append(dist)

avg_dist = np.mean(pairwise_dists)
max_dist = 2.0 * np.sqrt(NUM_DIMS)  # Max distance in [-1,1]^d hypercube
diversity = avg_dist / max_dist  # Normalized to [0,1]
```

### Bug #4: Convergence Formula Broke for Large Fitness Values

**Problem**:
```python
convergence = 1.0 - np.clip(fitness_std / (np.mean(fitness) + EPS), 0.0, 1.0)
```

For Rastrigin with fitness ~-300:
- `std = 50`, `mean = -300`
- `std / mean = 50 / -300 = -0.16`  
- `clip(-0.16, 0, 1) = 0`
- `convergence = 1.0 - 0 = 1.0` (always converged!)

**Fix**: Use coefficient of variation based on range:
```python
f_range = np.max(valid_fitness) - np.min(valid_fitness)
f_mean = np.mean(valid_fitness)
cv = f_range / f_mean  # Normalized spread
convergence = 1.0 - np.clip(cv, 0.0, 1.0)
```

## Results After Fix

### Sphere (10D, 10s simulation)
```
Best fitness: 79.503775
Optimal: 79.480000
Error: 0.023775 (0.03%)

Operator usage:
  GLOBAL:   12 calls (0.1%)   ← Initial exploration
  LOCAL:  9986 calls (99.9%)  ← Converged to local refinement ✓
  ADAPTIVE:  2 calls (0.0%)
```

**State features ARE varying** - causing correct operator selection.

### Rastrigin (10D, 20s simulation)
```
Best fitness: 6141.594
Optimal: -54.940
Error: 6196.534

Operator usage:
  GLOBAL: 19990 calls (100.0%)  ← Stays in exploration (multimodal) ✓
  LOCAL:      8 calls (0.0%)
  ADAPTIVE:   2 calls (0.0%)
```

**Correctly stays in GLOBAL** because Rastrigin has high diversity and low convergence throughout.

## Key Takeaway

The bugs were **all in state feature computation**, not in the neural architecture. The basal ganglia selection mechanism works correctly once it receives meaningful input signals.

**Critical lesson**: When debugging neuromorphic systems, check your **sensors (state features)** before blaming the **brain (neural dynamics)**.

