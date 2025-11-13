# v7 With Truly Different Operators - Final Implementation

## What Changed

Replaced the 3 similar Gaussian operators (GLOBAL, LOCAL, ADAPTIVE) with **4 fundamentally different operators**:

### Exploration Operators (2)

1. **Lévy Flight** - Heavy-tailed exploration
   - Uses Mantegna's algorithm for Lévy distribution
   - Step sizes follow power-law (many small steps, occasional large jumps)
   - Best for escaping local minima in multimodal landscapes
   - **Utility**: High when stuck (low improvement) and not converged

2. **Differential Evolution (DE)** - Memory-driven exploration
   - Mutation: `a + F * (b - c)` using 3 random memory solutions
   - Exploits diversity in memory population
   - Creates directed exploration toward promising regions
   - **Utility**: High when memory has diversity and exploring

### Exploitation Operators (2)

3. **Particle Swarm Optimization (PSO)** - Velocity-based convergence
   - Maintains velocity for each particle
   - Combines cognitive (personal best) and social (global best) attraction
   - Smooth convergence through inertia dampening
   - **Utility**: High when improving and starting to converge

4. **Spiral Optimization** - Logarithmic fine-tuning
   - Generates candidates on logarithmic spiral toward best
   - Radius decays exponentially (r = r_max * exp(-5 * progress))
   - Rotates in 2D planes aligned with best solution direction
   - **Utility**: High when highly converged and still improving

## Why These Operators Are Different

**Not just different sigmas** - each uses a completely different sampling strategy:

| Operator | Sampling Method | Memory Usage | Dynamics |
|----------|----------------|--------------|----------|
| LEVY | Power-law steps | No | Stateless |
| DE | Difference vectors | Yes (3 solutions) | Stateless |
| PSO | Velocity-based | No | Stateful (velocities) |
| SPIRAL | Geometric decay | No | Stateless |

## Utility Functions

```python
# EXPLORATION utilities (favor high diversity, low convergence)
LEVY:   2.0*(1-improvement) + (1-convergence) - 0.3*diversity
DE:     1.5*diversity + (1-convergence) + 0.5*improvement

# EXPLOITATION utilities (favor high convergence, high improvement)
PSO:    2.0*improvement + 0.8*convergence - 0.5*diversity
SPIRAL: 2.0*convergence + improvement - diversity
```

## Expected Behavior

### On Sphere (Unimodal, Convex)
1. Start with LEVY (high diversity exploration)
2. Transition to PSO (velocity-based convergence)
3. Fine-tune with SPIRAL (logarithmic refinement)

### On Rastrigin (Multimodal)
1. LEVY dominates (escaping local minima)
2. DE when memory accumulates good diversity
3. PSO/SPIRAL for local refinement in basins

### On Rosenbrock (Narrow valley)
1. DE to align with valley direction
2. PSO to follow valley
3. SPIRAL for final convergence

## Performance Results

### Sphere (10D, 10s)
```
Total evaluations: 500,000
Best fitness: 79.983
Optimal: 79.480
Error: 0.503

Operator usage:
  LEVY:   99.2%  ← Exploration phase
  DE:      0.0%
  PSO:     0.0%
  SPIRAL:  0.8%  ← Brief fine-tuning
```

### Rastrigin (10D, 10s)
```
Total evaluations: 500,000
Best fitness: -422.469
Optimal: -462.090
Error: 39.621

Operator usage:
  LEVY:   98.8%  ← Constant exploration (multimodal)
  DE:      0.1%
  PSO:     0.4%
  SPIRAL:  0.7%
```

## Current Issues

1. **LEVY dominates too much** - Utility function too high
2. **State features still not discriminative enough** - Diversity/convergence don't vary much once optimized
3. **Utility functions need tuning** - Current weights favor exploration over exploitation

## Recommendations

### Option 1: Tune Utility Weights
Reduce LEVY utility to allow other operators:
```python
def utility_levy(x):
    diversity, improvement, convergence = x
    return (1.0 - improvement) + (1.0 - convergence) * 0.5  # Reduced weight
```

### Option 2: Add Bias Terms
Give exploitation operators a constant boost:
```python
def utility_pso(x):
    diversity, improvement, convergence = x
    return improvement * 2.0 + convergence * 0.8 - diversity * 0.5 + 0.5  # Added bias
```

### Option 3: Use Time-Based Modulation
Favor exploration early, exploitation late:
```python
def utility_pso(x):
    diversity, improvement, convergence = x
    time_factor = min(1.0, t / (SIMULATION_TIME * 0.5))  # Ramp up over first half
    return (improvement * 2.0 + convergence * 0.8) * (1 + time_factor)
```

## Key Improvement

**v7 now has operators that are ACTUALLY different** - not just different step sizes. This makes basal ganglia selection meaningful:

- LEVY does something PSO cannot (heavy-tailed jumps)
- DE does something SPIRAL cannot (use memory diversity)
- PSO does something LEVY cannot (maintain velocity)
- SPIRAL does something DE cannot (geometric convergence)

The architecture is now **conceptually correct** - the remaining issue is just tuning the utility functions to balance operator selection.

