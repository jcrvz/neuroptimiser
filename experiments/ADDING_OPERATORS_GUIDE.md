# Adding Custom Operators to v7

## Overview

The v7 architecture is designed to be **easily extensible** with new optimization operators. This guide explains how to add operators step-by-step with concrete examples.

## Anatomy of an Operator

### Base Class Structure

```python
class Operator:
    """Base class for optimization operators"""
    
    def __init__(self, name, sigma):
        self.name = name
        self.sigma = sigma
    
    def generate_population(self, centre, sigma_override=None):
        """Generate LAMBDA candidates around centre"""
        sigma = sigma_override if sigma_override is not None else self.sigma
        if isinstance(sigma, (int, float)):
            sigma = sigma * np.ones(NUM_DIMS)
        
        noise = np.random.randn(LAMBDA, NUM_DIMS)
        candidates = centre + sigma * noise
        return np.clip(candidates, -1.0, 1.0)
```

**Key methods**:
- `__init__`: Initialize operator with name and sigma
- `generate_population`: Create LAMBDA candidates from centre

## Example 1: Cauchy Mutation Operator

Heavy-tailed distribution for escaping local minima:

```python
class CauchyMutation(Operator):
    """Heavy-tailed exploration using Cauchy distribution"""
    
    def __init__(self):
        super().__init__("CAUCHY", sigma=0.3)
    
    def generate_population(self, centre, sigma_override=None):
        sigma = sigma_override if sigma_override is not None else self.sigma
        if isinstance(sigma, (int, float)):
            sigma = sigma * np.ones(NUM_DIMS)
        
        # Cauchy distribution (heavy tails)
        noise = np.random.standard_cauchy((LAMBDA, NUM_DIMS))
        # Clip to prevent extreme outliers
        noise = np.clip(noise, -10, 10)
        
        candidates = centre + sigma * noise
        return np.clip(candidates, -1.0, 1.0)

# Register operator
OPERATORS["CAUCHY"] = CauchyMutation()
```

### Utility Function for Cauchy

```python
def utility_cauchy(x):
    """Use Cauchy when stuck in local minimum"""
    diversity, improvement, convergence = x
    # High utility when converged but not improving
    return convergence * (1.0 - improvement) + 0.3
```

## Example 2: Differential Evolution Operator

Uses difference vectors from memory:

```python
class DifferentialEvolution(Operator):
    """DE/rand/1 mutation strategy"""
    
    def __init__(self):
        super().__init__("DE", sigma=0.8)  # F parameter
    
    def generate_population(self, centre, sigma_override=None):
        F = sigma_override if sigma_override is not None else self.sigma
        
        # Need at least 3 solutions in memory
        valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
        valid_vectors = state["memory_vectors"][valid_mask]
        
        if len(valid_vectors) < 3:
            # Fallback to Gaussian
            return super().generate_population(centre, 0.5)
        
        candidates = []
        for _ in range(LAMBDA):
            # Select 3 random distinct solutions
            idx = np.random.choice(len(valid_vectors), 3, replace=False)
            a, b, c = valid_vectors[idx]
            
            # DE mutation: a + F * (b - c)
            mutant = a + F * (b - c)
            candidates.append(np.clip(mutant, -1.0, 1.0))
        
        return np.array(candidates)

OPERATORS["DE"] = DifferentialEvolution()
```

### Utility Function for DE

```python
def utility_de(x):
    """Use DE when memory has good diversity"""
    diversity, improvement, convergence = x
    # High utility with medium diversity and convergence
    return diversity * convergence + 0.2
```

## Example 3: Lévy Flight Operator

Long jumps with occasional short steps:

```python
class LevyFlight(Operator):
    """Lévy flight for multimodal landscapes"""
    
    def __init__(self, alpha=1.5):
        super().__init__("LEVY", sigma=1.0)
        self.alpha = alpha  # Lévy exponent
    
    def generate_population(self, centre, sigma_override=None):
        sigma = sigma_override if sigma_override is not None else self.sigma
        
        # Generate Lévy-distributed steps
        # Using Mantegna's algorithm
        beta = self.alpha
        sigma_u = (
            np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))
        ) ** (1 / beta)
        
        candidates = []
        for _ in range(LAMBDA):
            u = np.random.normal(0, sigma_u, NUM_DIMS)
            v = np.random.normal(0, 1, NUM_DIMS)
            step = u / (np.abs(v) ** (1 / beta))
            
            candidate = centre + sigma * step
            candidates.append(np.clip(candidate, -1.0, 1.0))
        
        return np.array(candidates)

OPERATORS["LEVY"] = LevyFlight()
```

### Utility Function for Lévy

```python
def utility_levy(x):
    """Use Lévy when exploration needed but not too converged"""
    diversity, improvement, convergence = x
    return (1.0 - convergence) * 0.7 + diversity * 0.3
```

## Example 4: CMA-ES Inspired Operator

Covariance matrix adaptation:

```python
class CMAInspired(Operator):
    """Simplified CMA-ES using memory covariance"""
    
    def __init__(self):
        super().__init__("CMA", sigma=None)
        self.C = None  # Covariance matrix
    
    def generate_population(self, centre, sigma_override=None):
        # Estimate covariance from memory
        valid_mask = state["memory_fitness"] < F_DEFAULT_WORST
        valid_vectors = state["memory_vectors"][valid_mask]
        
        if len(valid_vectors) < 5:
            # Not enough data, use isotropic
            return super().generate_population(centre, 0.3)
        
        # Compute empirical covariance
        centered = valid_vectors - np.mean(valid_vectors, axis=0)
        C = (centered.T @ centered) / (len(valid_vectors) - 1)
        
        # Regularize
        C += 0.01 * np.eye(NUM_DIMS)
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Fallback if not positive definite
            return super().generate_population(centre, 0.3)
        
        # Sample from N(centre, C)
        noise = np.random.randn(LAMBDA, NUM_DIMS)
        candidates = centre + (noise @ L.T)
        
        return np.clip(candidates, -1.0, 1.0)

OPERATORS["CMA"] = CMAInspired()
```

### Utility Function for CMA

```python
def utility_cma(x):
    """Use CMA when anisotropy detected"""
    diversity, improvement, convergence = x
    # High when medium convergence with improvement
    return (1.0 - np.abs(convergence - 0.5)) + improvement
```

## Integration Steps

### Step 1: Add Operator to OPERATORS Dictionary

```python
# After defining all operator classes (around line 150)
OPERATORS = {
    "GLOBAL": GlobalExploration(),
    "LOCAL": LocalRefinement(),
    "ADAPTIVE": AdaptiveSearch(),
    "CAUCHY": CauchyMutation(),      # NEW
    "DE": DifferentialEvolution(),    # NEW
    "LEVY": LevyFlight(),             # NEW
    "CMA": CMAInspired(),             # NEW
}
```

### Step 2: Update Neuron Count

```python
# Update n_operators variable (around line 270)
n_operators = 7  # Was 3, now 7 with new operators
```

### Step 3: Add Utility Ensembles

```python
# In model definition, after existing utility ensembles (around line 290)

utility_cauchy_ens = nengo.Ensemble(
    n_neurons=NEURONS_BG,
    dimensions=1,
    radius=3.0,
    label="Utility CAUCHY"
)

utility_de_ens = nengo.Ensemble(
    n_neurons=NEURONS_BG,
    dimensions=1,
    radius=3.0,
    label="Utility DE"
)

utility_levy_ens = nengo.Ensemble(
    n_neurons=NEURONS_BG,
    dimensions=1,
    radius=3.0,
    label="Utility LEVY"
)

utility_cma_ens = nengo.Ensemble(
    n_neurons=NEURONS_BG,
    dimensions=1,
    radius=3.0,
    label="Utility CMA"
)
```

### Step 4: Connect to State Ensemble

```python
# After existing state → utility connections (around line 320)

nengo.Connection(state_ensemble, utility_cauchy_ens, 
                 function=utility_cauchy, synapse=0.01)
nengo.Connection(state_ensemble, utility_de_ens, 
                 function=utility_de, synapse=0.01)
nengo.Connection(state_ensemble, utility_levy_ens, 
                 function=utility_levy, synapse=0.01)
nengo.Connection(state_ensemble, utility_cma_ens, 
                 function=utility_cma, synapse=0.01)
```

### Step 5: Connect to Basal Ganglia

```python
# After existing utility → BG connections (around line 330)

nengo.Connection(utility_cauchy_ens, bg.input[3], synapse=None)
nengo.Connection(utility_de_ens, bg.input[4], synapse=None)
nengo.Connection(utility_levy_ens, bg.input[5], synapse=None)
nengo.Connection(utility_cma_ens, bg.input[6], synapse=None)
```

### Step 6: Update State Tracking

```python
# In state dictionary initialization (around line 70)
state = {
    # ...existing fields...
    "operator_counts": {
        "GLOBAL": 0, 
        "LOCAL": 0, 
        "ADAPTIVE": 0,
        "CAUCHY": 0,    # NEW
        "DE": 0,        # NEW
        "LEVY": 0,      # NEW
        "CMA": 0,       # NEW
    },
}
```

## Designing Good Utility Functions

### General Principles

1. **Output range**: Should be roughly [-2, 2] for balanced competition
2. **State dependence**: Use all three state features when relevant
3. **Interpretability**: Should align with operator's intended use
4. **Uniqueness**: Each operator should have distinct activation conditions

### Template

```python
def utility_myoperator(x):
    """
    Describe when this operator should be selected
    
    Args:
        x: [diversity, improvement_rate, convergence]
        
    Returns:
        utility: Higher values = more likely to be selected
    """
    diversity, improvement, convergence = x
    
    # Example: High utility when [CONDITION]
    utility = (
        WEIGHT1 * FEATURE1 +
        WEIGHT2 * FEATURE2 +
        WEIGHT3 * FEATURE3 +
        BIAS
    )
    
    return utility
```

### Testing Utility Functions

```python
# Test on synthetic state values
test_states = [
    [0.1, 0.1, 0.9],  # Low diversity, low improvement, high convergence
    [0.9, 0.9, 0.1],  # High diversity, high improvement, low convergence
    [0.5, 0.5, 0.5],  # Medium everything
]

for state in test_states:
    print(f"State: {state}")
    print(f"  GLOBAL:   {utility_global(state):.2f}")
    print(f"  LOCAL:    {utility_local(state):.2f}")
    print(f"  ADAPTIVE: {utility_adaptive(state):.2f}")
    print(f"  MYOP:     {utility_myoperator(state):.2f}")
    print()
```

## Troubleshooting

### Operator Never Selected

**Symptom**: `operator_counts[MYOP] = 0` after full run

**Possible causes**:
1. Utility function too low compared to others
2. State features never reach activation range
3. Basal ganglia connection missing

**Fix**:
- Add bias term to utility: `return base_utility + 0.5`
- Check utility values during simulation with probe
- Verify connection to `bg.input[i]`

### Operator Always Selected

**Symptom**: `operator_counts[MYOP] = 99.9%`

**Possible causes**:
1. Utility function too high
2. Other operators have bugs

**Fix**:
- Reduce weights in utility function
- Add competitive term: `- 0.5 * convergence` etc.

### Poor Performance

**Symptom**: Optimization doesn't converge well

**Possible causes**:
1. Operator's sampling strategy doesn't match utility conditions
2. Sigma too large/small for intended use case

**Fix**:
- Align utility with operator behavior
- Tune sigma parameter empirically
- Test operator in isolation first

## Best Practices

1. **Start simple**: Test new operator in isolation before integrating
2. **Log everything**: Add probes to track operator selection
3. **Visualize utilities**: Plot utility values over time
4. **Test on known functions**: Verify on Sphere/Rastrigin first
5. **Document assumptions**: Explain when operator should activate

## Complete Example: Adding Simplex Operator

```python
# 1. Define operator
class SimplexSearch(Operator):
    """Nelder-Mead inspired local search"""
    
    def __init__(self):
        super().__init__("SIMPLEX", sigma=0.1)
    
    def generate_population(self, centre, sigma_override=None):
        sigma = sigma_override if sigma_override is not None else self.sigma
        
        # Generate simplex vertices around centre
        candidates = [centre]  # Include centre
        
        for i in range(NUM_DIMS):
            # Coordinate-wise perturbations
            vertex = centre.copy()
            vertex[i] += sigma
            candidates.append(np.clip(vertex, -1.0, 1.0))
            
            vertex = centre.copy()
            vertex[i] -= sigma
            candidates.append(np.clip(vertex, -1.0, 1.0))
        
        # Fill remaining with random
        while len(candidates) < LAMBDA:
            noise = np.random.randn(NUM_DIMS)
            candidate = centre + sigma * noise
            candidates.append(np.clip(candidate, -1.0, 1.0))
        
        return np.array(candidates[:LAMBDA])

# 2. Define utility
def utility_simplex(x):
    """Use simplex when highly converged and need precise local search"""
    diversity, improvement, convergence = x
    return convergence * 2.0 - diversity + improvement * 0.5

# 3. Register
OPERATORS["SIMPLEX"] = SimplexSearch()

# 4. Add to model (in appropriate sections)
n_operators = 4  # Update count

utility_simplex_ens = nengo.Ensemble(
    n_neurons=NEURONS_BG,
    dimensions=1,
    radius=3.0,
    label="Utility SIMPLEX"
)

nengo.Connection(state_ensemble, utility_simplex_ens, 
                 function=utility_simplex, synapse=0.01)
nengo.Connection(utility_simplex_ens, bg.input[3], synapse=None)

# 5. Update state
state["operator_counts"]["SIMPLEX"] = 0
```

Done! Your new operator is integrated and will compete with others via basal ganglia selection.

