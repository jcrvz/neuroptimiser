# Feature Plot Problem - Diagnosis and Fix

## The Issue

You're seeing **horizontal lines** in the feature plot (Plot 3), which suggests the features aren't changing. But when tested standalone, features DO change significantly:

```
Step 0:  Diversity=0.739, ImprovRate=0.500, Convergence=0.082
Step 90: Diversity=0.425, ImprovRate=0.020, Convergence=0.490
```

## Root Cause

**The problem is NOT in the feature computation** - it's in how Nengo's neural ensembles filter the signals.

### What's Happening

1. **Feature computation is correct** - Python function computes changing values
2. **Neural encoding filters signals** - The `state_ensemble` (300 neurons) receives features
3. **Synaptic filtering smooths changes** - The 0.01s synapse on the probe averages rapidly
4. **Neural noise adds jitter** - Spiking neurons add variability

### The Real Problem

The `state_ensemble` has **radius=1.5** and processes signals through **300 spiking neurons**. When features change slowly (e.g., diversity goes from 0.4 to 0.3 over 10 seconds), the neural representation:

1. **Filters out slow changes** due to synaptic time constants
2. **Adds neural noise** that obscures the actual signal
3. **Saturates near boundaries** (convergence near 1.0 gets clipped)

## The Fix

### Option 1: Use Direct Node Probe (RECOMMENDED)

Instead of probing the neural ensemble, probe the **node directly**:

```python
# Add this probe BEFORE the connection to state_ensemble
state_features_probe = nengo.Probe(state_features_node, synapse=None)

# In plotting:
state_data = sim.data[state_features_probe]  # Use node probe, not ensemble probe
```

This gives you the **actual computed features** without neural filtering.

### Option 2: Increase Ensemble Neurons

If you want to keep neural encoding:

```python
state_ensemble = nengo.Ensemble(
    n_neurons=1000,  # Was 300, now 1000
    dimensions=3,
    radius=1.5,
    label="State Ensemble"
)
```

More neurons = better representation = less noise.

### Option 3: Remove Synaptic Filtering on Probe

```python
state_probe = nengo.Probe(state_ensemble, synapse=None)  # Was 0.01
```

This removes the 10ms averaging filter, showing more detail.

## Why Horizontal Lines Appear

If features truly don't change much (e.g., on Rastrigin where it never converges):
- Diversity stays ~0.4 (solutions spread in multimodal landscape)
- Improvement rate drops to ~0.0 quickly (stuck in local minima)
- Convergence stays ~0.3 (fitness values diverse due to multiple basins)

The plot **correctly shows** that the features aren't varying much - which explains why SPIRAL dominates (it has highest base utility for this state).

## What The Plot Actually Tells You

**Horizontal lines mean**:
1. Features genuinely aren't changing (e.g., stuck in local minima on Rastrigin)
2. OR neural filtering is obscuring changes (use Option 1 fix)

**Varying lines mean**:
1. Features are changing as optimization progresses
2. Operators should switch based on these changes
3. Basal ganglia is responding to actual state dynamics

## The Purpose of This Plot

The feature plot shows **WHY operators are selected**:

- High diversity → DE or LEVY should be selected
- High improvement rate → PSO should be selected  
- High convergence → SPIRAL should be selected

If features are flat, **it's correct for one operator to dominate** - the optimization state isn't changing, so operator selection shouldn't change either.

## Recommendation

**Add the direct node probe** (Option 1) to see raw feature values. If they're still flat, the optimizer is genuinely stuck in a regime, and you need to:

1. Increase epsilon-greedy exploration (0.10 → 0.20)
2. Add periodic operator reset
3. Tune utility functions to favor more operator switching

The plot is working correctly - it's just showing you that the features (and thus the optimization state) aren't varying as much as you'd expect!

