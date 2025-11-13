# v7 is fundamentally flawed for continuous optimization

## The Core Problem

**Basal ganglia operator selection requires operators to do DIFFERENT things.**

Current v7:
- GLOBAL: sample with sigma=0.5
- LOCAL: sample with sigma=0.05  
- ADAPTIVE: sample with per-dim sigma

These are **all the same strategy** (Gaussian sampling) with different step sizes. Once converged, all three produce nearly identical behavior because:

1. Solutions cluster near optimum → diversity low
2. Best rarely improves → improvement rate = 0
3. Fitness values similar → convergence = 1

**Result**: Features don't discriminate between operators meaningfully.

## What ACTUALLY Works for Continuous Optimization

**Option 1: Abolish basal ganglia, use simple schedule**
```python
if t < 5.0:
    sigma = 0.5  # Exploration
elif best_improving:
    sigma = 0.05  # Exploitation
else:
    sigma = 0.5  # Restart exploration
```

**Option 2: Use TRULY different operators**
- **Random search**: Uniform sampling in full space
- **Local gradient**: Finite difference gradient descent
- **Evolutionary**: Maintain population, use recombination
- **Nelder-Mead**: Simplex-based local search

These do **qualitatively different** things, so basal ganglia makes sense.

## Recommendation

**For continuous optimization, v6 was better** - it had:
- Simple sigma adaptation based on improvement
- Mode switching (explore/exploit)
- Per-dimension variance tracking

v7 adds complexity (basal ganglia, multiple ensembles) **without benefit** because the operators aren't sufficiently different.

**If you want v7 to work**, you MUST:
1. Implement truly different operators (not just different sigmas)
2. OR abandon basal ganglia and use simpler sigma scheduling
3. OR redesign for a problem class where discrete operator choice matters (e.g., constrained optimization, multi-objective)

The current v7 is **overengineered for the wrong problem**.

