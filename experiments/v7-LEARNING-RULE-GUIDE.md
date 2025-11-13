# Adaptive Utility Learning in v7

## Implementation

I've added a **reward-based learning rule** that adapts operator selection weights online based on performance.

### Learning Rule (Lines 551-568)

```python
# After each operator execution:
if last_operator improved best_f:
    reward = (improvement_amount) / (current_best_f)
    weight[operator] += learning_rate * reward  # Increase weight
    weight[operator] = clip(weight, 0.1, 5.0)
else:
    weight[operator] -= 0.01  # Small penalty
    weight[operator] = max(0.1, weight)
```

**Key parameters**:
- `learning_rate = 0.1` - How fast weights adapt
- Weight range: `[0.1, 5.0]` - Prevents complete suppression or explosion
- Penalty: `0.01` - Small constant penalty for no improvement

### How It Works

1. **Track last operator**: Remember which operator was just used
2. **Measure reward**: Did it improve `best_f`?
3. **Update weight**: Increase if improved, decrease if not
4. **Modulate utility**: `utility = base_utility * learned_weight`

### Current Results

**Sphere (15s simulation)**:
```
Operator usage:
  LEVY:    0.0%  [weight: 1.016]  ← Explored early, then penalized
  DE:      0.0%  [weight: 1.002]  
  PSO:     0.5%  [weight: 0.372]  ← Some use during convergence
  SPIRAL: 99.5%  [weight: 0.100]  ← Dominated despite minimum weight!
```

## The Problem: Convergence Trap

Once the system converges:
- `convergence = 1.0` (fitness values similar)
- `improvement_rate = 0.0` (no more improvements)
- `diversity = 0.02` (solutions clustered)

**SPIRAL has high base utility** when converged:
```python
base = convergence * 0.8 + improvement * 0.4 + 0.1
     = 1.0 * 0.8 + 0.0 * 0.4 + 0.1
     = 0.9  (high!)
```

Even with weight=0.1, `utility = 0.9 * 0.1 = 0.09` beats other operators.

## Solutions

### Option 1: Add Exploration Bonus

Force periodic exploration regardless of state:

```python
def utility_levy(x):
    diversity, improvement, convergence = x
    base = (1.0 - improvement) * 0.5 + (1.0 - convergence) * 0.3
    
    # Add time-based exploration bonus
    time_since_last_levy = state["operator_counts"]["LEVY"] / sum(state["operator_counts"].values())
    exploration_bonus = 0.5 if time_since_last_levy < 0.01 else 0.0  # Force exploration every 100 timesteps
    
    return (base + exploration_bonus) * state["utility_weights"]["LEVY"]
```

### Option 2: Epsilon-Greedy Selection

Add random exploration to basal ganglia:

```python
def population_generator_func(t, operator_selection):
    epsilon = 0.1  # 10% random selection
    
    if np.random.rand() < epsilon:
        # Random operator selection
        operator_idx = np.random.randint(n_operators)
    else:
        # Use basal ganglia selection
        operator_idx = np.argmax(operator_selection)
    
    operator_name = list(OPERATORS.keys())[operator_idx]
    # ...rest of code...
```

### Option 3: Adaptive Learning Rate

Increase learning rate when stagnating:

```python
# Detect stagnation
if len(state["improvement_history"]) >= 50:
    recent_improvements = sum(state["improvement_history"][-50:])
    if recent_improvements < 2:  # Less than 2 improvements in 50 steps
        learning_rate = 0.3  # Higher learning rate to escape
    else:
        learning_rate = 0.1  # Normal
```

### Option 4: Multi-Armed Bandit (UCB)

Use Upper Confidence Bound instead of pure weights:

```python
def utility_with_ucb(base_utility, operator_name):
    n_total = sum(state["operator_counts"].values())
    n_operator = state["operator_counts"][operator_name]
    
    # UCB bonus: explore operators used less
    if n_operator > 0:
        exploration_bonus = np.sqrt(2 * np.log(n_total) / n_operator)
    else:
        exploration_bonus = float('inf')  # Force trying unused operators
    
    return base_utility * state["utility_weights"][operator_name] + exploration_bonus
```

### Option 5: Reset Weights Periodically

Prevent premature convergence to single operator:

```python
# In population_generator_func, every 1000 timesteps:
if state["total_evals"] % (1000 * LAMBDA) == 0:
    # Reset all weights to 1.0
    for op in state["utility_weights"]:
        state["utility_weights"][op] = 1.0
```

## Recommended Configuration

**For best results, combine:**

1. **Rebalanced base utilities** (current)
2. **Epsilon-greedy** (Option 2) with ε=0.05
3. **UCB exploration bonus** (Option 4)

```python
def population_generator_func(t, operator_selection):
    # Epsilon-greedy
    epsilon = 0.05
    
    if np.random.rand() < epsilon:
        operator_idx = np.random.randint(n_operators)
    else:
        # Add UCB to selection
        utilities_with_ucb = []
        for i, op_name in enumerate(OPERATORS.keys()):
            n_total = sum(state["operator_counts"].values()) + 1
            n_op = state["operator_counts"][op_name] + 1
            ucb_bonus = np.sqrt(2 * np.log(n_total) / n_op)
            utilities_with_ucb.append(operator_selection[i] + 0.1 * ucb_bonus)
        
        operator_idx = np.argmax(utilities_with_ucb)
    
    operator_name = list(OPERATORS.keys())[operator_idx]
    # ...rest of learning rule...
```

## Tuning Guide

### If LEVY dominates:
- Reduce `base = (1.0 - improvement) * 0.5` to `0.3`
- Increase penalty: `0.01` → `0.02`

### If SPIRAL dominates:
- Reduce convergence weight: `0.8` → `0.5`
- Add epsilon-greedy (Option 2)

### If operators oscillate too much:
- Reduce learning rate: `0.1` → `0.05`
- Increase penalty: `0.01` → `0.005`

### If no operator gets selected:
- Increase learning rate: `0.1` → `0.2`
- Add exploration bonus (Option 1)

## Key Insight

The learning rule **works correctly** - weights adapt based on performance. The issue is that **base utility formulas create basins of attraction** for certain operators in certain states.

**Solution**: Don't just learn weights - also add exploration mechanisms (epsilon-greedy, UCB) to prevent convergence to suboptimal operator.

This mirrors **biological learning**: the brain doesn't just strengthen successful pathways, it also maintains exploratory behavior to discover better solutions.

