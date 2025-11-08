# %%
import matplotlib.pyplot as plt
import nengo
import numpy as np
from ioh import get_problem

from neuroptimiser.utils import tro2s, trs2o

# %%

PROBLEM_ID = 10
PROBLEM_INS = 1
NUM_OBJS = 1
NUM_DIMS = 10

NEURONS_PER_DIM = 50 * NUM_DIMS
NEURONS_PER_ENS = 100
SIMULATION_TIME = 10.0

# KEY CHANGE: Population size
LAMBDA = 50  # Generate 50 candidates per timestep
MU = max(4, LAMBDA // 2)

problem = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMS)
problem.reset()
print(problem)

EPS = 1e-12
X_LOWER_BOUND = problem.bounds.lb
X_UPPER_BOUND = problem.bounds.ub

V_INITIAL_GUESS = np.random.uniform(-1.0, 1.0, NUM_DIMS)
X_INITIAL_GUESS = trs2o(V_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
F_INITIAL_GUESS = problem(X_INITIAL_GUESS)

F_DEFAULT_WORST = 10 ** np.ceil(np.log10(F_INITIAL_GUESS) + EPS)


def eval_obj_func(v_scaled):
    x_orig = trs2o(v_scaled, X_LOWER_BOUND, X_UPPER_BOUND)
    f_val = problem(x_orig)
    return f_val


TAU_IMPROV_FAST = 0.010
TAU_IMPROV_SLOW = 0.050
INIT_TIME = 0.001
K_IMPROV = 2.0

SIG_MAX = 0.5
SIG_MIN = 1e-6
EXPLOIT_WIN = max(2.0, SIMULATION_TIME / 8)

# %%

model = nengo.Network(label="nNeurOpti Parallel Evaluation", seed=69)
with model:
    # STATE
    state = {
        "lb": X_LOWER_BOUND.copy(),
        "ub": X_UPPER_BOUND.copy(),
        "best_v": None,
        "best_f": None,
        "mode": "explore",
        "memory_vectors": np.zeros((MU, NUM_DIMS)),
        "memory_fitness": np.full(MU, F_DEFAULT_WORST),
        "memory_age": np.zeros(MU),
        # NEW: Store population statistics per timestep
        "population_mean_f": None,
    }

    # MEMORY NETWORK (unchanged)
    memory_net = nengo.networks.EnsembleArray(
        n_neurons=100,
        n_ensembles=MU,
        ens_dimensions=NUM_DIMS + 1,
        radius=2.0,
        label="Memory Network"
    )

    # VARIANCE TRACKER (unchanged)
    variance_tracker = nengo.networks.EnsembleArray(
        n_neurons=50,
        n_ensembles=NUM_DIMS,
        radius=1.0,
        label="Variance Tracker"
    )


    def variance_input_func(t, memory_flat):
        if t < INIT_TIME:
            return 0.25 * np.ones(NUM_DIMS)

        memory = memory_flat.reshape((MU, NUM_DIMS + 1))
        vectors = memory[:, :NUM_DIMS]
        per_dim_std = np.std(vectors, axis=0)
        base_sigma = np.clip(per_dim_std * 2.0, 0.05, 0.5)
        return base_sigma


    variance_input_node = nengo.Node(
        variance_input_func,
        size_in=MU * (NUM_DIMS + 1),
        size_out=NUM_DIMS,
        label="Variance Input"
    )

    # MODE CONTROLLER (unchanged)
    mode_integrator = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        radius=2.0,
        label="Mode Integrator"
    )

    nengo.Connection(
        mode_integrator,
        mode_integrator,
        synapse=0.1,
        transform=0.95
    )


    def mode_decision_func(t, integrated_improv):
        mode_signal = 1.0 if integrated_improv > 0.7 else 0.0
        state["mode"] = "exploit" if mode_signal > 0.5 else "explore"
        return mode_signal


    mode_decision_node = nengo.Node(
        mode_decision_func,
        size_in=1,
        size_out=1,
        label="Mode Decision"
    )

    nengo.Connection(
        mode_integrator,
        mode_decision_node,
        synapse=None
    )

    # CENTRE COMPUTER (unchanged)
    centre_computer = nengo.networks.EnsembleArray(
        n_neurons=100,
        n_ensembles=NUM_DIMS,
        radius=2.0,
        label="Centre Computer"
    )


    def centre_from_memory_func(t, memory_flat):
        if t < INIT_TIME:
            return V_INITIAL_GUESS.copy()

        memory = memory_flat.reshape((MU, NUM_DIMS + 1))
        vectors = memory[:, :NUM_DIMS]
        fitness = memory[:, NUM_DIMS]

        valid_mask = fitness < F_DEFAULT_WORST
        if not np.any(valid_mask):
            return V_INITIAL_GUESS.copy()

        valid_fitness = fitness[valid_mask]
        valid_vectors = vectors[valid_mask]

        ranks = np.argsort(np.argsort(valid_fitness))  # 0=best, N-1=worst
        weights = 1.0 / (ranks + 1.0)  # Harmonic weighting
        weights /= np.sum(weights)

        centroid = np.average(valid_vectors, axis=0, weights=weights)
        return np.clip(centroid, -1.0, 1.0)


    centre_from_memory_node = nengo.Node(
        centre_from_memory_func,
        size_in=MU * (NUM_DIMS + 1),
        size_out=NUM_DIMS,
        label="Centre from Memory"
    )

    nengo.Connection(
        centre_from_memory_node,
        centre_computer.input,
        synapse=0.05
    )


    # SIGMA MODULATOR (unchanged)
    def sigma_modulator_func(t, inputs):
        mode_signal = inputs[0]
        base_sigma = inputs[1:]
        scale = 0.2 if mode_signal > 0.5 else 1.0
        return base_sigma * scale


    sigma_modulator = nengo.Node(
        sigma_modulator_func,
        size_in=1 + NUM_DIMS,
        size_out=NUM_DIMS,
        label="Sigma Modulator"
    )

    nengo.Connection(mode_decision_node, sigma_modulator[0], synapse=None)
    nengo.Connection(variance_tracker.output, sigma_modulator[1:], synapse=None)


    # ==================================================================
    # KEY CHANGE: PARALLEL POPULATION GENERATION
    # ==================================================================
    def population_generator_func(t, inputs):
        """Generate LAMBDA candidates simultaneously"""
        if t < INIT_TIME:
            return np.zeros(LAMBDA * (NUM_DIMS + 1))

        centre = inputs[:NUM_DIMS]
        sigma = inputs[NUM_DIMS:2 * NUM_DIMS]

        # Generate LAMBDA candidates at once
        noise = np.random.randn(LAMBDA, NUM_DIMS)
        candidates = centre + sigma * noise  # Broadcasting
        candidates = np.clip(candidates, -1.0, 1.0)

        # Evaluate all candidates
        fitness = np.array([eval_obj_func(v) for v in candidates])

        # Update memory with ALL candidates (competitive update)
        for i in range(LAMBDA):
            worst_idx = np.argmax(state["memory_fitness"])
            if fitness[i] < state["memory_fitness"][worst_idx]:
                state["memory_vectors"][worst_idx] = candidates[i]
                state["memory_fitness"][worst_idx] = fitness[i]
                state["memory_age"][worst_idx] = 0.0

        state["memory_age"] += 1.0

        # Update best-so-far from population
        best_pop_idx = np.argmin(fitness)
        if state["best_f"] is None or fitness[best_pop_idx] < state["best_f"]:
            state["best_v"] = candidates[best_pop_idx]
            state["best_f"] = fitness[best_pop_idx]

        # Store population mean fitness for improvement signal
        state["population_mean_f"] = np.mean(fitness)

        # Return memory state (for connections)
        output = np.zeros((MU, NUM_DIMS + 1))
        for i in range(MU):
            output[i, :NUM_DIMS] = state["memory_vectors"][i]
            output[i, NUM_DIMS] = state["memory_fitness"][i]

        return output.flatten()


    population_node = nengo.Node(
        population_generator_func,
        size_in=2 * NUM_DIMS,
        size_out=MU * (NUM_DIMS + 1),
        label="Population Generator"
    )

    # Connect centre and sigma to population generator
    nengo.Connection(
        centre_computer.output,
        population_node[:NUM_DIMS],
        synapse=None
    )

    nengo.Connection(
        sigma_modulator,
        population_node[NUM_DIMS:],
        synapse=None
    )

    # Connect population output to memory network (for visualization)
    nengo.Connection(
        population_node,
        memory_net.input,
        synapse=None
    )

    # Connect memory to variance tracker
    nengo.Connection(
        population_node,
        variance_input_node,
        synapse=None
    )

    nengo.Connection(
        variance_input_node,
        variance_tracker.input,
        synapse=0.1
    )

    # Connect memory to centre computer
    nengo.Connection(
        population_node,
        centre_from_memory_node,
        synapse=None
    )


    # ==================================================================
    # IMPROVEMENT SIGNAL (modified to use population mean)
    # ==================================================================
    def features_func(t):
        """Compute improvement from population mean vs best-so-far"""
        if t < INIT_TIME or state["population_mean_f"] is None:
            return 0.0

        # Fast track: current population mean
        # Slow track: best-so-far (already smoothed by memory update rate)
        improvement = state["best_f"] - state["population_mean_f"]
        return np.clip(improvement * K_IMPROV, 0.0, 1.0)


    features_input_node = nengo.Node(
        features_func,
        label="Features Gate"
    )

    nengo.Connection(
        features_input_node,
        mode_integrator,
        synapse=None,
        transform=2.0
    )


    # ==================================================================
    # PROBES (modified to track population statistics)
    # ==================================================================
    def best_f_probe(t):
        return state["best_f"] if state["best_f"] is not None else F_DEFAULT_WORST


    def best_v_probe(t):
        return state["best_v"] if state["best_v"] is not None else V_INITIAL_GUESS


    def pop_mean_f_probe(t):
        return state["population_mean_f"] if state["population_mean_f"] is not None else F_DEFAULT_WORST


    best_f_node = nengo.Node(best_f_probe, label="Best F Probe")
    best_v_node = nengo.Node(best_v_probe, label="Best V Probe")
    pop_mean_f_node = nengo.Node(pop_mean_f_probe, label="Pop Mean F Probe")


    def mode_probe_func(t):
        return 1.0 if state["mode"] == "exploit" else 0.0


    mode_probe_node = nengo.Node(mode_probe_func, label="Mode Probe")

    fbest_val = nengo.Probe(best_f_node, synapse=0.0)
    vbest_val = nengo.Probe(best_v_node, synapse=0.0)
    pop_mean_val = nengo.Probe(pop_mean_f_node, synapse=0.0)
    centre_val = nengo.Probe(centre_computer.output, synapse=0.0)
    sigma_val = nengo.Probe(sigma_modulator, synapse=0.0)
    features_val = nengo.Probe(features_input_node, synapse=0.0)
    mode_val = nengo.Probe(mode_probe_node, synapse=0.0)

# %%
with nengo.Simulator(model) as sim:
    sim.reset(seed=69)
    sim.run(SIMULATION_TIME)

# %%
total_evals = len(sim.data[fbest_val]) * LAMBDA
print(f"Total evaluations: {total_evals} ({LAMBDA} per timestep)")
print(f"{'xid':3s}: {'xbest':>8s} :: {'xopt':>7s} | {'diff':>7s}")
print("-" * 45)

vbest_values = sim.data[vbest_val]
xbest = np.array([trs2o(vbest_values[i, :], X_LOWER_BOUND, X_UPPER_BOUND) for i in range(vbest_values.shape[0])])

for i in range(NUM_DIMS):
    print(
        f" x{i + 1:02d}: {xbest[-1, i]:7.4f} :: {problem.optimum.x[i]:7.4f} | {xbest[-1, i] - problem.optimum.x[i]:7.4f}")

print(
    f"fbest: {sim.data[fbest_val][-1][0]:.4f} (opt: {problem.optimum.y}), diff: {sim.data[fbest_val][-1][0] - problem.optimum.y:.4f}")

# %%
plt.figure(figsize=(12, 8), dpi=150)
plt.plot(sim.trange(), sim.data[pop_mean_val] - problem.optimum.y, label="Population mean error", alpha=0.5,
         linewidth=0.5)
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far error", linewidth=2)

ax1 = plt.gca()
plt.xlim(0, SIMULATION_TIME)
plt.xlabel("Time, s")
plt.ylabel("Objective value error")
plt.yscale("log")
plt.legend(loc="lower left")

ax2 = ax1.twinx()

mode_signal = sim.data[mode_val][:, 0]
exploit_mask = mode_signal > 0.5
for i in range(len(exploit_mask) - 1):
    if exploit_mask[i]:
        ax2.axvspan(sim.trange()[i], sim.trange()[i + 1], color="yellow", alpha=0.1)

ax2.plot(sim.trange(), mode_signal, "y:", linewidth=2.0, alpha=0.3, label="Exploit mode")
ax2.plot(sim.trange(), sim.data[features_val], color="magenta", marker=".", markersize=1, linestyle="dashed", alpha=0.5,
         label="Improvement signal")

ax2.set_ylabel("Improvement rate")
ax2.set_yscale("linear")
ax2.legend(loc="upper right")

plt.title(
    f"{problem.meta_data.name}, {NUM_DIMS}D | Total evals: {total_evals} | fbest: {sim.data[fbest_val][-1][0]:.2f}")
plt.show()
