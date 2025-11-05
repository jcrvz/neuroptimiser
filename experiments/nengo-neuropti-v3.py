#%%
import matplotlib.pyplot as plt
import nengo
from sortedcontainers import SortedList
import numpy as np
from ioh import get_problem
from nengo.utils.matplotlib import rasterplot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 1  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_DIMENSIONS  = 10  # Number of dimensions for the problem
NEURONS_PER_DIM = 50
SIMULATION_TIME = 50.0  # seconds

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMENSIONS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMENSIONS)

V_LOWER_BOUND   = -1.0 * np.ones(NUM_DIMENSIONS)
V_UPPER_BOUND   = 1.0 * np.ones(NUM_DIMENSIONS)
V_INITIAL_GUESS = tro2s(X_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
F_INITIAL_GUESS = problem(X_INITIAL_GUESS)

# Keep copies of the original global bounds
X_LOWER_BOUND0  = X_LOWER_BOUND.copy()
X_UPPER_BOUND0  = X_UPPER_BOUND.copy()

# Exploitation (search-space shrinking) schedule
SEED_VALUE      = 69
INIT_TIME       = 0.002     # seconds before starting shrink operations
DEFAULT_WAIT    = 0.10     # seconds between shrink operations
STAG_RESET      = 0.20     # seconds of stagnation before a RESET to global bounds
RESET_WAIT      = 0.10
EPS             = 1e-12   # small value to ensure numerical ordering of bounds
MIN_WIDTH       = 1e-3   # do not shrink any dimension below this absolute width proportion
ACTION_DELAY    = 0.00  # seconds to retarget the EA to the best_v after a shrink
TAU             = 0.002  # synaptic time constant for filtering inputs
# MIN_WIDTH_FRAC   = 0.02    # do not shrink any dimension below 2% of the original rang
ARCH_LENGTH     = 20

#%%

# Create the Nengo model
model = nengo.Network(label="nNeurOpti V1", seed=69)
with model:
    # INITIALISATION
    # --------------------------------------------------------------

    # LOCAL STATE
    state = {
        "lb": V_LOWER_BOUND.copy(),
        "ub": V_UPPER_BOUND.copy(),
        "best_v": V_INITIAL_GUESS.copy(),
        "best_f": F_INITIAL_GUESS,
        "last_best_f": F_INITIAL_GUESS,
        "width_proportion": 1.0,
        "last_adjust": 0.0,
        "last_reset_time": -1e9,
        "t_stag_start": 0.0,
        "archive": SortedList(key=lambda _x: _x[1]),
    }

    # Evaluate the objective function
    def f_obj(t, state_vector):
        """Evaluate the objective function at the given state vector."""
        _v      = np.clip(state_vector, -1.0, 1.0)  # Ensure within bounds, consider toroidal space later
        v       = trs2o(_v, state["lb"], state["ub"])
        x_vals  = trs2o(v, X_LOWER_BOUND, X_UPPER_BOUND)
        # Implement here the transformation for integer variables if needed
        fv      = problem(x_vals)
        return np.concatenate((v, [fv]))

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------

    # [Main] Ensemble to represent the current state vector
    ea          = nengo.networks.EnsembleArray(
        label           = "ens_array",
        n_neurons       = NEURONS_PER_DIM,
        n_ensembles     = NUM_DIMENSIONS,
        ens_dimensions  = 1,
        radius          = 1.1,
        # intercepts  = nengo.dists.Uniform(-0.9, 0.9),
        max_rates   = nengo.dists.Uniform(100,220),
        encoders    = nengo.dists.UniformHypersphere(surface=True),
        neuron_type     = nengo.AdaptiveLIF(),
        # neuron_type     = nengo.LIF(),
        seed            = SEED_VALUE,
    )

    # Pulser function to inject the initial guess at t=0 and best_v after shrinks
    def pulser(t):
        if t < INIT_TIME:
            return V_INITIAL_GUESS.copy()
        # if state["best_v"] is not None:
        #     return state["best_v"]
        return np.zeros(NUM_DIMENSIONS)

    # Pulser node to trigger actions
    pulser_node = nengo.Node(
        label="pulser",
        output=pulser,
        size_out=NUM_DIMENSIONS,
    )

    # Selector / controller node to move towards the best-so-far
    eta = 0.6  # "Learning rate"


    def hs(t, v):
        direction = state["best_v"] - v

        # Only add noise when search space is wide (early exploration)
        if state["width_proportion"] > 0.3:
            noise = 0.05 * state["width_proportion"] * np.random.randn(NUM_DIMENSIONS)
        else:
            noise = 0  # Pure exploitation when converged

        return eta * direction + noise


    hs_node     = nengo.Node(
        label       = "hs",
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = hs
    )

    # Scheduler function to trigger shrink operations
    def neuromodulator(t, best_f):
        if state["best_f"] < state["last_best_f"]:
            state["t_stag_start"] = t
            state["last_best_f"] = state["best_f"]

        stagnated = (t - state["t_stag_start"]) >= STAG_RESET
        can_reset = (t - state["last_reset_time"]) >= RESET_WAIT

        if (state["best_v"] is None) or (t < INIT_TIME) or (
                (t - state["last_adjust"] < DEFAULT_WAIT)):
            return state["width_proportion"]

        # Decide on the action based on improvement and mode
        if (stagnated and can_reset): # or state["width_proportion"] <= 1e-6:
            proportion                  = -1.0  # RESET to global bounds if stagnated
        elif stagnated:
            proportion = 1.0  # Hold while waiting for cooldown
        else:
            proportion = np.random.uniform(0.25, 0.5) # Shrink otherwise

        if (state["width_proportion"] <= MIN_WIDTH) or np.any(
                state["ub"] - state["lb"] <= MIN_WIDTH):
            proportion = -1.0  # RESET if any dimension is too small

        # Apply the action: shrink / expand the search space, or RESET to global bounds
        if proportion == -1.0: # RESET action only if in Local mode
            # RESET: restore original global bounds and width proportion
            new_lb = V_LOWER_BOUND.copy()
            new_ub = V_UPPER_BOUND.copy()
            state["width_proportion"] = 1.0

            # new_best_v = np.random.uniform(V_LOWER_BOUND, V_UPPER_BOUND, NUM_DIMENSIONS)
            # state["best_v"] = new_best_v.copy()
            # state["best_f"] = None  # Force re-evaluation after reset
            new_best_v = state["best_v"]

            state["last_reset_time"] = t
            state["last_adjust"] = t
            state["t_stag_start"] = t
        else:
            # Current widths and target widths after shrinking
            half_width  = 0.5 * np.maximum(
                (state["ub"] - state["lb"]) * proportion,  MIN_WIDTH
            )

            new_best_v = state["best_v"]

            new_lb = np.maximum(new_best_v - half_width, V_LOWER_BOUND)
            new_ub = np.minimum(new_best_v + half_width, V_UPPER_BOUND)

            new_lb = np.minimum(new_lb, new_ub - EPS)
            new_ub = np.maximum(new_ub, new_lb + EPS)

            state["width_proportion"]   *= proportion
            state["last_adjust"]        = t
            state["t_stag_start"]       = t

        state["lb"], state["ub"] = new_lb, new_ub
        state["best_v"] = np.clip(new_best_v, -1.0, 1.0)

        return state["width_proportion"]


    # Scheduler node to trigger shrink operations
    shrink_node = nengo.Node(
        label       = "neuromodulator",
        output      = neuromodulator,
        size_out    = 1,
        size_in     = 1,
    )

    # Selector node to keep track of the best-so-far
    def local_selector(t, _vf):
        v      = _vf[:NUM_DIMENSIONS]
        # v       = trs2o(_v, state["lb"], state["ub"])
        fv      = float(_vf[NUM_DIMENSIONS])

        if state["best_f"] is None or (fv < state["best_f"] and t >= INIT_TIME):
            # Register the previous best
            # state["last_best_f"] = state["best_f"]

            # Update the best-so-far
            state["best_f"] = fv
            # store best in ORIGINAL space under current dynamic bounds
            state["best_v"] = v.copy()

            # Reset stagnation counter
            state["archive"].add( (v.copy(), fv) )

            if len(state["archive"]) > ARCH_LENGTH:
                state["archive"].pop()

        if len(state["archive"]) == 0:
            return np.concatenate((v, [fv]))
        else:
            best_archive_v, best_archive_f = state["archive"][0]
            return np.concatenate((best_archive_v, [best_archive_f]))

    selector_node   = nengo.Node(
        label       = "selector",
        output      = local_selector,
        size_in     = NUM_DIMENSIONS + 1,
        size_out    = NUM_DIMENSIONS + 1,
    )

    # Objective function node
    obj_node    = nengo.Node(
        label       = "f_obj",
        output      = f_obj,
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS+1,
    )

    # CONNECTIONS
    # --------------------------------------------------------------

    # [Pulser] --- v --> [EA input]
    nengo.Connection(
        pre         = pulser_node,
        post        = ea.input,
        synapse     = TAU,
    )

    # [EA output] --- x --> [Selector]
    nengo.Connection(
        pre         = ea.output,
        post        = hs_node,
        synapse     = 0,
    )

    # [Selector] --- x (delayed) --> [EA input]
    nengo.Connection(
        pre         = hs_node,
        post        = ea.input,
        synapse     = TAU,
        transform   = np.eye(NUM_DIMENSIONS),
    )

    # [EA output] --- x --> [Objective function]
    nengo.Connection(
        pre         = ea.output,
        post        = obj_node,
        synapse     = 0,
    )

    # [Objective function] --- (x,f) --> [Selector]
    nengo.Connection(
        pre         = obj_node,
        post        = selector_node,
        synapse     = 0,
    )

    # [Selector] --- fbest --> [fbest_only]
    # nengo.Connection(
    #     pre         = selector_node,
    #     post        = fbest_only,
    #     synapse     = 0,
    # )

    # [Selector] --- f --> [shrink_scheduler]
    nengo.Connection(
        pre        = selector_node[-1],
        post       = shrink_node,
        synapse    = 0,
    )

    # [Selector] --- xbest --> [xbest_only]
    # nengo.Connection(
    #     pre         = selector_node,
    #     post        = xbest_only,
    #     synapse     = 0,
    # )

    # PROBES
    # --------------------------------------------------------------
    shrink_trigger  = nengo.Probe(shrink_node, synapse=None)
    ens_lif_val     = nengo.Probe(ea.output, synapse=0.0)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val       = nengo.Probe(selector_node[-1], synapse=0.0)
    vbest_val       = nengo.Probe(selector_node[:NUM_DIMENSIONS], synapse=0.0)
    ea_spk          = [nengo.Probe(ens.neurons, synapse=0.0) for ens in ea.ensembles]

#%%
# Create our simulator
with nengo.Simulator(model) as sim:
    sim.reset(seed=69)
    # Run it for 1 second
    sim.run(SIMULATION_TIME)

#%%
print("Evals", len(sim.data[obj_val]))
print(f"{'xid':3s}: {'lower':>8s} {'upper':>8s}, {'xbest':>8s} :: {'xopt':>7s} | {'diff':>7s}")
print("-" * 55)
for i in range(NUM_DIMENSIONS):
    print(f" x{i+1:02d}: [{np.min(sim.data[vbest_val][:,i]):.4f}, {np.max(sim.data[vbest_val][:,i]):.4f}], ", end="")
    print(f"{sim.data[vbest_val][-1][i]:7.4f} :: {problem.optimum.x[i]:7.4f} | {sim.data[vbest_val][-1][i] - problem.optimum.x[i]:7.4f}")

print(f"fbest: {sim.data[fbest_val][-1][0]:.4f} (opt: {problem.optimum.y}), diff: {sim.data[fbest_val][-1][0] - problem.optimum.y:.4f}")

#%%
v = sim.data[vbest_val][-1]
f = problem(trs2o(v, X_LOWER_BOUND, X_UPPER_BOUND))
print(f"f(xbest): {f:.4f} (opt: {problem.optimum.y}), diff: {f - problem.optimum.y:.4f}")

#%% Process actions
action_proportions = sim.data[shrink_trigger].astype(float).flatten()
action_times = np.arange(len(action_proportions)) * sim.dt

change_proportions = np.diff(action_proportions)
change_indices = np.where(change_proportions != 0)[0]

def add_colors():
    pass
    # # Indices where the action proportion changes
    # for idx in change_indices:
    #     delta = change_proportions[idx]
    #     t = action_times[idx + 1]
    #     if delta < 0:
    #         color   = "red"
    #         marker  = "v"
    #     elif delta == 0:
    #         color   = "black"
    #         marker  = "o"
    #     else:
    #         color   = "blue"
    #         marker  = "^"
    #     plt.axvline(t, color=color, alpha=0.3, linestyle="--", marker=marker)

# Plot the action proportions over time
# plt.figure(figsize=(12, 8), dpi=150)
# add_colors()
# plt.plot(sim.trange(), action_proportions, color="black")
# plt.show()

#%%
# plt.figure()
# # plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value")
# # plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, 1e0, 1e10, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
# add_colors()
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")
# plt.xlim(0, SIMULATION_TIME)
# plt.yscale("log")
# plt.legend()
# plt.title(f"Obj. func. err. vs time | fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
# plt.show()

#%%
# Plot the decoded output of the ensemble
plt.figure(figsize=(12, 8), dpi=150)
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -5, 5, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
add_colors()
plt.plot(sim.trange(), sim.data[vbest_val], label=[f"v{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, SIMULATION_TIME)
plt.legend()
plt.show()

#%%
# # Plot the decoded output of the ensemble
# plt.figure(figsize=(12, 8), dpi=150)
# # plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -1, 1, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
# add_colors()
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# # plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# # plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
# plt.xlim(0, SIMULATION_TIME)
# plt.legend()
# plt.show()

#%%
plt.figure(figsize=(12, 8), dpi=150)
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, 1e0, 1e9, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
add_colors()
plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value error")
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far error")
# plt.hlines(problem.optimum.y, 0, simulation_time, colors="r", linestyles="dashed", label="Optimal value")
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")
plt.xlim(0, SIMULATION_TIME)
plt.yscale("log")
plt.legend()
plt.title(f"{problem.meta_data.name}-{problem.meta_data.instance}ins, {problem.meta_data.n_variables}D | "
          f"fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()

#%%
# # Plot the spiking output of the ensemble
# plt.figure(figsize=(12, 8), dpi=150)
# spikes_cat = np.hstack([sim.data[p] for p in ea_spk])
# rasterplot(sim.trange(), spikes_cat) #, use_eventplot=True)
# plt.xlim(0, SIMULATION_TIME)
# plt.show()

