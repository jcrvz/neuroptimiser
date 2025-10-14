#%%
import matplotlib.pyplot as plt
# from collections import deque
import numpy as np

import nengo
from nengo.utils.ensemble import sorted_neurons
from nengo.utils.matplotlib import rasterplot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

from ioh import get_problem

#%%

PROBLEM_ID      = 3  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_DIMENSIONS  = 10  # Number of dimensions for the problem
NEURONS_PER_DIM = 100
SIMULATION_TIME = 50.0  # seconds

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMENSIONS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMENSIONS)

# Keep copies of the original global bounds
X_LOWER_BOUND0  = X_LOWER_BOUND.copy()
X_UPPER_BOUND0  = X_UPPER_BOUND.copy()

# Exploitation (search-space shrinking) schedule
SEED_VALUE      = 69
SHRINK_EVERY    = 1.0     # seconds between shrink operations

# Action space: proportion of current range to keep after a shrink
ACTIONS         = [-1.0, 0.25, 0.5, 0.75, 1.0]  # -1.0 = RESET to global bounds
q_eps_init      = 0.2
q_eps_min       = 0.01
q_eps_decay     = 0.05
q_alpha         = 0.1
q_gamma         = 0.95

EPS             = 1e-12   # small value to ensure numerical ordering of bounds
MIN_WIDTH       = 1e-12   # do not shrink any dimension below this absolute width
ACTION_DELAY    = 0.01  # seconds to retarget the EA to the best_v after a shrink
# MIN_WIDTH_FRAC   = 0.02    # do not shrink any dimension below 2% of the original rang

#%%

# Create the Nengo model
model = nengo.Network(label="nNeurOpti V1", seed=69)
with (model):
    # INITIALISATION
    # --------------------------------------------------------------
    v0_state    = tro2s(X_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
    # Input nodes for the bounds and initial guess
    # input_x_l    = nengo.Node(label="x_l", output=X_LOWER_BOUND)
    # input_x_u    = nengo.Node(label="x_u", output=X_UPPER_BOUND)
    # input_x_0    = nengo.Node(label="x_0", output=X_INITIAL_GUESS)

    # LOCAL STATE
    state = {
        "lb": X_LOWER_BOUND.copy(),
        "ub": X_UPPER_BOUND.copy(),
        "best_x": None,          # best in ORIGINAL space
        "best_v": v0_state.copy(),
        "best_f": None,
        "width_proportion": 1.0,
        "last_adjust": 0.0,
        "retarget_until": 0.0,   # time until which we inject best_v after a shrink
        "last_action_id": ACTIONS.index(1.0),  # start with no shrink
        "reward": 0.0, # last reward obtained
    }

    q_table = np.zeros((len(ACTIONS),))

    # Evaluate the objective function
    def f_obj(t, state_vector):
        """Evaluate the objective function at the given state vector."""
        v       = np.clip(state_vector, -1.0, 1.0)  # Ensure within bounds, consider toroidal space later
        x_vals  = trs2o(v, state["lb"], state["ub"])
        # Implement here the transformation for integer variables if needed
        fv      = problem(x_vals)
        return np.concatenate((v, [fv]))

    state["best_f"]         = f_obj(None, v0_state)[-1]
    # state["last_best_f"]    = state["best_f"]
    state["best_x"]         = trs2o(state["best_v"], state["lb"], state["ub"])

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------

    # [Main] Ensemble to represent the current state vector
    ea          = nengo.networks.EnsembleArray(
        label           = "ens_array",
        n_neurons       = NEURONS_PER_DIM,
        n_ensembles     = NUM_DIMENSIONS,
        ens_dimensions  = 1,
        radius          = 1.0,
        intercepts  = nengo.dists.Uniform(-0.9, 0.9),
        max_rates   = nengo.dists.Uniform(80,220),
        encoders    = nengo.dists.UniformHypersphere(surface=True),
        neuron_type     = nengo.LIF(),
        seed            = SEED_VALUE,
    )

    # Pulser function to inject the initial guess at t=0 and best_v after shrinks
    def pulser(t):
        if t < 0.01:
            return v0_state.copy()
        if t <= state["retarget_until"] and state["best_v"] is not None:
            return state["best_v"]
        return np.zeros(NUM_DIMENSIONS)

    # Pulser node to trigger actions
    pulser_node = nengo.Node(
        label="pulser",
        output=pulser,
        size_out=NUM_DIMENSIONS,
    )

    # Selector / controller node to move towards the best-so-far
    eta = 0.6  # "Learning rate"
    def hs(t, x):
        if state["best_v"] is None:
            return np.zeros_like(x)
        # best_v = state["best_v"]
        # return x + eta * (state["best_v"] - x) + np.random.normal(0, 0.1, size=x.shape)
        # return (state["best_v"] - x)
        return x + eta * (state["best_v"] - x) #+ np.random.normal(0, 0.1, size=x.shape)  # Add some noise for exploration

    hs_node     = nengo.Node(
        label       = "hs",
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = hs
    )

    # Scheduler function to trigger shrink operations
    def neuromodulator(t, best_f):
        _reward                 = 1.0 if best_f.item() < state["best_f"] else -1.0
        state["reward"]         += _reward

        # Trigger every SHRINK_EVERY seconds after a short warmup
        if (state["best_v"] is None) or (t < 0.1) or (t - state["last_adjust"] < SHRINK_EVERY):
            return state["width_proportion"]

        # Q-learning update using the last reward obtained
        last_reward         = state["reward"]
        last_action_idx     = int(state["last_action_id"])
        q_value             = q_table[last_action_idx]
        q_value             += q_alpha * (last_reward + q_gamma * q_table.max() - q_value)
        q_table[last_action_idx] = q_value

        # Choose action based on epsilon-greedy policy
        q_eps_t = max(q_eps_min, q_eps_init * np.exp(-q_eps_decay * t))
        if np.random.rand() < q_eps_t:
            action_idx = np.random.randint(len(ACTIONS))
        else:
            max_q       = np.max(q_table)
            best_idxs   = np.flatnonzero(q_table == max_q)
            action_idx  = np.random.choice(best_idxs)
        proportion = ACTIONS[action_idx]

        # Apply the action: shrink / expand the search space, or RESET to global bounds
        if proportion == -1.0:
            # RESET: restore original global bounds and width proportion
            new_lb = X_LOWER_BOUND0.copy()
            new_ub = X_UPPER_BOUND0.copy()
            state["width_proportion"] = 1.0

            # Recompute best_v under the RESET bounds
            new_best_v = tro2s(state["best_x"], new_lb, new_ub)

        else:
            # Current widths and target widths after shrinking
            half_width  = 0.5 * np.maximum(
                (state["ub"] - state["lb"]) * proportion,  MIN_WIDTH
            )

            # Center the new box around the current best (in original space)
            center      = trs2o(state["best_v"], state["lb"], state["ub"])
            # Enforce minimum width and clip to global box
            new_lb      = np.maximum(center - half_width, X_LOWER_BOUND0)
            new_ub      = np.minimum(center + half_width, X_UPPER_BOUND0)

            # Ensure numerical ordering
            new_lb      = np.minimum(new_lb, new_ub - EPS)
            new_ub      = np.maximum(new_ub, new_lb + EPS)

            # Recompute best_v under the NEW bounds so it stays centred on the same best_x
            if state["best_x"] is not None:
                new_best_v = tro2s(state["best_x"], new_lb, new_ub)
            else:
                new_best_v = state["best_v"]

            state["width_proportion"]   *= proportion

        # Update the local state
        state["lb"], state["ub"]= new_lb, new_ub
        state["best_v"]         = np.clip(new_best_v, -1.0, 1.0)
        state["retarget_until"] = t + ACTION_DELAY
        state["last_adjust"]    = t

        # Update last best f and reward
        state["last_action_id"]     = action_idx
        state["cum_diff"]           = 0.0

        return state["width_proportion"]

    # Scheduler node to trigger shrink operations
    shrink_node = nengo.Node(
        label       = "neuromodulator",
        output      = neuromodulator,
        size_out    = 1,
        size_in     = 1,
    )

    # Selector node to keep track of the best-so-far
    def local_selector(t, x):
        v       = x[:NUM_DIMENSIONS]
        fv      = float(x[NUM_DIMENSIONS])
        if state["best_f"] is None or (fv < state["best_f"] and t >= 0.01):
            state["best_f"] = fv
            state["best_v"] = v.copy()
            # store best in ORIGINAL space under current dynamic bounds
            state["best_x"] = trs2o(state["best_v"], state["lb"], state["ub"])

        if state["best_v"] is None:
            return np.concatenate((v, [fv]))
        else:
            return np.concatenate((state["best_v"], [state["best_f"]]))

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

    # [Aux] Nodes to extract only fbest and xbest from the selector output
    fbest_only = nengo.Node(
        size_in     = NUM_DIMENSIONS+1,
        size_out    = 1,
        output      = lambda t, x: x[-1]
    )
    xbest_only = nengo.Node(
        size_in     = NUM_DIMENSIONS+1,
        size_out    = NUM_DIMENSIONS,
        output      = lambda t, x: trs2o(x[:NUM_DIMENSIONS], state["lb"], state["ub"])
    )


    # CONNECTIONS
    # --------------------------------------------------------------

    # [Pulser] --- v --> [EA input]
    nengo.Connection(
        pre         = pulser_node,
        post        = ea.input,
        synapse     = None,
    )

    # [EA output] --- x --> [Selector]
    nengo.Connection(
        pre         = ea.output,
        post        = hs_node,
        synapse     = 0,
    )

    # [Selector] --- x (delayed) --> [EA input]
    tau = 0.1
    nengo.Connection(
        pre         = hs_node,
        post        = ea.input,
        synapse     = tau,
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
    nengo.Connection(
        pre         = selector_node,
        post        = fbest_only,
        synapse     = 0,
    )

    # [Selector] --- f --> [shrink_scheduler]
    nengo.Connection(
        pre        = selector_node[-1],
        post       = shrink_node,
        synapse    = 0,
    )

    # [Selector] --- xbest --> [xbest_only]
    nengo.Connection(
        pre         = selector_node,
        post        = xbest_only,
        synapse     = 0,
    )

    # PROBES
    # --------------------------------------------------------------
    shrink_trigger  = nengo.Probe(shrink_node, synapse=None)
    ens_lif_val     = nengo.Probe(ea.output, synapse=0.01)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val       = nengo.Probe(fbest_only, synapse=0.01)
    xbest_val       = nengo.Probe(xbest_only, synapse=0.01)
    ea_spk          = [nengo.Probe(ens.neurons, synapse=0.01) for ens in ea.ensembles]

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
    print(f" x{i+1:02d}: [{np.min(sim.data[xbest_val][:,i]):.4f}, {np.max(sim.data[xbest_val][:,i]):.4f}], ", end="")
    print(f"{sim.data[xbest_val][-1][i]:7.4f} :: {problem.optimum.x[i]:7.4f} | {sim.data[xbest_val][-1][i] - problem.optimum.x[i]:7.4f}")

print(f"fbest: {sim.data[fbest_val][-1][0]:.4f} (opt: {problem.optimum.y}), diff: {sim.data[fbest_val][-1][0] - problem.optimum.y:.4f}")
#%%
x = sim.data[xbest_val][-1]
f = problem(x)
print(f"f(xbest): {f:.4f} (opt: {problem.optimum.y}), diff: {f - problem.optimum.y:.4f}")

#%% Process actions
action_proportions = sim.data[shrink_trigger].astype(float).flatten()
action_times = np.arange(len(action_proportions)) * sim.dt

change_proportions = np.diff(action_proportions)
change_indices = np.where(change_proportions != 0)[0]

def add_colors():
    # Indices where the action proportion changes
    for idx in change_indices:
        delta = change_proportions[idx]
        t = action_times[idx + 1]
        if delta < 0:
            color   = "red"
            marker  = "v"
        elif delta == 0:
            color   = "black"
            marker  = "o"
        else:
            color   = "blue"
            marker  = "^"
        plt.axvline(t, color=color, alpha=0.3, linestyle="--", marker=marker)

# Plot the action proportions over time
add_colors()
plt.plot(sim.trange(), action_proportions, color="black")
plt.show()

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
plt.figure()
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -5, 5, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
add_colors()
plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, SIMULATION_TIME)
plt.legend()
plt.show()

#%%
# Plot the decoded output of the ensemble
plt.figure()
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -1, 1, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
add_colors()
plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, SIMULATION_TIME)
plt.legend()
plt.show()

#%%
plt.figure()
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, 1e0, 1e9, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
add_colors()
plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value error")
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far error")
# plt.hlines(problem.optimum.y, 0, simulation_time, colors="r", linestyles="dashed", label="Optimal value")
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")
plt.xlim(0, SIMULATION_TIME)
plt.yscale("log")
plt.legend()
plt.title(f"Obj. func. err. vs time | fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()

#%%
# Plot the spiking output of the ensemble
plt.figure()
spikes_cat = np.hstack([sim.data[p] for p in ea_spk])
rasterplot(sim.trange(), spikes_cat) #, use_eventplot=True)
plt.xlim(0, SIMULATION_TIME)
plt.show()

