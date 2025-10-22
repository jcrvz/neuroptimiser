#%%
import secrets

import matplotlib.pyplot as plt
import nengo
# from collections import deque
import numpy as np
from ioh import get_problem
from nengo.utils.matplotlib import rasterplot, implot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 1  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_DIMENSIONS  = 10  # Number of dimensions for the problem

NEURONS_PER_DIM = 100
NEURONS_PER_ENS = 200
SIMULATION_TIME = 5.0  # seconds

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMENSIONS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMENSIONS)

# Keep copies of the original global bounds
X_LOWER_BOUND0  = X_LOWER_BOUND.copy()
X_UPPER_BOUND0  = X_UPPER_BOUND.copy()

# Dict of features: smi, cooldown_deg, stagnation_deg, narrowness_deg, reset_deg
FTIDX           = {
    "SMI":             0,                  # Smoothed improvement (higher is better)
    "CDD":             1,                  # Cooldown degree (higher means less time since last action)
    "SGD":             2,                  # Stagnation degree (higher means more time since last improvement)
    "NRD":             3,                  # Narrowness degree (higher means narrower search box)
    "RSD":             4,                  # Can-reset degree (higher means more time since last RESET)
}

# List of actions
ACTIONS         = ["SHRINK", "RESET"]
NUM_ACTIONS     = len(ACTIONS)

# Exploitation (search-space shrinking) schedule
SEED_VALUE      = 69
INIT_WAIT_TIME  = 0.01
DEFAULT_WAIT    = 0.02       # seconds between action operations
STAG_WAIT       = 0.5       # seconds of stagnation before a RESET to global bounds
EPS             = 1e-12     # small value to ensure numerical ordering of bounds
MIN_WIDTH       = 1e-6      # do not shrink any dimension below this absolute width
MIN_PROPORTION  = 1e-4      # do not shrink the box below this proportion of the original box
MIN_IMPROVEMENT = 1e-6      # minimum improvement to consider the optimisation progressing
ALPHA_IMPROV    = 0.8       # EWMA alpha for improvement smoothing / depends on dt
PES_LR_RATE    = 1e-4      # learning rate for the PES rule on the utility ensemble
TAU            = 0.2       # time constant for integrating centre/spread updates
CENTRE_LEAK    = 0.03       # leak for centre integrator
SPREAD_LEAK    = 0.08       # leak for spread integrator

SPREAD_MAX     = 1.5
SPREAD_MIN     = 0.05
NUM_FEATURES    = 2


#%%

# Create the Nengo model
model = nengo.Network(label="nNeurOpti V1", seed=69)
with model:
    # INITIALISATION
    # --------------------------------------------------------------
    v0_state    = tro2s(X_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)

    # LOCAL STATE
    state = {
        "lb":               X_LOWER_BOUND.copy(),   # dynamic local bounds
        "ub":               X_UPPER_BOUND.copy(),
        "best_x":           None,                   # best in ORIGINAL space
        "best_v":           v0_state.copy(),        # best in SCALED space
        "best_f":           None,                   # best objective value
        "smoothed_improv":  1.0,                    # smoothed improvement metric
        "t_stag_start":     0.0,                    # time when stagnation started
        "prev_best_f":      None,                   # previous best objective value
    }
    # rng     =   np.random.default_rng(SEED_VALUE)

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------

    # ================[ ENVIRONMENT: OBJECTIVE FUNCTION ]================

    # Evaluate the objective function
    def obj_func(t, state_vector):
        """Evaluate the objective function at the given state vector.

        Args:
            t (float): Current time (not used, yet).
            state_vector (np.ndarray): State vector in scaled space [-1, 1].

        Returns:
            np.ndarray: Concatenation of the state vector and its objective value.
        """
        v       = np.clip(state_vector, -1.0, 1.0)   # Ensure within [-1, 1]
        x_vals  = np.array(trs2o(v, state["lb"], state["ub"]))        # Transform to original space

        fv      = problem(x_vals)                           # Evaluate the objective function
        return np.concatenate((v, [fv]))

    # Initialise the best-so-far
    state["best_f"]         = obj_func(None, v0_state)[-1]
    state["prev_best_f"]    = state["best_f"]
    state["best_x"]         = np.array(trs2o(state["best_v"], state["lb"], state["ub"]))

    # Objective function node - Equivalent to the Environment in RL
    obj_node    = nengo.Node(
        label       = "f_obj",
        output      = obj_func,
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS+1,
    )

    # ================[ COMPOSITION ENSEMBLES ]================

    init_c_node = nengo.Node(
        size_in     = 0,
        size_out    = NUM_DIMENSIONS,
        output      = lambda t: v0_state.copy() if t < INIT_WAIT_TIME else np.zeros(NUM_DIMENSIONS),
    )

    init_s_node = nengo.Node(
        size_in     = 0,
        size_out    = NUM_DIMENSIONS,
        output      = lambda t: np.ones(NUM_DIMENSIONS) if t < INIT_WAIT_TIME else np.zeros(NUM_DIMENSIONS),
    )

    centre_ens      = nengo.Ensemble(
        n_neurons   = NEURONS_PER_ENS + 20 * NUM_DIMENSIONS,
        dimensions  = NUM_DIMENSIONS,
        radius      = 1.5,
    )

    spread_ens      = nengo.Ensemble(
        n_neurons   = NEURONS_PER_ENS + 20 * NUM_DIMENSIONS,
        dimensions  = NUM_DIMENSIONS,
        radius      = 1.5,
    )

    def compose_v(t, data):
        variable    = data[:NUM_DIMENSIONS]
        centre      = data[NUM_DIMENSIONS:2*NUM_DIMENSIONS]
        spread      = np.clip(data[2*NUM_DIMENSIONS:], SPREAD_MIN, SPREAD_MAX)

        return np.clip(centre + variable * spread, -1.0, 1.0)

    compose_node = nengo.Node(
        label       = "compose_v",
        size_in     = 3 * NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = compose_v,
    )

    # ================[ MOTOR ENSEMBLE ]================
    # Ensemble to represent the current candidate solution

    motor_ens   = nengo.networks.EnsembleArray(
        label           = "motor",
        n_neurons       = NEURONS_PER_DIM,
        n_ensembles     = NUM_DIMENSIONS,
        ens_dimensions  = 1,
        radius          = 1.0,
        intercepts      = nengo.dists.Uniform(-0.99, 0.99),
        max_rates   = nengo.dists.Uniform(80,220),
        encoders        = nengo.dists.UniformHypersphere(surface=True),
        neuron_type     = nengo.AdaptiveLIF(),
        seed            = SEED_VALUE,
    )

    def _features_func(t):
        """Extract features from the current state for utility computation.
        """
        # Read the current improvement
        # The smoothed improvement (smi) measures how well the optimisation is progressing, in [0, 1], where 1 is best, 0 is worst.
        smi         = np.clip(state["smoothed_improv"], 0.0, 1.0)

        # Stagnation degree
        # The stagnation degree measures how long since the last improvement, in [0, 1], where 1 means ready to RESET.
        stagnation_deg  = np.clip(
            (t - state["t_stag_start"]) / STAG_WAIT,
            0.0, 1.0)

        # Update the current features
        return  np.array([smi, stagnation_deg],
            dtype=float)

    # state["curr_features"] = _features_func(0.0)


    # Features node to extract features for utility computation
    features_node = nengo.Node(
        label       = "features",
        output      = _features_func,
        size_in     = 0,
        size_out    = NUM_FEATURES,
    )

    # Selector node to keep track of the best-so-far
    def selector_func(t, x):

        # Read the current candidate solution and its objective value
        v       = x[:NUM_DIMENSIONS]
        fv      = float(x[NUM_DIMENSIONS])

        if state["prev_best_f"] is None:
            state["prev_best_f"] = fv

        # Update improvement metrics
        if state["best_f"] is None or (fv < state["best_f"] and t >= INIT_WAIT_TIME):
            # Register the previous best
            # Update the best-so-far
            state["best_f"] = fv
            state["best_v"] = v.copy()
            # store best in ORIGINAL space under current dynamic bounds
            state["best_x"] = trs2o(state["best_v"], state["lb"], state["ub"])

        # Improvement since last step (minimisation)
        difference = max(0.0, state["prev_best_f"] - state["best_f"])
        denominator = EPS + abs(state["prev_best_f"])
        improvement = difference / denominator
        state["smoothed_improv"] = (1 - ALPHA_IMPROV) * state["smoothed_improv"] + ALPHA_IMPROV * improvement

        # Update stagnation timer
        if difference > MIN_IMPROVEMENT:
            state["t_stag_start"] = t
        state["prev_best_f"] = state["best_f"]

        if state["best_v"] is None:
            return np.concatenate((v, [fv]))
        else:
            return np.concatenate((state["best_v"], [state["best_f"]]))

    selector_node   = nengo.Node(
        label       = "selector",
        output      = selector_func,
        size_in     = NUM_DIMENSIONS + 1,
        size_out    = NUM_DIMENSIONS + 1,
    )


    # Heads that output desired increments for centre and spread
    c_update_ens   = nengo.Ensemble(
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_DIMENSIONS,
    )
    s_update_ens   = nengo.Ensemble(
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_DIMENSIONS,
    )

    # Encode features
    features_ens = nengo.Ensemble(
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_FEATURES,
    )




    # Selector / controller node to move towards the best-so-far
    def hs(t, x):
        if state["best_v"] is None:
            return np.zeros_like(x)

        eta     = 1.0
        sigma   = 0.05

        return eta * (state["best_v"] - x) + np.random.normal(0, sigma, size=NUM_DIMENSIONS)

    hs_node     = nengo.Node(
        label       = "hs",
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = hs
    )


    # Pulser function to inject the initial guess at t=0 and best_v after shrinks
    def pulser_func(t):
        if (t < INIT_WAIT_TIME):
            return v0_state.copy()
        # if t <= state["retarget_until"] and state["best_v"] is not None:
            # return state["best_v"] + np.random.normal(0, 0.01, size=NUM_DIMENSIONS)
        return np.zeros(NUM_DIMENSIONS)

    # Pulser node to trigger actions
    pulser_node = nengo.Node(
        label="pulser",
        output=pulser_func,
        size_out=NUM_DIMENSIONS,
    )

    # TODO: this could be a memory
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

    # LEARNING RULES (Teaching signals)
    def _c_teaching_func(t, x):
        """Centre update teaching signal."""
        best_v      = x[:NUM_DIMENSIONS]
        curr_v      = x[NUM_DIMENSIONS:2*NUM_DIMENSIONS]
        smi, sgd    = x[2*NUM_DIMENSIONS:]

        # Move centre toward best_v, scaled by smi and sgd
        # centre_val  = sgd * best_v - 10 * (sgd * curr_v)
        centre_val = (smi + 0.5*sgd) * (best_v - curr_v)
        return centre_val

    def _s_teaching_func(t, x):
        """Spread update teaching signal."""
        smi,sgd     = x

        # Reduce spread when improving, increase when stagnating
        # spread_val  = 0.02 * (sgd - 0.5 * (1.0 - smi))
        spread_val  = sgd + (1.0 - sgd) * MIN_WIDTH
        # spread_val = 0.2 * (sgd - smi)
        return spread_val * np.ones(NUM_DIMENSIONS)

    c_teaching_node = nengo.Node(
        label       = "c_teach",
        size_in     = 2 * NUM_DIMENSIONS + NUM_FEATURES,
        size_out    = NUM_DIMENSIONS,
        output      = _c_teaching_func,
    )

    s_teaching_node = nengo.Node(
        label       = "s_teach",
        size_in     = NUM_FEATURES,
        size_out    = NUM_DIMENSIONS,
        output      = _s_teaching_func,
    )

    # CONNECTIONS
    # --------------------------------------------------------------

    # [Pulser] --- v --> [EA input]
    nengo.Connection(
        pre         = pulser_node,
        post        = motor_ens.input,
        synapse     = None,
    )

    # [EA output] --- x --> [Selector]
    nengo.Connection(
        pre         = motor_ens.output,
        post        = hs_node,
        synapse     = 0,
    )

    # [Selector] --- x (delayed) --> [EA input]
    nengo.Connection(
        pre         = hs_node,
        post        = motor_ens.input,
        synapse     = TAU,
        transform   = np.eye(NUM_DIMENSIONS),
    )

    # [EA output] + [Centre] + [Spread] --> [Compose v]
    nengo.Connection(
        pre         = motor_ens.output,
        post        = compose_node[:NUM_DIMENSIONS],
        synapse     = 0,
    )
    nengo.Connection(
        pre         = centre_ens,
        post        = compose_node[NUM_DIMENSIONS:2*NUM_DIMENSIONS],
        synapse     = 0.02,
    )
    nengo.Connection(
        pre         = spread_ens,
        post        = compose_node[2*NUM_DIMENSIONS:],
        synapse     = 0.02,
    )

    # [Compose v] --- v --> [Objective function]
    nengo.Connection(
        pre         = compose_node,
        post        = obj_node,
        synapse     = 0,
    )

    # [Objective function] --- (x,f) --> [Selector]
    nengo.Connection(
        pre         = obj_node,
        post        = selector_node,
        synapse     = 0,
    )

    # [Features] --- features --> [Features ensemble]
    nengo.Connection(
        pre         = features_node,
        post        = features_ens,
        synapse     = None,
    )

    # Learn feature from updates with PES rule
    c_connect = nengo.Connection(
        pre                 = features_ens,
        post                = c_update_ens,
        function            = lambda x: np.zeros(NUM_DIMENSIONS),
        learning_rule_type  = nengo.PES(learning_rate=PES_LR_RATE),
    )
    s_connect = nengo.Connection(
        pre                 = features_ens,
        post                = s_update_ens,
        function            = lambda x: np.zeros(NUM_DIMENSIONS),
        learning_rule_type  = nengo.PES(learning_rate=PES_LR_RATE),
    )

    # Integrate updates into centre and spread ensembles

    nengo.Connection(
        pre         = init_c_node,
        post        = centre_ens,
        synapse     = TAU,
    )

    nengo.Connection(
        pre         = init_s_node,
        post        = spread_ens,
        synapse     = TAU,
    )

    nengo.Connection(
        pre         = c_update_ens,
        post        = centre_ens,
        synapse     = TAU,
    )
    nengo.Connection(
        pre         = s_update_ens,
        post        = spread_ens,
        synapse     = TAU,
    )

    # Make centre/spread "stateful" via an NEF integrator: τ dx/dt = u
    nengo.Connection(
        pre         = centre_ens,
        post        = centre_ens,
        synapse     = TAU,
        transform   = (1.0 - CENTRE_LEAK) * np.eye(NUM_DIMENSIONS),
    )
    nengo.Connection(
        pre         = spread_ens,
        post        = spread_ens,
        synapse     = TAU,
        transform   = (1.0 - SPREAD_LEAK) * np.eye(NUM_DIMENSIONS),
    )

    # [Selector] --- fbest --> [fbest_only]
    nengo.Connection(
        pre         = selector_node,
        post        = fbest_only,
        synapse     = 0,
    )


    # [Selector] --- xbest --> [xbest_only]
    nengo.Connection(
        pre         = selector_node,
        post        = xbest_only,
        synapse     = 0,
    )

    # [Motor, Selector, Features] --> [c_teaching, s_teaching]
    nengo.Connection(
        pre         = selector_node[:NUM_DIMENSIONS],
        post        = c_teaching_node[:NUM_DIMENSIONS],
        synapse     = 0,
    )
    nengo.Connection(
        pre         = motor_ens.output,
        post        = c_teaching_node[NUM_DIMENSIONS:2*NUM_DIMENSIONS],
        synapse     = 0.01, # delay to allow selector to update first, how much? 0.01 ?
    )
    nengo.Connection(
        pre         = features_node,
        post        = c_teaching_node[2*NUM_DIMENSIONS:],
        synapse     = 0.01,
    )

    nengo.Connection(
        pre         = features_node,
        post        = s_teaching_node,
        synapse     = 0,
    )

    nengo.Connection(
        pre         = c_teaching_node,
        post        = c_connect.learning_rule,
        synapse     = TAU,
    )
    nengo.Connection(
        pre         = s_teaching_node,
        post        = s_connect.learning_rule,
        synapse     = TAU,
    )

    # PROBES
    # --------------------------------------------------------------
    # shrink_trigger  = nengo.Probe(_proportion_node, synapse=None)
    ens_lif_val     = nengo.Probe(motor_ens.output, synapse=0.01)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val       = nengo.Probe(fbest_only, synapse=0.01)
    xbest_val       = nengo.Probe(xbest_only, synapse=0.01)
    ea_spk          = [nengo.Probe(ens.neurons, synapse=0.01) for ens in motor_ens.ensembles]

    centre_val      = nengo.Probe(c_teaching_node, synapse=0.01)
    spread_val      = nengo.Probe(s_teaching_node, synapse=0.01)
    features_val    = nengo.Probe(features_node, synapse=0.01)

    # utility_vals    = nengo.Probe(utility_ens, synapse=0.01)

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


#%%
# plt.figure()
# # plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value")
# # plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, 1e0, 1e10, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
#
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")
# plt.xlim(0, SIMULATION_TIME)
# plt.yscale("log")
# plt.legend()
# plt.title(f"Obj. func. err. vs time | fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
# plt.show()

#%%
# Plot the decoded output of the ensemble
plt.figure(figsize=(12, 8), dpi=150)
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -5, 5, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, SIMULATION_TIME)
plt.legend()
plt.show()

#%%
# Plot the decoded output of the ensemble
plt.figure(figsize=(12, 8), dpi=150)
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -1, 1, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, SIMULATION_TIME)
plt.legend()
plt.show()

#%%
plt.figure(figsize=(12, 8), dpi=150)
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, 1e0, 1e9, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value error")
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far error")
# plt.hlines(problem.optimum.y, 0, simulation_time, colors="r", linestyles="dashed", label="Optimal value")
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")

# add second axis for centre and spread
ax1 = plt.gca()
plt.xlim(0, SIMULATION_TIME)
plt.xlabel("Time, s")
plt.ylabel("Objective value error")
plt.yscale("log")
plt.legend()

ax2 = ax1.twinx()
ax2.plot(sim.trange(), np.average(sim.data[centre_val], axis=1), "g--", label="Centre")
ax2.plot(sim.trange(), np.average(sim.data[spread_val], axis=1), "m--", label="Spread")
ax2.plot(sim.trange(), sim.data[features_val][:,0], "c:", label="SMI")
ax2.plot(sim.trange(), sim.data[features_val][:,1], "y:", label="SGD")
ax2.set_ylabel("Centre / Spread values")
ax2.set_yscale("linear")
ax2.legend(loc="upper right")

plt.title(f"Obj. func. err. vs time | fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()

#%%
# Plot the spiking output of the ensemble
# spikes_cat = []
# for i, p in enumerate(ea_spk):
#     spikes = sim.data[p]
#     # indices = sorted_neurons(motor_ens.ensembles[i], sim, iterations=250)
#     # spikes_cat.append(spikes[:, indices])
#     spikes_cat.append(spikes)
# spikes_cat = np.hstack(spikes_cat)

plt.figure(figsize=(12, 8), dpi=150)
spikes_cat = np.hstack([sim.data[p] for p in ea_spk])
# rasterplot(sim.trange(), spikes_cat) #, use_eventplot=True)
plt.imshow(spikes_cat, aspect="auto", cmap="pink_r", origin="lower",)

t_indices = plt.gca().get_xticks().astype(int)
t_indices = t_indices[t_indices >= 0.0]
t_indices[-1] -= 1
t_labels = [f"{sim.trange()[i]:.1f}" for i in t_indices]
plt.xticks(t_indices, t_labels)

plt.xlabel("Time, s")
plt.ylabel("Neuron")
plt.title("Spikes")
# plt.xlim(0, SIMULATION_TIME)
plt.show()
