#%%
import secrets

import matplotlib.pyplot as plt
import nengo
from collections import deque
import numpy as np
from ioh import get_problem
from nengo.utils.matplotlib import rasterplot, implot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 15  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_DIMENSIONS  = 10  # Number of dimensions for the problem

NEURONS_PER_DIM = 100
NEURONS_PER_ENS = 200
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
DEFAULT_WAIT    = 0.01       # seconds between action operations
STAG_WAIT       = max(0.01, SIMULATION_TIME * 0.1)       # seconds of stagnation before a RESET to global bounds
EPS             = 1e-12     # small value to ensure numerical ordering of bounds
MIN_WIDTH       = 1e-6      # do not shrink any dimension below this absolute width
MIN_PROPORTION  = 1e-4      # do not shrink the box below this proportion of the original box
MIN_IMPROVEMENT = 1e-6      # minimum improvement to consider the optimisation progressing
ALPHA_IMPROV    = 0.8       # EWMA alpha for improvement smoothing / depends on dt
TAU_1           = 0.02       # time constant
TAU_2           = 0.02        # time constant
CENTRE_LEAK    = 0.03       # leak for centre integrator
SPREAD_LEAK    = 0.08       # leak for spread integrator

SPREAD_MAX     = 1.0
SPREAD_MIN     = 1e-6
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
        "curr_features":    np.array([0.0, 1.0]),   # current features vector
        "archive":          deque(maxlen=64),       # small archive of best solutions
        "stag_pulse":       False,                  # flag for stagnation pulse
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

    def compose_v(t, data):
        variable    = data[:NUM_DIMENSIONS]
        centre      = data[NUM_DIMENSIONS:2*NUM_DIMENSIONS]
        spread      = np.clip(data[2*NUM_DIMENSIONS:], MIN_WIDTH, SPREAD_MAX)

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
        # max_rates   = nengo.dists.Uniform(80,220),
        encoders        = nengo.dists.UniformHypersphere(surface=True),
        neuron_type     = nengo.LIF(),
        seed            = SEED_VALUE,
    )

    def _features_func(t):
        """Extract features from the current state for utility computation.
        """
        # Read the current improvement
        if state["curr_features"][1] >= 1.0:
            scale                   = max(1.0, abs(state["best_f"]))
            state["prev_best_f"]    = 1e6 * scale

        # Improvement since last step (minimisation)
        difference  = max(0.0, state["prev_best_f"] - state["best_f"])
        denominator = EPS + abs(state["prev_best_f"])
        improvement = difference / denominator
        state["smoothed_improv"] = np.clip(
            ALPHA_IMPROV * state["smoothed_improv"] + (1 - ALPHA_IMPROV) * improvement,
            0.0, 1.0).astype(float)

        # The smoothed improvement (smi) measures how well the optimisation is progressing, in [0, 1], where 1 is best, 0 is worst.
        smi             = state["smoothed_improv"]

        if improvement >= MIN_IMPROVEMENT:
            state["t_stag_start"] = t
            state["stag_pulse"] = False

        # Stagnation degree
        # The stagnation degree measures how long since the last improvement, in [0, 1], where 1 means ready to RESET.
        stagnation_deg  = np.clip(
            (t - state["t_stag_start"]) / STAG_WAIT,
            0.0, 1.0)

        # Update the current features
        state["curr_features"]  = np.array([smi, stagnation_deg], dtype=float)
        return state["curr_features"]

    # state["curr_features"] = _features_func(0.0)


    # Features node to extract features for utility computation
    features_node = nengo.Node(
        label       = "features",
        output      = _features_func,
        size_in     = 0,
        size_out    = NUM_FEATURES,
    )


    def _add_to_archive(v, f):
        """Add to archive with distance-based deduplication."""
        MIN_ARCHIVE_DIST = 0.1  # in scaled space

        # Remove near-duplicates
        kept = []
        for vi, fi in state["archive"]:
            if np.linalg.norm(v - vi) >= MIN_ARCHIVE_DIST:
                kept.append((vi, fi))

        kept.append((v.copy(), float(f)))
        kept.sort(key=lambda t: t[1])  # sort by fitness
        state["archive"] = deque(kept[:16], maxlen=16)  # keep top 16

    # Selector node to keep track of the best-so-far
    def selector_func(t, x):

        # Read the current candidate solution and its objective value
        v       = x[:NUM_DIMENSIONS]
        fv      = float(x[NUM_DIMENSIONS])

        # if state["prev_best_f"] is None:
        #     state["prev_best_f"] = fv

        # Update improvement metrics
        if state["best_f"] is None or (fv < state["best_f"] and t >= INIT_WAIT_TIME):
            _add_to_archive(state["best_v"], state["best_f"])
            # Register the previous best
            state["prev_best_f"] = state["best_f"]

            # Update the best-so-far
            state["best_f"] = fv
            state["best_v"] = v.copy()
            # store best in ORIGINAL space under current dynamic bounds
            state["best_x"] = trs2o(state["best_v"], state["lb"], state["ub"])

            # Add to small archive
            state["archive"].append((state["best_v"], state["best_f"]))

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


    # Selector / controller node to move towards the best-so-far
    def hs(t, x):
        if state["best_v"] is None:
            return np.zeros_like(x)

        smi, sgd    = state["curr_features"]
        eta         = 0.3 * (1.0 - sgd)
        sigma       = 0.02 + 0.15 * sgd

        # # OPTION 2: Heavy-tail only when stagnated
        # # Optional heavy tail during full stagnation using existing RNG
        # if sgd >= 1.0:
        #     # Replace Gaussian with a crude heavy-tail using a Student-t approximation
        #     z = np.random.normal(0, 1, size=NUM_DIMENSIONS)
        #     u = np.random.normal(0, 1, size=NUM_DIMENSIONS)
        #     heavy = z / np.sqrt((u * u) / 3.0 + 1e-9)  # ~ t with ν≈3, no new knob
        #     return np.zeros_like(x) + 0.5 * heavy  # no attraction while sgd==1

        return eta * (state["best_v"] - x) + np.random.normal(0, sigma, size=NUM_DIMENSIONS)

    hs_node     = nengo.Node(
        label       = "hs",
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = hs
    )

    def _archive_centroid_scaled():
        if not state["archive"]:
            return np.zeros(NUM_DIMENSIONS)

        vb, fvb     = zip(*state["archive"])
        vb          = np.array(vb)
        fvb         = np.array(fvb, dtype=float)

        # Rank by fitness (lower is better). weights = 1 / rank
        orders          = np.argsort(fvb)
        ranks           = np.empty_like(orders)
        ranks[orders]   = np.arange(1, len(fvb) + 1)
        weights         = 1.0 / ranks
        weights         /= weights.sum()
        centroid        = (weights[:, None] * vb).sum(axis=0)
        # centroid    = np.sum(vb * weights, axis=0)
        return centroid

    # Pulser function to inject the initial guess at t=0 and best_v after shrinks
    # def pulser_func(t):
    #     if t < INIT_WAIT_TIME:
    #         return v0_state.copy()
    #     # if t - state["t_stag_start"] >= 2 * STAG_WAIT:
    #     #     sigma       = sgd * 0.5
    #     #     return state["best_v"] + np.random.normal(0, sigma, size=NUM_DIMENSIONS)
    #
    #     smi, sgd    = state["curr_features"]
    #     # Option 1:
    #     if (sgd >= 1.0): # and (not state["stag_pulse"]):
    #         return np.zeros(NUM_DIMENSIONS)
    #         # state["stag_pulse"] = True
    #         # zero centre to sample around 0, or archive centroid for informed restart
    #         # v_seed      = _archive_centroid_scaled() if state["archive"] else np.zeros(NUM_DIMENSIONS)
    #         # jitter      = np.random.normal(0.0, 0.05, NUM_DIMENSIONS)
    #         # return np.clip(v_seed + jitter, -1.0, 1.0)
    #
    #     # Option 3: Partial re-sampling of dimensions during stagnation
    #     # if sgd >= 1.0 and np.random.rand() < 0.5:
    #     #     # Base seed: zero centre or archive centroid if you already have it
    #     #     v_seed = np.zeros(NUM_DIMENSIONS)
    #     #     # If you kept the small archive helper, uncomment:
    #     #     # v_seed = _archive_centroid_scaled() if state["archive"] else v_seed
    #     #
    #     #     # Rotate a simple dimension mask with no new knobs
    #     #     # Bout 1: even dims; Bout 2: odd dims; then repeat
    #     #     mask = np.zeros(NUM_DIMENSIONS, dtype=bool)
    #     #     mask[np.random.choice(NUM_DIMENSIONS, size=NUM_DIMENSIONS//2, replace=False)] = True
    #     #
    #     #     # Sample only masked dims globally in scaled space; keep others
    #     #     new_vals = np.random.uniform(-1.0, 1.0, size=NUM_DIMENSIONS)
    #     #     v_curr = state["best_v"] if state["best_v"] is not None else v_seed
    #     #     v_next = np.where(mask, new_vals, v_curr)
    #     #
    #     #     return np.clip(v_next, -1.0, 1.0)
    #
    #     return np.zeros(NUM_DIMENSIONS)

    # # Pulser node to trigger actions
    # pulser_node = nengo.Node(
    #     label="pulser",
    #     output=pulser_func,
    #     size_out=NUM_DIMENSIONS,
    # )

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
        best_v = x[:NUM_DIMENSIONS]
        smi, sgd = x[2 * NUM_DIMENSIONS:]

        # Move centre toward best_v, scaled by smi and sgd
        # weight      = sgd ** 3.0
        # centre_val  = (1.0 - weight) * best_v
        return best_v

    def _s_teaching_func(t, x):
        smi, sgd = x

        # Shrink when improving (smi high), expand when stagnating (sgd high)
        target = SPREAD_MIN + (SPREAD_MAX - SPREAD_MIN) * np.clip(sgd - 0.3 * smi, 0.0, 1.0)
        return target * np.ones(NUM_DIMENSIONS)


    c_teaching_node = nengo.Node(
        label       = "centre_control",
        size_in     = 2 * NUM_DIMENSIONS + NUM_FEATURES,
        size_out    = NUM_DIMENSIONS,
        output      = _c_teaching_func,
    )

    s_teaching_node = nengo.Node(
        label       = "spread_control",
        size_in     = NUM_FEATURES,
        size_out    = NUM_DIMENSIONS,
        output      = _s_teaching_func,
    )

    # Replace teaching nodes with integrator ensembles

    centre_int = nengo.networks.EnsembleArray(
        n_neurons=NEURONS_PER_ENS, n_ensembles=NUM_DIMENSIONS,
        ens_dimensions=1, radius=1.5, label="centre_integrator"
    )

    spread_int = nengo.networks.EnsembleArray(
        n_neurons=NEURONS_PER_ENS, n_ensembles=NUM_DIMENSIONS,
        ens_dimensions=1, radius=2.0, label="spread_integrator"
    )

    # CONNECTIONS
    # --------------------------------------------------------------

    nengo.Connection(
        pre         = motor_ens.output,
        post        = motor_ens.input,
        synapse     = TAU_1,
        transform   = 0.95 * np.eye(NUM_DIMENSIONS),
    )

    nengo.Connection(
        pre        = motor_ens.output,
        post       = hs_node,
        synapse    = None,
    )

    nengo.Connection(
        pre         = hs_node,
        post        = motor_ens.input,
        synapse     = TAU_1,
        transform   = 0.3 * np.eye(NUM_DIMENSIONS)
    )

    # [Pulser] --- v --> [EA input]
    # nengo.Connection(
    #     pre         = pulser_node,
    #     post        = motor_ens.input,
    #     synapse     = None,
    # )

    # [Selector] --- x (delayed) --> [EA input]
    # nengo.Connection(
    #     pre         = hs_node,
    #     post        = motor_ens.input,
    #     synapse     = TAU_1,
    #     transform   = np.eye(NUM_DIMENSIONS),
    # )

    # [EA output] + [Centre] + [Spread] --> [Compose v]
    nengo.Connection(
        pre         = motor_ens.output,
        post        = compose_node[:NUM_DIMENSIONS],
        synapse     = 0,
    )
    # nengo.Connection(
    #     pre         = c_teaching_node,
    #     post        = compose_node[NUM_DIMENSIONS:2*NUM_DIMENSIONS],
    #     synapse     = TAU_2,
    # )
    # nengo.Connection(
    #     pre         = s_teaching_node,
    #     post        = compose_node[2*NUM_DIMENSIONS:],
    #     synapse     = TAU_2,
    # )

    # Self-recurrence with leak
    nengo.Connection(centre_int.output, centre_int.input, synapse=0.05, transform=(1.0 - CENTRE_LEAK))
    nengo.Connection(spread_int.output, spread_int.input, synapse=0.05, transform=(1.0 - SPREAD_LEAK))

    # Teaching signals drive the integrators
    nengo.Connection(c_teaching_node, centre_int.input, synapse=0.02, transform=CENTRE_LEAK)
    nengo.Connection(s_teaching_node, spread_int.input, synapse=0.02, transform=SPREAD_LEAK)

    # Use integrator outputs in composition
    nengo.Connection(centre_int.output, compose_node[NUM_DIMENSIONS:2 * NUM_DIMENSIONS], synapse=0)
    nengo.Connection(spread_int.output, compose_node[2 * NUM_DIMENSIONS:], synapse=0)

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
        synapse     = TAU_1,
    )
    nengo.Connection(
        pre         = motor_ens.output,
        post        = c_teaching_node[NUM_DIMENSIONS:2*NUM_DIMENSIONS],
        synapse     = TAU_1, # delay to allow selector to update first, how much? 0.01 ?
    )
    nengo.Connection(
        pre         = features_node,
        post        = c_teaching_node[2*NUM_DIMENSIONS:],
        synapse     = TAU_1,
    )

    nengo.Connection(
        pre         = features_node,
        post        = s_teaching_node,
        synapse     = TAU_1,
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
plt.imshow(spikes_cat.T, aspect="auto", cmap="pink_r", origin="lower",)

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
