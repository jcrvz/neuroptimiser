#%%
import matplotlib.pyplot as plt
import nengo
# from collections import deque
import numpy as np
from ioh import get_problem
from nengo.utils.matplotlib import rasterplot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 12  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_DIMENSIONS  = 10  # Number of dimensions for the problem
NEURONS_PER_DIM = 100
NEURONS_PER_ENS = 20
SIMULATION_TIME = 20.0  # seconds

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMENSIONS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMENSIONS)

# Keep copies of the original global bounds
X_LOWER_BOUND0  = X_LOWER_BOUND.copy()
X_UPPER_BOUND0  = X_UPPER_BOUND.copy()

# List of actions
ACTIONS         = ["SHRINK", "HOLD", "RESET"]
NUM_ACTIONS     = len(ACTIONS)

# Exploitation (search-space shrinking) schedule
SEED_VALUE      = 69
INIT_WAIT_TIME  = 0.1
DEFAULT_WAIT    = 1.0       # seconds between shrink operations
STAG_RESET      = 3.0       # seconds of stagnation before a RESET to global bounds
EPS             = 1e-12     # small value to ensure numerical ordering of bounds
MIN_WIDTH       = 1e-6      # do not shrink any dimension below this absolute width
MIN_PROPORTION  = 1e-4      # do not shrink the box below this proportion of the original box
MIN_IMPROVEMENT = 1e-6      # minimum improvement to consider the optimisation progressing
ACTION_DELAY    = 0.01      # seconds to retarget the EA to the best_v after a shrink
ALPHA_IMPROV    = 0.9       # EWMA alpha for improvement smoothing
# MIN_WIDTH_FRAC   = 0.02    # do not shrink any dimension below 2% of the original rang

# Q learning parameters
ALPHA_Q         = 0.1       # learning rate
GAMMA_Q         = 0.9       # discount factor
SOFTMAX_TEMP    = 1.0       # softmax temperature for action selection

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
        "width_proportion": 1.0,                    # current width proportion relative to original box
        "last_adjust":      0.0,                    # time of last shrink / reset
        "retarget_until":   0.0,                    # time until which to retarget the EA to best_v
        "last_reset_time":  0.0,                   # time of last RESET action
        "t_stag_start":     0.0,                    # time when stagnation started
        "wait_time":        0.1,           # seconds between shrink/reset operations
        "reset_cooldown":   5.0,                    # seconds between RESET actions
        "smoothed_improv":  0.0,                    # smoothed improvement metric
        "q_w":               None,                   # weights: shape (NUM_ACTIONS, NUM_FEATURES)
        "prev_features":    None,                   # previous features: shape (NUM_FEATURES,)
        "prev_action":      None,                   # previous action index executed
        "prev_best_f":      None,                   # previous best objective value
    }

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
        x_vals  = trs2o(v, state["lb"], state["ub"])        # Transform to original space

        fv      = problem(x_vals)                           # Evaluate the objective function
        return np.concatenate((v, [fv]))

    # Initialise the best-so-far
    state["best_f"]         = obj_func(None, v0_state)[-1]
    state["prev_best_f"]    = state["best_f"]
    state["best_x"]         = trs2o(state["best_v"], state["lb"], state["ub"])

    # Objective function node - Equivalent to the Environment in RL
    obj_node    = nengo.Node(
        label       = "f_obj",
        output      = obj_func,
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS+1,
    )

    # ================[ MOTOR ENSEMBLE ]================
    # Ensemble to represent the current candidate solution

    motor_ens   = nengo.networks.EnsembleArray(
        label           = "motor",
        n_neurons       = NEURONS_PER_DIM,
        n_ensembles     = NUM_DIMENSIONS,
        ens_dimensions  = 1,
        radius          = 1.0,
        intercepts      = nengo.dists.Uniform(-0.9, 0.9),
        # max_rates   = nengo.dists.Uniform(100,220),
        encoders        = nengo.dists.UniformHypersphere(surface=True),
        neuron_type     = nengo.AdaptiveLIF(),
        seed            = SEED_VALUE,
    )

    # ================[ BASAL GANGLIA AND THALAMUS ]================
    # Networks to select actions based on utilities

    basal_ganglia = nengo.networks.BasalGanglia(
        dimensions              = NUM_ACTIONS,        # 3 actions: shrink, hold, reset
        n_neurons_per_ensemble  = NEURONS_PER_ENS,       # neurons per action ensemble
    )

    thalamus = nengo.networks.Thalamus(
        dimensions              = NUM_ACTIONS,        # 3 actions: shrink, hold, reset
        n_neurons_per_ensemble  = NEURONS_PER_ENS,       # neurons per action ensemble
    )

    # ================[ UTILITY ENSEMBLE ]================
    # Ensemble to represent the utility of each action

    def _features_func(t):
        """Extract features from the current state for utility computation.
        """

        # Read the current improvement
        smi         = np.clip(state["smoothed_improv"], 0.0, 1.0)

        # Stagnation degree
        stag_time       = max(0.0, t - state["t_stag_start"])
        stagnation_deg  = np.clip(stag_time / STAG_RESET, 0.0, 1.0)

        # Cooldown degree
        cool_left       = max(0.0, state["wait_time"] - (t - state["last_adjust"]))
        cooldown_deg    = np.clip(1.0 - (cool_left / state["wait_time"]), 0.0, 1.0)

        # Narrowness degree
        width_prop      = np.clip(state["width_proportion"], 0.0, 1.0)
        if width_prop <= MIN_PROPORTION:
            narrowness_deg = 1.0
        else:
            narrowness_deg = np.clip((1.0 - width_prop) / (1.0 - MIN_PROPORTION), 0.0, 1.0)

        # Can-reset degree
        since_reset     = t - state["last_reset_time"]
        reset_deg       = np.clip(since_reset / state["reset_cooldown"], 0.0, 1.0)

        return np.array(
            [1.0, smi, stagnation_deg, cooldown_deg, narrowness_deg, reset_deg],
            dtype=float)

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def utility_func(t, best_f):
        """Compute utilities for each action based on the current state.
        """
        bf  = float(best_f.item())

        if state["q_w"] is None:
            rng                     = np.random.default_rng(SEED_VALUE)
            state["q_w"]            = rng.normal(0.0, 1e-3, size=(NUM_ACTIONS, 6))  # 6 features

        # Improvement since last step (minimisation)
        difference                  = max(0.0, state["prev_best_f"] - bf)
        denominator                 = EPS + abs(state["prev_best_f"])
        improvement                 = difference / denominator
        state["smoothed_improv"]    = (1 - ALPHA_IMPROV) * state["smoothed_improv"] + ALPHA_IMPROV * improvement

        # Update stagnation timer
        if bf < state["prev_best_f"] - MIN_IMPROVEMENT:
            state["t_stag_start"] = t
        state["prev_best_f"] = bf

        # Get the features
        features                    = _features_func(t)

        # TD(0) update of Q-values for the previous action (recorded in premotor)
        if state["prev_features"] is not None and state["prev_action"] is not None:
            # Compute Q-values for current state, boostrapped (i.e., max_a Q(s',a))
            q_next      = state["q_w"] @ features
            target      = improvement + GAMMA_Q * np.max(q_next)
            q_prev      = state["q_w"][state["prev_action"], :] @ state["prev_features"]
            td_error    = target - q_prev

            # Update weights for the previous action
            state["q_w"][state["prev_action"], :] += ALPHA_Q * td_error * state["prev_features"]

        # Compute current Q-values for all actions
        q_now   = state["q_w"] @ features

        # Softmax to get action probabilities
        logits     = (q_now - np.max(q_now)) / max(1e3 * EPS, SOFTMAX_TEMP)
        expvals    = np.exp(np.clip(logits, -50, 50))
        probs       = expvals / (np.sum(expvals) + EPS)

        # Store current features for next update
        state["prev_features"]  = features

        return probs.astype(float)

    utility_ens = nengo.Node(
        label       = "utility",
        output      = utility_func,
        size_in     = 1,
        size_out    = NUM_ACTIONS,
    )

    def premotor_func(t, action_vector):
        """Pre-motor function to modulate the motor ensemble based on the selected action.
        """
        if t < INIT_WAIT_TIME:
            return 1.0  # HOLD during initialisation

        action_values           = action_vector
        action_idx              = int(np.argmax(action_values))
        action                  = ACTIONS[action_idx]
        state["prev_action"]    = action_idx

        if action == "SHRINK":
            # smi         = state["smoothed_improv"]
            # base        = 0.35 + 0.45 * float(np.clip(1.0 - 10.0 * smi, 0.0, 1.0))
            # proportion  = float(np.clip(base, 0.05, 0.8))
            # proportion  = 0.15 + 0.85 * state["width_proportion"]
            proportion  = np.random.uniform(0.25, 0.5)
            # proportion  = 0.5
        elif action == "HOLD":
            proportion  = 1.0  # Keep the search space unchanged
        elif action == "RESET":
            proportion  = -1.0 # RESET to global bounds
        else:
            proportion  = 1.0  # Default to HOLD

        return proportion

    premotor_node = nengo.Node(
        label       = "premotor",
        output      = premotor_func,
        size_in     = NUM_ACTIONS,
        size_out    = 1,
    )

    # Scheduler function to trigger shrink operations
    def reframing_func(t, input_vector):
        # Read the current position and the action proportion
        current_v   = input_vector[:-1]
        proportion  = input_vector[-1]

        in_cooldown = (t - state["last_adjust"]) < state["wait_time"]
        if (state["best_v"] is None) or (t < INIT_WAIT_TIME) \
                or (proportion > 1.0 - MIN_PROPORTION) \
                or (in_cooldown and proportion >= 0.0): \
                # or (t < state["retarget_until"]):
            # HOLD action: do nothing
            return current_v

        if proportion < 0.0: # RESET action only if in Local mode
            state["width_proportion"]   = 1.0
            new_lb, new_ub              = X_LOWER_BOUND0.copy(), X_UPPER_BOUND0.copy()

            # Recompute best_v under the RESET bounds
            if state["best_x"] is not None:
                new_best_v              = tro2s(state["best_x"], new_lb, new_ub)
            else:
                new_best_v              = tro2s(X_INITIAL_GUESS, new_lb, new_ub)

            state["best_x"]             = trs2o(new_best_v, new_lb, new_ub)
            state["last_reset_time"]    = t

        else:
            proportion  = max(MIN_PROPORTION, min(1.0, proportion))

            # Current widths and target widths after shrinking
            half_width  = 0.5 * np.maximum(
                (state["ub"] - state["lb"]) * proportion,  MIN_WIDTH
            )

            # Centrer the new box around the current best (in original space)
            centre      = trs2o(state["best_v"], state["lb"], state["ub"])

            # Enforce minimum width and clip to global box
            new_lb      = np.maximum(centre - half_width, X_LOWER_BOUND0)
            new_ub      = np.minimum(centre + half_width, X_UPPER_BOUND0)

            # Ensure numerical ordering
            new_lb      = np.minimum(new_lb, new_ub - EPS)
            new_ub      = np.maximum(new_ub, new_lb + EPS)

            # Recompute best_v under the NEW bounds so it stays centred on the same best_x
            new_best_v  = tro2s(state["best_x"], new_lb, new_ub) if state["best_x"] is not None else state["best_v"]

            # Track the current width proportion
            state["width_proportion"]   = min(1.0, max(MIN_PROPORTION, state["width_proportion"] * proportion))

        state["last_adjust"]        = t

        # Update the local state
        state["lb"], state["ub"]    = new_lb, new_ub
        state["best_v"]             = np.clip(new_best_v, -1.0, 1.0)
        # state["retarget_until"]     = t + ACTION_DELAY

        return current_v

    # Scheduler node to trigger shrink operations
    reframing_node = nengo.Node(
        label       = "reframing",
        output      = reframing_func,
        size_in     = NUM_DIMENSIONS + 1,
        size_out    = NUM_DIMENSIONS,
    )

    # Selector node to keep track of the best-so-far
    def selector_func(t, x):
        v       = x[:NUM_DIMENSIONS]
        fv      = float(x[NUM_DIMENSIONS])

        if state["best_f"] is None or (fv < state["best_f"] and t >= 0.01):
            # Register the previous best
            # state["last_best_f"] = state["best_f"]

            # Update the best-so-far
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
        output      = selector_func,
        size_in     = NUM_DIMENSIONS + 1,
        size_out    = NUM_DIMENSIONS + 1,
    )

    # Selector / controller node to move towards the best-so-far
    def hs(t, x):
        if state["best_v"] is None:
            return np.zeros_like(x)

        in_cooldown = (t - state["last_adjust"]) < state["wait_time"]

        eta     = 0.8 if in_cooldown else 0.2
        sigma   = 0.01 if in_cooldown else 0.08

        return x + eta * (state["best_v"] - x) + np.random.normal(0, sigma, size=NUM_DIMENSIONS)

    hs_node     = nengo.Node(
        label       = "hs",
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = hs
    )


    # Pulser function to inject the initial guess at t=0 and best_v after shrinks
    def pulser_func(t):
        if t < 0.01:
            return v0_state.copy()
        # if t <= state["retarget_until"] and state["best_v"] is not None:
        #     return state["best_v"]
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


    # CONNECTIONS
    # --------------------------------------------------------------

    # [Pulser] --- v --> [EA input]
    nengo.Connection(
        pre         = pulser_node,
        post        = motor_ens.input,
        synapse     = None,
    )

    # [Utility] --- x --> [Basal ganglia]
    nengo.Connection(
        pre         = utility_ens,
        post        = basal_ganglia.input,
        synapse     = 0.01,
    )

    # [Basal ganglia] --- action --> [Thalamus]
    nengo.Connection(
        pre         = basal_ganglia.output,
        post        = thalamus.input,
        synapse     = 0.01,
    )

    # [Thalamus] --- action --> [Premotor]
    nengo.Connection(
        pre         = thalamus.output,
        post        = premotor_node,
        synapse     = 0.01,
    )

    # [Premotor] --- proportion --> [Reframing]
    nengo.Connection(
        pre         = premotor_node,
        post        = reframing_node[-1],
        synapse     = 0.01,
    )

    # [EA output] --- x --> [Selector]
    nengo.Connection(
        pre         = motor_ens.output,
        post        = hs_node,
        synapse     = 0,
    )

    # [Selector] --- x (delayed) --> [EA input]
    tau = 0.1
    nengo.Connection(
        pre         = hs_node,
        post        = motor_ens.input,
        synapse     = tau,
        transform   = np.eye(NUM_DIMENSIONS),
    )

    # [EA output] --- x --> [Objective function]
    nengo.Connection(
        pre         = motor_ens.output,
        post        = reframing_node[:-1],
        synapse     = 0,
    )

    # [Reframing] -> [Objective function]
    nengo.Connection(
        pre        = reframing_node,
        post       = obj_node,
        synapse    = 0,
    )

    # [Objective function] --- (x,f) --> [Selector]
    nengo.Connection(
        pre         = obj_node,
        post        = selector_node,
        synapse     = 0,
    )

    # [Selector] --- (fbest) ---> [Utility]
    nengo.Connection(
        pre         = selector_node[-1],
        post        = utility_ens,
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

    def _proportion_func(t):
        # if t < state["retarget_until"]:
        return state["width_proportion"]
        # return 1.0

    _proportion_node = nengo.Node(
        label       = "proportion",
        size_in     = 0,
        size_out    = 1,
        output      = _proportion_func,
    )

    # PROBES
    # --------------------------------------------------------------
    shrink_trigger  = nengo.Probe(_proportion_node, synapse=None)
    ens_lif_val     = nengo.Probe(motor_ens.output, synapse=0.01)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val       = nengo.Probe(fbest_only, synapse=0.01)
    xbest_val       = nengo.Probe(xbest_only, synapse=0.01)
    ea_spk          = [nengo.Probe(ens.neurons, synapse=0.01) for ens in motor_ens.ensembles]

    utility_vals    = nengo.Probe(utility_ens, synapse=0.01)

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

action_data = sim.data[utility_vals]

# plt.figure(figsize=(10, 5))

offsets = [0] * 3 #[-0.5, 0.0, 0.5]
for i, action_label in enumerate(ACTIONS):
    action_label = action_label
    plt.plot(sim.trange() + offsets[i], action_data[:, i], '-',
             label=action_label, markersize=2)
plt.xlim(offsets[0], SIMULATION_TIME)
plt.legend()
plt.title("Action utilities over time")
plt.show()

#%%
# Plot the spiking output of the ensemble
# plt.figure()
# spikes_cat = np.hstack([sim.data[p] for p in ea_spk])
# rasterplot(sim.trange(), spikes_cat) #, use_eventplot=True)
# plt.xlim(0, SIMULATION_TIME)
# plt.show()
