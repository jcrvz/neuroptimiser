#%%
import secrets

import matplotlib.pyplot as plt
import nengo
from collections import deque
from sortedcontainers import SortedList
import numpy as np
from ioh import get_problem
from nengo.utils.matplotlib import rasterplot, implot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 10  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_OBJS        = 1   # Number of objectives (only single-objective supported here)
NUM_DIMS        = 2  # Number of dimensions for the problem

NEURONS_PER_DIM = 20 * NUM_DIMS
NEURONS_PER_ENS = 100
SIMULATION_TIME = 50.0  # seconds

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMS)

# Keep copies of the original global bounds
X_LOWER_BOUND0  = X_LOWER_BOUND.copy()
X_UPPER_BOUND0  = X_UPPER_BOUND.copy()

# Exploitation (search-space shrinking) schedule
SEED_VALUE      = 69
# SAMPLING_TIME   = 1.0
NUM_FEATURES    = 2
STAG_WAIT       = 3.0
ALPHA_IMPROV    = 0.20
EPS             = 1e-12

TAU_0 = TAU_1 = TAU_2 = 0.005
TAU_3           = 0.01
LEAK            = 1.0

LAMBDA          = 100
MU              = max(4, LAMBDA // 2)
SPREAD_MIN      = 1e-6
SPREAD_MAX      = 0.5

THRESHOLD_V    = 1e-3

_DENOM         = np.sqrt(max(3, NUM_DIMS))
GAIN_1         = 5.0 / _DENOM
GAIN_2         = 1.0 / (0.5 * GAIN_1)
GAIN_CENTRE    = 5.0
ETA_S          = 0.3 / _DENOM
ETA_G          = 0.2 / _DENOM

#%%

# Create the Nengo model
model = nengo.Network(label="nNeurOpti V1", seed=69)
with model:
    # INITIALISATION
    # --------------------------------------------------------------
    # LOCAL STATE
    state = {
        "lb":               X_LOWER_BOUND.copy(),   # dynamic local bounds
        "ub":               X_UPPER_BOUND.copy(),
        "best_v":           X_INITIAL_GUESS.copy(),
        "prev_best_v":      X_INITIAL_GUESS.copy(),
        "best_f":           None,                   # best objective value
        "prev_best_f":      None,                   # previous best objective value
        "smoothed_improv":  1.0,                    # smoothed improvement metric
        "t_last_improv":    0.0,                    # time when stagnation started
        "archive":          SortedList(key=lambda _x: _x[1]),       # small archive to represent the sampling phase
        "sigma":            0.25,                    # current spread value
    }

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------

    # NOISE SOURCE: Generates perturbations
    def noise_func(t):
        _z  = np.random.normal(0.0, 1.0, NUM_DIMS)
        return np.clip(_z, -3.0, 3.0)

    noise_node = nengo.Node(
        label       = "Noise Source",
        output      = noise_func,
        size_out    = NUM_DIMS,
    )

    motor_noise     = nengo.Ensemble(
        label           = "Motor Noise",
        n_neurons       = NEURONS_PER_ENS,
        dimensions      = NUM_DIMS,
        radius          = 3.0,
    )

    # Initial conditions for motors
    init_centre     = nengo.Node(
        label       = "Init Centre",
        output      = lambda t: X_INITIAL_GUESS.copy() if t < 0.01 else np.zeros(NUM_DIMS),
        size_out    = NUM_DIMS,
    )
    init_spread = nengo.Node(
        label       = "Init Spread",
        output      = lambda t: np.log(0.25) * np.ones(NUM_DIMS) if t < 0.01 else np.zeros(NUM_DIMS),
        size_out    = NUM_DIMS,
    )

    # Centre control
    def centroid_func(t):
        if state["archive"]:
            mu      = max(1, min(MU, len(state["archive"]) // 2))
            elite   = np.array([v for v,_ in state["archive"][:mu]])
            ranks   = np.arange(1, mu+1)
            weights = np.log(mu + 0.5) - np.log(ranks)
            weights /= np.sum(weights)
            return weights @ elite
        return state["best_v"]

    centroid_node = nengo.Node(
        label       = "EW Centroid",
        output      = centroid_func,
        size_out    = NUM_DIMS,
    )

    # Error ensemble to compute error between best and centroid
    error_centre = nengo.Ensemble(
        label           = "Centre Error",
        n_neurons       = NEURONS_PER_ENS + 10 * NUM_DIMS,
        dimensions      = NUM_DIMS,
        radius          = 2.0,
    )

    # CENTRE Ensemble and control
    motor_centre  = nengo.Ensemble(
        label           = "Motor Centre",
        n_neurons       = NEURONS_PER_DIM,
        dimensions      = NUM_DIMS,
        radius          = 2.0,
    )

    # Scalar gain

    gain_bias = nengo.Node(
        label       = "Gain Bias",
        output      = 0.5 * GAIN_1 * np.ones(NUM_DIMS),
    )

    # SPREAD (log) Ensemble and control
    motor_logsigma = nengo.Ensemble(
        label           = "Motor Spread",
        n_neurons       = NEURONS_PER_DIM,
        dimensions      = NUM_DIMS,
        radius          = 1.5,
    )

    # Shrink when stagnation is detected
    shrink_gate = nengo.networks.Product(
        label       = "Shrink Gate",
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_DIMS,
    )

    # Grow when significant improvement is detected
    grow_gate = nengo.networks.Product(
        label       = "Grow Gate",
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_DIMS,
    )

    # COMPOSE NODE: Combine centre and spread into state vector
    mult_spread_noise   = nengo.networks.Product(
        label       = "Spread * Noise",
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_DIMS,
    )


    # Neural adder to combine centre and spread*noise
    candidate_vector = nengo.networks.EnsembleArray(
        label           = "Candidate vector",
        n_neurons       = NEURONS_PER_DIM,
        n_ensembles     = NUM_DIMS,
        ens_dimensions  = 1,
        radius          = 1.5,
    )

    # OBJECTIVE FUNCTION NODE: Evaluate the objective function
    def objective_func(t, variables):
        v   = np.clip(variables, -1.0, 1.0)
        x   = np.array(trs2o(v, state["lb"], state["ub"]))
        fv  = problem(x)
        return np.concatenate((v, [fv]))

    state["best_f"] = objective_func(0.0, tro2s(X_INITIAL_GUESS, state["lb"], state["ub"]))[-1]

    obj_node = nengo.Node(
        label       = "Problem",
        output      = objective_func,
        size_in     = NUM_DIMS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    # SELECTOR NODE: Keep track of best-so-far solution
    def selector_func(t, vf):
        # Extract candidate solution and its objective value
        v       = vf[:NUM_DIMS]
        fv      = float(vf[NUM_DIMS])

        # Archive update
        state["archive"].add((v.copy(), fv))

        # Keep archive size limited
        if len(state["archive"]) > LAMBDA:
            state["archive"].pop()

        # Update improvement metrics
        if state["best_f"] is None or (fv < state["best_f"]):
            state["prev_best_f"]    = state["best_f"] if state["best_f"] is not None else fv
            state["best_f"]         = fv

            state["best_v"]         = v.copy()

            state["t_last_improv"]   = t

        return np.concatenate((state["best_v"], [state["best_f"]]))

    selector_node   = nengo.Node(
        label       = "Selector",
        output      = selector_func,
        size_in     = NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    # FEATURES ENSEMBLE: Extract features from the current evaluation

    def _feat_success_rate(t):
        if len(state["archive"]) < 2:
            return 0.5

        f_values = [fv for _, fv in state["archive"]]
        median_f = np.median(f_values)
        num_success = np.sum(f_values < median_f)
        return num_success / len(f_values)

    def _feat_fitness_diversity(t):
        if len(state["archive"]) < 2:
            return 0.0

        mu      = max(2, min(MU, len(state["archive"]) // 2))
        elite_f = [fv for _, fv in state["archive"][:mu]]

        std_eli = np.std(elite_f)
        mean_eli= np.mean(elite_f) + EPS

        return std_eli / mean_eli

    def _improvement_rate(t):
        if state["prev_best_f"] is None:
            return 0.0

        delta_f     = max(0.0, state["prev_best_f"] - state["best_f"])
        rel_improv  = delta_f / (abs(state["prev_best_f"]) + EPS)

        time_since  = max(EPS, t - state["t_last_improv"])
        decay       = np.exp(-time_since / STAG_WAIT)
        return rel_improv * decay

    # Features input node
    def features_input_func(t):
        features = []
        for _feat in [_feat_success_rate, _improvement_rate]:
            feat_val = _feat(t)
            features.append(feat_val)

        return np.clip(features, 0.0, 1.0)

    features_input_node = nengo.Node(
        label       = "Features Input",
        output      = features_input_func,
        size_out    = NUM_FEATURES,
    )

    # Features ensemble to extract features for utility computation
    neuromodulator_ens = nengo.Ensemble(
        label       = "Neuromodulator",
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = NUM_FEATURES,
        radius      = 1.0,
    )


    # TODO: this could be a memory
    # [Aux] Nodes to extract only fbest and xbest from the selector output
    fbest_only = nengo.Node(
        size_in     =NUM_DIMS + NUM_OBJS,
        size_out    = 1,
        output      = lambda t, _x: _x[-1]
    )
    xbest_only = nengo.Node(
        size_in     =NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS,
        output      = lambda t, x: trs2o(x[:NUM_DIMS], state["lb"], state["ub"])
    )

    # INITIALISATION CONNECTIONS
    # --------------------------------------------------------------
    nengo.Connection(init_centre, motor_centre, synapse=TAU_0)
    nengo.Connection(init_spread, motor_logsigma, synapse=TAU_0)

    # CENTRE CONTROL LOOP
    # --------------------------------------------------------------
    # Error: centroid - centre
    nengo.Connection(centroid_node, error_centre, synapse=None, transform=np.eye(NUM_DIMS))
    nengo.Connection(motor_centre, error_centre, synapse=None, transform=-np.eye(NUM_DIMS))

    # Centre integrator and update
    nengo.Connection(motor_centre, motor_centre, synapse=TAU_3, transform=0.9 * np.eye(NUM_DIMS))
    nengo.Connection(error_centre, motor_centre, synapse=TAU_3, transform=GAIN_CENTRE * np.eye(NUM_DIMS))

    # SPREAD CONTROL LOOP
    # --------------------------------------------------------------
    # Leaky integrator
    nengo.Connection(motor_logsigma, motor_logsigma, synapse=TAU_3, transform=LEAK * np.eye(NUM_DIMS))

    # Grow (improvement-driven)
    nengo.Connection(gain_bias, grow_gate.A, synapse=None, transform=GAIN_2 * np.eye(NUM_DIMS))
    nengo.Connection(neuromodulator_ens[0], grow_gate.B, synapse=TAU_1, transform=ETA_G * np.ones((NUM_DIMS, 1)))
    nengo.Connection(grow_gate.output, motor_logsigma, synapse=TAU_3, transform=np.eye(NUM_DIMS))

    # Shrink (stagnation-driven)
    nengo.Connection(gain_bias, shrink_gate.A, synapse=None, transform=GAIN_2 * np.eye(NUM_DIMS))
    nengo.Connection(neuromodulator_ens[1], shrink_gate.B, synapse=TAU_1, transform=ETA_S * np.ones((NUM_DIMS, 1)),
                     function=lambda _x: 1.0 - np.clip(_x, 0.0, 1.0))
    nengo.Connection(shrink_gate.output, motor_logsigma, synapse=TAU_3, transform=-np.eye(NUM_DIMS))

    # CANDIDATE GENERATION
    # --------------------------------------------------------------
    # Noise and scale
    nengo.Connection(noise_node, motor_noise, synapse=TAU_1)
    nengo.Connection(motor_logsigma, mult_spread_noise.A, synapse=TAU_2,
                     function=lambda _x: np.clip(np.exp(_x), SPREAD_MIN, SPREAD_MAX))
    nengo.Connection(motor_noise, mult_spread_noise.B, synapse=TAU_2)

    # Compose candidate
    nengo.Connection(motor_centre, candidate_vector.input, synapse=TAU_2)
    nengo.Connection(mult_spread_noise.output, candidate_vector.input, synapse=TAU_2)

    # EVALUATION AND SELECTION
    # --------------------------------------------------------------
    nengo.Connection(candidate_vector.output, obj_node, synapse=None)
    nengo.Connection(obj_node, selector_node, synapse=None)

    # NEUROMODULATION (FEATURE INPUTS)
    # --------------------------------------------------------------
    nengo.Connection(features_input_node, neuromodulator_ens, synapse=TAU_2)

    # OUTPUTS
    # --------------------------------------------------------------
    nengo.Connection(selector_node, fbest_only, synapse=None)
    nengo.Connection(selector_node, xbest_only, synapse=None)

    # PROBES
    # --------------------------------------------------------------
    # ens_lif_val     = nengo.Probe(motor.output, synapse=0.01)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val       = nengo.Probe(fbest_only, synapse=0.01)
    xbest_val       = nengo.Probe(xbest_only, synapse=0.01)
    # ea_spk          = [nengo.Probe(ens.neurons, synapse=0.01) for ens in motor.ensembles]

    centre_val      = nengo.Probe(motor_centre,         synapse=0.01)
    spread_val      = nengo.Probe(motor_logsigma, synapse=0.01)
    features_val    = nengo.Probe(neuromodulator_ens,   synapse=0.01)

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
for i in range(NUM_DIMS):
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
plt.legend(loc="lower left")

ax2 = ax1.twinx()
ax2.plot(sim.trange(), np.average(sim.data[centre_val], axis=1), "g--", label="Centre")
ax2.plot(sim.trange(), np.exp(np.average(sim.data[spread_val], axis=1)), "m--", label="Spread")
ax2.plot(sim.trange(), sim.data[features_val][:,0], "c:", label="SMI")
ax2.plot(sim.trange(), sim.data[features_val][:,1], "y:", label="SGD")
ax2.set_ylabel("Centre / Spread values")
ax2.set_yscale("linear")
ax2.legend(loc="upper right")

plt.title(f"{problem.meta_data.name}-{problem.meta_data.instance}ins, {problem.meta_data.n_variables}D | "
          f"fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()

#%%
# Plot the decoded output of the ensemble
plt.figure(figsize=(12, 8), dpi=150)
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -5, 5, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, SIMULATION_TIME)
plt.xlabel("Time, s")
plt.ylabel(r"$x_{best}$ values")
plt.legend()
plt.show()

#%%
# # Plot the decoded output of the ensemble
# plt.figure(figsize=(12, 8), dpi=150)
# # plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -1, 1, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# # plt.plot(sim.trange(), sim.data[xbest_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# # plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
# plt.xlim(0, SIMULATION_TIME)
# plt.legend()
# plt.show()

#%%
# Plot the spiking output of the ensemble
# spikes_cat = []
# for i, p in enumerate(ea_spk):
#     spikes = sim.data[p]
#     # indices = sorted_neurons(motor_ens.ensembles[i], sim, iterations=250)
#     # spikes_cat.append(spikes[:, indices])
#     spikes_cat.append(spikes)
# spikes_cat = np.hstack(spikes_cat)


# plt.figure(figsize=(12, 8), dpi=150)
# spikes_cat = np.hstack([sim.data[p] for p in ea_spk])
# # rasterplot(sim.trange(), spikes_cat) #, use_eventplot=True)
# plt.imshow(spikes_cat.T, aspect="auto", cmap="hsv", origin="lower",)
#
# t_indices = plt.gca().get_xticks().astype(int)
# t_indices = t_indices[t_indices >= 0.0]
# t_indices[-1] -= 1
# t_labels = [f"{sim.trange()[i]:.1f}" for i in t_indices]
# plt.xticks(t_indices, t_labels)
#
# plt.xlabel("Time, s")
# plt.ylabel("Neuron")
# plt.title("Spikes")
# # plt.xlim(0, SIMULATION_TIME)
# plt.show()
