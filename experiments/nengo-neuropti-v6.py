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

PROBLEM_ID      = 12  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_DIMS        = 10  # Number of dimensions for the problem
NUM_OBJS        = 1   # Number of objectives (only single-objective supported here)

NEURONS_PER_DIM = 100
NEURONS_PER_ENS = 200
SIMULATION_TIME = 20.0  # seconds

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
ALPHA_IMPROV    = 0.8
EPS             = 1e-12
TAU             = 0.01

LAMBDA          = 64
MU              = LAMBDA // 2
SIGMA_MIN       = 1e-3
SIGMA_MAX       = 0.5
SPREAD_MAX      = 1.0
SPREAD_MIN      = 1e-6
D_EFF           = max(3, NUM_DIMS // 3)
PSUCC_TARGET    = 0.2
PSUCC_WINDOW    = 1.5


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
        "best_x":           tro2s(X_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND),
        "best_f":           None,                   # best objective value
        "prev_best_f":      None,                   # previous best objective value
        "smoothed_improv":  1.0,                    # smoothed improvement metric
        "t_last_improv":    0.0,                    # time when stagnation started
        "curr_features":    np.array([1.0, 0.0]),   # [smi, sgd]
        "archive":          deque(maxlen=LAMBDA),       # small archive to represent the sampling phase
        "archive_sorted":   None,                 # sorted archive
        "sampling_stage":   True,                 # whether we are in sampling stage
        "sigma":            0.25,                    # current spread value
    }

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------

    # MOTOR ENSEMBLE: Perturbation generator
    motor       = nengo.networks.EnsembleArray(
        label           = "Neural State",
        n_neurons       = NEURONS_PER_DIM,
        n_ensembles     = NUM_DIMS,
        ens_dimensions  = 1,
        radius          = 1.0,
        # intercepts      = nengo.dists.Uniform(-0.99, 0.99),
        # max_rates       = nengo.dists.Uniform(80,220),
        # encoders        = nengo.dists.UniformHypersphere(surface=True),
        # neuron_type     = nengo.LIF(),
        # seed=SEED_VALUE,
    )

    # CENTRE and SPREAD Ensembles
    centre_ens  = nengo.Ensemble(
        label       = "Centre",
        n_neurons   = NEURONS_PER_ENS + 20 * NUM_DIMS,
        dimensions  = NUM_DIMS,
        radius      = 1.5,
    )
    spread_ens = nengo.Ensemble(
        label       = "Spread",
        n_neurons   = NEURONS_PER_ENS + 20 * NUM_DIMS,
        dimensions  = NUM_DIMS,
        radius      = 1.5,
    )

    # COMPOSE NODE: Combine centre and spread into state vector
    def compose_func(t, data):
        perturbation    = data[:NUM_DIMS]
        centre          = data[NUM_DIMS:2 * NUM_DIMS]
        spread          = np.clip(data[2 * NUM_DIMS:], SPREAD_MIN, SPREAD_MAX)

        return np.clip(centre + spread * perturbation, -1.0, 1.0)

    compose_node = nengo.Node(
        label       = "Composition",
        size_in     = 3 * NUM_DIMS,
        size_out    = NUM_DIMS,
        output      = compose_func,
    )

    # OBJECTIVE FUNCTION NODE: Evaluate the objective function
    def objective_func(t, variables):
        v   = np.clip(variables, -1.0, 1.0)
        x   = np.array(trs2o(v, state["lb"], state["ub"]))
        fv  = problem(x)
        return np.concatenate((v, [fv]))

    obj_node = nengo.Node(
        label       = "Problem",
        output      = objective_func,
        size_in     = NUM_DIMS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    state["best_f"]     = objective_func(0.0, state["best_v"])[-1]


    # SELECTOR NODE: Keep track of best-so-far solution
    def selector_func(t, vf):
        v       = vf[:NUM_DIMS]
        fv      = float(vf[NUM_DIMS])

        # Update improvement metrics
        if state["best_f"] is None or (fv < state["best_f"]):
            state["prev_best_f"]    = state["best_f"] if state["best_f"] is not None else fv
            state["best_f"]         = fv

            state["best_v"]         = v.copy()
            state["best_x"]         = trs2o(state["best_v"], state["lb"], state["ub"])

            state["t_last_improv"]   = t
            # state["archive"].append((state["best_v"], state["best_f"]))


        return np.concatenate((state["best_v"], [state["best_f"]]))

    selector_node   = nengo.Node(
        label       = "Selector",
        output      = selector_func,
        size_in     = NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    # FEATURES NODE: Extract features for utility computation
    def features_func(t, vf):
        # Update archive with current sample
        v       = vf[:NUM_DIMS]
        fv      = float(vf[NUM_DIMS:NUM_DIMS + NUM_OBJS])

        # Update the current features based on improvement and stagnation
        if state["prev_best_f"] is None:
            state["prev_best_f"]    = state["best_f"]

        difff       = max(0.0, state["prev_best_f"] - state["best_f"])
        denom       = EPS + abs(state["prev_best_f"])
        improv      = difff / denom

        smi, sgd = state["curr_features"]

        sgd         = np.clip((t - state["t_last_improv"]) / STAG_WAIT, 0.0, 1.0)
        smi         = np.clip(ALPHA_IMPROV * smi + (1 - ALPHA_IMPROV) * improv, 0.0, 1.0)

        # If it is in stagnation, we consider the sampling is over, so we update other features
        if sgd >= 1.0:
            state["sampling_stage"]     = True
            state["archive"].append((v.copy(), fv))
        else:
            state["sampling_stage"]     = False
            state["archive_sorted"]     = sorted(state["archive"], key=lambda item: item[1])

        state["curr_features"]      = np.array([smi, sgd], dtype=float)

        return state["curr_features"]

    # Features node to extract features for utility computation
    features_node = nengo.Node(
        label       = "features",
        output      = features_func,
        size_in     = NUM_DIMS + NUM_OBJS,
        size_out    = NUM_FEATURES,
    )

    # CENTRE CONTROLLER NODE: Compute centre based on features
    def centre_control(t, x):
        smi, sgd = x

        if state["archive"]:
            A        = list(state["archive"])
            A_sorted = sorted(A, key=lambda item: item[1])
            mu       = max(1, min(MU, len(A_sorted)//2))
            v_bar    = np.mean([v for v, _ in A_sorted[:mu]], axis=0)
        else:
            v_bar    = best_v

        return v_bar

    centre_ctrl_node = nengo.Node(
        label       = "Centre Control",
        size_in     = NUM_FEATURES,
        size_out    = NUM_DIMS,
        output      = centre_control,
    )

    # SPREAD CONTROLLER NODE: Compute spread based on features
    def spread_control(t, x):
        smi, sgd = x

        # Anisotropic sigma: each dimension gets its own value based on features
        # Example: modulate sigma by SMI and SGD for each dimension
        s_star = np.clip(sigma * (1.0 + 0.5 * smi + 0.5 * sgd * np.arange(NUM_DIMS) / max(1, NUM_DIMS - 1)),
                         SPREAD_MIN, SPREAD_MAX)
        return s_star

    spread_ctrl_node = nengo.Node(
        label       = "Spread Control",
        size_in     = NUM_FEATURES,
        size_out    = NUM_DIMS,
        output      = spread_control,
    )

    # PERTURBATION CONTROLLER NODE: Compute perturbation based on features
    def perturbation_control(t, u):
        return np.random.normal(0.0, 1.0, size=NUM_DIMS)

    perturb_ctrl_node = nengo.Node(
        label       = "Perturbation Control",
        size_in     = NUM_DIMS,
        size_out    = NUM_DIMS,
        output      = perturbation_control,
    )

    # PULSER NODE: Clock signal to trigger new proposals
    # Pulser function to inject the initial guess at t=0 and best_v after shrinks
    def pulser_func(t):
        return state["best_v"] if t < 1e-3 else np.random.normal(0.0, 0.1, NUM_DIMS)

    # Pulser node to trigger actions
    pulser_node = nengo.Node(
        label       = "Trigger",
        output      = pulser_func,
        size_out    = NUM_DIMS,
    )

    # TODO: this could be a memory
    # [Aux] Nodes to extract only fbest and xbest from the selector output
    fbest_only = nengo.Node(
        size_in     =NUM_DIMS + NUM_OBJS,
        size_out    = 1,
        output      = lambda t, x: x[-1]
    )
    xbest_only = nengo.Node(
        size_in     =NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS,
        output      = lambda t, x: trs2o(x[:NUM_DIMS], state["lb"], state["ub"])
    )

    # CONNECTIONS
    # --------------------------------------------------------------

    # Motor loop
    nengo.Connection(pulser_node,       motor.input,                                    synapse=None)
    nengo.Connection(motor.output, perturb_ctrl_node, synapse=None)
    nengo.Connection(perturb_ctrl_node, motor.input, synapse=TAU)

    # Composition of centre, spread, and perturbation
    nengo.Connection(motor.output, compose_node[:NUM_DIMS], synapse=None)
    nengo.Connection(centre_ens, compose_node[NUM_DIMS:2 * NUM_DIMS], synapse=TAU)
    nengo.Connection(spread_ens, compose_node[2 * NUM_DIMS:], synapse=TAU)

    # Environment evaluation
    nengo.Connection(compose_node,              obj_node,                                       synapse=None)
    nengo.Connection(obj_node,                  selector_node,                                  synapse=None)
    nengo.Connection(obj_node,                  features_node,                                  synapse=None)

    # Centre control
    nengo.Connection(features_node, centre_ctrl_node, synapse=TAU)
    nengo.Connection(centre_ctrl_node, centre_ens, synapse=None)
    # nengo.Connection(centre_ens, centre_ens, synapse=TAU, transform=0.95*np.eye(NUM_DIMS))

    # Spread control
    nengo.Connection(features_node, spread_ctrl_node, synapse=TAU)
    nengo.Connection(spread_ctrl_node, spread_ens, synapse=None)
    # nengo.Connection(spread_ens, spread_ens, synapse=TAU, transform=0.95*np.eye(NUM_DIMS))

    # Outputs
    nengo.Connection(selector_node, fbest_only, synapse=None)
    nengo.Connection(selector_node, xbest_only, synapse=None)

    # PROBES
    # --------------------------------------------------------------
    ens_lif_val     = nengo.Probe(motor.output, synapse=0.01)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val       = nengo.Probe(fbest_only, synapse=0.01)
    xbest_val       = nengo.Probe(xbest_only, synapse=0.01)
    ea_spk          = [nengo.Probe(ens.neurons, synapse=0.01) for ens in motor.ensembles]

    centre_val      = nengo.Probe(centre_ctrl_node, synapse=0.01)
    spread_val      = nengo.Probe(spread_ctrl_node, synapse=0.01)
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
ax2.plot(sim.trange(), np.average(sim.data[spread_val], axis=1), "m--", label="Spread")
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
