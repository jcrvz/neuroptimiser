#%%
import matplotlib.pyplot as plt
import nengo
from sortedcontainers import SortedList
import numpy as np
from ioh import get_problem

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 10  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_OBJS        = 1   # Number of objectives (only single-objective supported here)
NUM_DIMS        = 2  # Number of dimensions for the problem

NEURONS_PER_DIM = 20 * NUM_DIMS
NEURONS_PER_ENS = 100
SIMULATION_TIME = 1.0  # seconds

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMS)
V_INITIAL_GUESS = tro2s(X_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
F_INITIAL_GUESS = problem(X_INITIAL_GUESS)

def eval_obj_func(v_scaled):
    # Transform from scaled to original
    x_orig  = trs2o(v_scaled, X_LOWER_BOUND, X_UPPER_BOUND)
    f_val   = problem(x_orig)
    return f_val

# Exploitation (search-space shrinking) schedule
EPS             = 1e-12

TAU_0           = 0.001
TAU_3           = 0.005

TAU_IMPROV_FAST = 0.001  # 5 ms
TAU_IMPROV_SLOW = 0.002  # 50 ms
INIT_TIME       = 0.001   # time to initialise the optimiser
K_IMPROV        = 10.0
P_STAR          = 0.5   # target success rate (CSA-inspired, 1/5 rule)
ETA_SIG         = 1.0 #0.3 / np.sqrt(max(3, NUM_DIMS))

LEAK            = 1.0

LAMBDA          = 50
MU              = max(4, LAMBDA // 2)
SIG_MAX         = 0.5
SIG_MIN         = 1e-12
LOGSIG_MAX      = np.log(SIG_MAX)
LOGSIG_MIN      = np.log(SIG_MIN)
SUMLOGSIG       = LOGSIG_MAX + LOGSIG_MIN
DIFLOGSIG       = LOGSIG_MAX - LOGSIG_MIN

EXPLOIT_WIN     = 0.01  # s

THRESHOLD_V    = 1e-3

GAIN_1         = 1.0

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
        "best_v":           V_INITIAL_GUESS.copy(),
        "prev_best_v":      V_INITIAL_GUESS.copy(),
        "best_f":           F_INITIAL_GUESS,                   # best objective value
        "prev_best_f":      F_INITIAL_GUESS,                   # previous best objective value
        "smoothed_improv":  1.0,                    # smoothed improvement metric
        "archive":          SortedList(key=lambda _x: _x[1]),       # small archive to represent the sampling phase
        "t_last_improv":   0.0,                    # time of last improvement
        "sigma":            0.25,                    # current spread value
    }

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------
    # def improve_gate_func(t, fvals):
    #     # Extract best and previous best objective values
    #     curr_f      = fvals[0]
    #     best_f      = fvals[1]
    #
    #     if best_f - curr_f > EPS:
    #         return 1.0
    #
    #     return 0.0
    #
    # improve_gate_node = nengo.Node(
    #     label       = "Improve Gate",
    #     output      = improve_gate_func,
    #     size_in     = 2 * NUM_OBJS,
    #     size_out    = 1,
    # )

    # Gate noise by the improvement pulse: B_effective = (1 - gate) * noise
    def _noise_gate_func(t, feature):
        if (t - state["t_last_improv"]) < EXPLOIT_WIN or t < INIT_TIME:
            return np.zeros(NUM_DIMS)

        return np.random.normal(0.0, 1.0, NUM_DIMS)

    noise_gate = nengo.Node(
        label       = "Noise Gate",
        output      = _noise_gate_func,
        size_in     = 1,
        size_out    = NUM_DIMS,
    )

    # Spread target node
    def spread_target_func(t, _improv_rates):
        if t < INIT_TIME:
            return np.ones(NUM_DIMS)

        if (t - state["t_last_improv"]) < EXPLOIT_WIN:
            return -1.0 * np.ones(NUM_DIMS)
        spreads = np.clip(_improv_rates, 0.0, 1.0)

        # Adapt spread based on improvement rate
        amgis = 1.0 - 2.0 * spreads

        return amgis * np.ones(NUM_DIMS)


    spread_target_node = nengo.Node(
        label       = "Spread Target",
        output      = spread_target_func,
        size_in     = NUM_DIMS,
        size_out    = NUM_DIMS,
    )

    def _sigmoid_normalise(x):
        x = np.clip(x, 0.0, 1.0)

        # x = 1.0 if x >= 1e-3 else x

        return 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))

    # Centre target node
    def centre_target_func(t, improv_rates):
        if t < INIT_TIME:
            return V_INITIAL_GUESS.copy()

        if not state["archive"] or (t - state["t_last_improv"]) < EXPLOIT_WIN:
            return state["best_v"]

        # Compute the centroid of the top-mu solutions
        mu          = max(2, min(MU, len(state["archive"]) // 2))
        elite       = np.array([v for v,_ in state["archive"][:mu]])
        centroid   = np.mean(elite, axis=0)

        # Blend between best-so-far and centroid based on success rate
        # return improv_rate * state["best_v"] + (1.0 - improv_rate) * centroid
        return centroid

    centre_target_node = nengo.Node(
        label       = "Centre Target",
        output      = centre_target_func,
        size_in     = NUM_DIMS,
        size_out    = NUM_DIMS,
    )

    # CENTRE CONTROL LOOP
    motor_centre = nengo.Ensemble(
        label           = "Motor Centre",
        n_neurons       = NEURONS_PER_ENS,
        dimensions      = NUM_DIMS,
        radius          = 2.0,
    )

    # Amgis Ensemble and control: amgis in [-1, 1] -> |ln(sigma)| in [0.0, 12.0] -> ln(sigma) in [-12.0, 0.0] | sigma in [1e-12, 1.0]
    motor_amgis = nengo.Ensemble(
        label           = "Motor Spread",
        n_neurons       = NEURONS_PER_DIM,
        dimensions      = NUM_DIMS,
        radius          = 2.0,
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
        if t < INIT_TIME:
            v   = V_INITIAL_GUESS.copy()
            fv  = eval_obj_func(v)
            return  np.concatenate((v, [fv]))

        v   = np.clip(variables, -1.0, 1.0)
        fv  = eval_obj_func(v)
        return np.concatenate((v, [fv]))

    obj_node = nengo.Node(
        label       = "Problem",
        output      = objective_func,
        size_in     = NUM_DIMS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    # SELECTOR NODE: Keep track of best-so-far solution
    def selector_func(t, vf):
        if t < INIT_TIME:
            return np.concatenate((state["best_v"], [state["best_f"]]))

        # Extract candidate solution and its objective value
        v       = vf[:NUM_DIMS]
        fv      = float(vf[NUM_DIMS])

        # Archive update
        state["archive"].add((v.copy(), fv))

        # Keep archive size limited
        if len(state["archive"]) > LAMBDA:
            state["archive"].pop()

        # Update improvement metrics
        if fv < state["best_f"]:
            state["prev_best_f"]    = state["best_f"] if state["best_f"] is not None else fv
            state["best_v"]         = v.copy()
            state["best_f"]         = fv


        return np.concatenate((state["best_v"], [state["best_f"]]))

    selector_node   = nengo.Node(
        label       = "Selector",
        output      = selector_func,
        size_in     = NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    # FEATURES ENSEMBLE: Extract features from the current evaluation

    # def _feat_stagnation_rate(t):
    #     if state["prev_best_f"] is None:
    #         return 0.0
    #     time_since  = max(EPS, t - state["t_last_improv"])
    #     stag_rate   = (time_since / STAG_WAIT) ** 3
    #     return stag_rate

    # def _feat_success_rate(t):
    #     if len(state["archive"]) < 2:
    #         return 0.5
    #
    #     f_values = [fv for _, fv in state["archive"]]
    #     median_f = np.median(f_values)
    #     num_success = np.sum(f_values < median_f)
    #     return num_success / len(f_values)

    # def _feat_fitness_diversity(t):
    #     if len(state["archive"]) < 2:
    #         return 0.0
    #
    #     mu      = max(2, min(MU, len(state["archive"]) // 2))
    #     elite_f = [fv for _, fv in state["archive"][:mu]]
    #
    #     std_eli = np.std(elite_f)
    #     mean_eli= np.mean(elite_f) + EPS
    #
    #     return std_eli / mean_eli

    # Features node
    def features_func(t, _diff_vf):
        if t < INIT_TIME:
            return np.ones(NUM_DIMS)

        # _curr_prev_vf = [curr_v, curr_f, prev_v, prev_f]
        _diffv  = _diff_vf[:NUM_DIMS]
        _difff  = _diff_vf[NUM_DIMS: NUM_DIMS + NUM_OBJS]

        # If improvement occurred, update state
        if _difff > 0.0:
            state["t_last_improv"] = t
            # fast growth towards 1.0 when improvement occurs
            val_f   = 1.0 #0.1 + 0.9 * (1.0 - np.exp(-50.0 * _rel_improf))
        else:
            val_f   = 0.0

        # Compute difference in solution vectors
        vals_v       = val_f * _diffv

        # Assemble features
        # vals        = np.concatenate((val_v, [val_f]))

        # return features
        return np.clip(vals_v, -1.0, 1.0)

    features_input_node = nengo.Node(
        label       = "Features Gate",
        output      = features_func,
        size_in     = NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS,
    )

    init_node = nengo.Node(
        label       = "Initialiser",
        output      = lambda t: np.ones(NUM_DIMS) if t < INIT_TIME else np.zeros(NUM_DIMS),
    )

    # Features ensemble to extract features for utility computation
    neuromodulator_ens = nengo.networks.EnsembleArray(
        label       = "Neuromodulator",
        n_neurons   = NEURONS_PER_ENS,
        n_ensembles = NUM_DIMS,
        ens_dimensions = 1,
        radius      = 1.1,
    )

    # INITIALISATION CONNECTIONS
    # --------------------------------------------------------------
    nengo.Connection(init_node, centre_target_node[:NUM_DIMS], synapse=None)
    nengo.Connection(init_node, spread_target_node[:NUM_DIMS], synapse=None)
    nengo.Connection(init_node, neuromodulator_ens.input, synapse=None)

    # CENTRE CONTROL LOOP
    # --------------------------------------------------------------
    # Compute error between target and current centre
    nengo.Connection(neuromodulator_ens.output, centre_target_node[:NUM_DIMS], synapse=0.0)
    nengo.Connection(neuromodulator_ens.output, spread_target_node[:NUM_DIMS], synapse=0.0)

    nengo.Connection(centre_target_node, motor_centre, synapse=None)
    nengo.Connection(spread_target_node, motor_amgis, synapse=None,
                     transform=np.eye(NUM_DIMS))

    # CANDIDATE GENERATION
    # --------------------------------------------------------------
    # Noise and scale

    def amgisIIsigma(_amgis):
        _amgis          = np.clip(_amgis, -1.0, 1.0)
        _log_sigma  = 0.5 * (DIFLOGSIG * _amgis + SUMLOGSIG)
        return np.clip(np.exp(_log_sigma), SIG_MIN, SIG_MAX)

    nengo.Connection(motor_amgis, mult_spread_noise.A, synapse=None,
                     function=amgisIIsigma)
    nengo.Connection(neuromodulator_ens.output[-1], noise_gate, synapse=None)
    nengo.Connection(noise_gate,              mult_spread_noise.B, synapse=None)


    # Compose candidate
    nengo.Connection(motor_centre, candidate_vector.input, synapse=None)
    nengo.Connection(mult_spread_noise.output, candidate_vector.input, synapse=None)

    # EVALUATION AND SELECTION
    # --------------------------------------------------------------
    nengo.Connection(candidate_vector.output, obj_node, synapse=None)
    nengo.Connection(obj_node, selector_node, synapse=None)

    # nengo.Connection(obj_node[-1], improve_gate_node[0], synapse=0.0)
    # nengo.Connection(selector_node[-1], improve_gate_node[1], synapse=0.0)

    # NEUROMODULATION (FEATURE INPUTS)
    # --------------------------------------------------------------
    nengo.Connection(selector_node, features_input_node,
                     synapse=TAU_IMPROV_SLOW, transform=+K_IMPROV)
    nengo.Connection(selector_node, features_input_node,
                     synapse=TAU_IMPROV_FAST, transform=-K_IMPROV)

    # Keep stagnation/other feature on channel 1
    nengo.Connection(features_input_node, neuromodulator_ens.input, synapse=None)

    # PROBES
    # --------------------------------------------------------------
    # ens_lif_val     = nengo.Probe(motor.output, synapse=0.01)  # 10ms filter
    obj_val         = nengo.Probe(obj_node[-1],         synapse=0.0)
    fbest_val       = nengo.Probe(selector_node[-1],    synapse=0.0)
    vbest_val       = nengo.Probe(selector_node[:NUM_DIMS], synapse=0.0)

    centre_val      = nengo.Probe(motor_centre,         synapse=0.0)
    spread_val      = nengo.Probe(mult_spread_noise.A,       synapse=0.0)
    # features_val    = nengo.Probe(neuromodulator_ens,   synapse=0.0)
    features_val    = nengo.Probe(features_input_node,   synapse=0.0)

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

vbest_values    = sim.data[vbest_val]

xbest   = np.array([trs2o(vbest_values[i,:], X_LOWER_BOUND, X_UPPER_BOUND) for i in range(vbest_values.shape[0])])

for i in range(NUM_DIMS):
    print(f" x{i+1:02d}: [{np.min(xbest[:,i]):.4f}, {np.max(xbest[:,i]):.4f}], ", end="")
    print(f"{xbest[-1,i]:7.4f} :: {problem.optimum.x[i]:7.4f} | {xbest[-1,i] - problem.optimum.x[i]:7.4f}")

print(f"fbest: {sim.data[fbest_val][-1][0]:.4f} (opt: {problem.optimum.y}), diff: {sim.data[fbest_val][-1][0] - problem.optimum.y:.4f}")
#%%
x = xbest[-1, :]
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
plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value error", alpha=0.3)
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far error")
# plt.hlines(problem.optimum.y, 0, simulation_time, colors="r", linestyles="dashed", label="Optimal value")
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")

# when an ensemble
centre_values   = sim.data[centre_val]
# when a list of probes
# centre_values   = np.column_stack([sim.data[centre_val[i]] for i in range(NUM_DIMS)])

centre_scaled = np.clip(centre_values, -1.0, 1.0)
f_centre = np.array([eval_obj_func(centre_scaled[i,:]) for i in range(centre_scaled.shape[0])])

# plt.plot(sim.trange(), f_centre - problem.optimum.y, "g", marker=".", markersize=1, alpha=0.25)
f_centre_smooth = np.convolve(f_centre, np.ones(100)/100, mode='same')
plt.plot(sim.trange(), f_centre_smooth - problem.optimum.y, "g--", label="Centre error")

# add second axis for centre and spread
ax1 = plt.gca()
plt.xlim(0 - INIT_TIME, SIMULATION_TIME)
plt.xlabel("Time, s")
plt.ylabel("Objective value error")
plt.yscale("log")
plt.legend(loc="lower left")

ax2 = ax1.twinx()
# ax2.plot(sim.trange(), np.average(sim.data[centre_val], axis=1), "g--", label="Centre")
# ax2.plot(sim.trange(), np.exp(np.average(sim.data[spread_val], axis=1)), "m--", label="Spread")

colors_1 = plt.cm.rainbow(np.linspace(0, 1, NUM_DIMS))
for i in range(NUM_DIMS):
    improv_rate = sim.data[features_val][:,i]
    # succes_rate = sim.data[features_val][:,1]
    ax2.plot(sim.trange(), improv_rate, color=colors_1[i], marker=".", markersize=1, linestyle="", alpha=0.2)
    ax2.plot(sim.trange(), np.convolve(improv_rate, np.ones(100)/100, mode='same'), linestyle="dashed", color=colors_1[i], label=f"Dim {i+1}")
    # ax2.plot(sim.trange(), succes_rate, "r", marker=".", markersize=1, linestyle="", alpha=0.2)
    # ax2.plot(sim.trange(), np.convolve(succes_rate, np.ones(100)/100, mode='same'), "r--", label="Stag. Rate")


ax2.set_ylabel("Improvement rate")
ax2.set_yscale("linear")
ax2.legend(loc="upper right")

plt.title(f"{problem.meta_data.name}-{problem.meta_data.instance}ins, {problem.meta_data.n_variables}D | "
          f"fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()

#%%
# Plot the decoded output of the ensemble
fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=150, sharex=True)

ax1 = axs[0]
# plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i+1}" for i in range(NUM_DIMS)])
# plt.vlines(sim.data[shrink_trigger].nonzero()[0]*sim.dt, -5, 5, colors="grey", linestyles="dashed", label="Shrink", alpha=0.2)
ax1.plot(sim.trange(), xbest, label=[f"x{i + 1}" for i in range(NUM_DIMS)])

for i in range(NUM_DIMS):
    ax1.hlines(problem.optimum.x[i], 0, SIMULATION_TIME, colors="k", linestyles="dashed", label=f"xopt" if i==0 else None)

# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
ax1.set_xlim(0, SIMULATION_TIME)
ax1.set_ylabel(r"$x_{best}$ values")

ax2 = axs[1]
ax3 = axs[2]
# Color lines for each dimension
# colors_2 = plt.cm.vanimo(np.linspace(0, 1, NUM_DIMS))
num_samples = centre_values.shape[0]
every_n = max(1, num_samples // 50)
for i in range(NUM_DIMS):
    ax2.plot(sim.trange(), centre_values[:, i], color=colors_1[i], linestyle="", marker=".", markersize=1, alpha=0.2,)
    ax2.plot(sim.trange(), np.convolve(centre_values[:,i], np.ones(100)/100, mode='same'), alpha=1.0,
             color=colors_1[i], linestyle="dashed", marker="o", markevery=every_n, label=f"Centre dim {i+1}")
    ax3.plot(sim.trange(), sim.data[spread_val][:, i], color=colors_1[i], linestyle="", marker=".", markersize=1, alpha=0.2,)
    ax3.plot(sim.trange(), np.convolve(sim.data[spread_val][:,i], np.ones(100)/100, mode='same'), alpha=1.0,
             color=colors_1[i], linestyle="dashed", marker="x", markevery=every_n, label=f"Spread dim {i + 1}")

ax2.set_ylabel("Centre value")
ax3.set_ylabel("Spread value")
ax2.set_yscale("linear")
ax3.set_yscale("log")

ax3.set_xlabel("Time, s")

ax1.legend(loc="upper right", framealpha=0.9)
# ax2.legend(loc="upper right", framealpha=0.9)

plt.show()
