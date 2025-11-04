#%%
import matplotlib.pyplot as plt
import nengo
from sortedcontainers import SortedList
import numpy as np
from ioh import get_problem

# Transformations
from neuroptimiser.utils import tro2s, trs2o

#%%

PROBLEM_ID      = 1  # Problem ID from the IOH framework
PROBLEM_INS     = 1  # Problem instance
NUM_OBJS        = 1   # Number of objectives (only single-objective supported here)
NUM_DIMS        = 10  # Number of dimensions for the problem

NEURONS_PER_DIM = 100 * NUM_DIMS
NEURONS_PER_ENS = 200
SIMULATION_TIME = 20.0  # seconds

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

TAU_IMPROV_FAST = 0.002  # 5 ms
TAU_IMPROV_SLOW = 0.010  # 50 ms
INIT_TIME       = 0.001   # time to initialise the optimiser
K_IMPROV        = 10.0
P_STAR          = 0.5   # target success rate (CSA-inspired, 1/5 rule)
ETA_SIG         = 1.0 #0.3 / np.sqrt(max(3, NUM_DIMS))

LEAK            = 1.0

LAMBDA          = 50
MU              = max(4, LAMBDA // 2)
SIG_MAX         = 0.5
SIG_MIN         = 1e-6
LOGSIG_MAX      = np.log(SIG_MAX)
LOGSIG_MIN      = np.log(SIG_MIN)
SUMLOGSIG       = LOGSIG_MAX + LOGSIG_MIN
DIFLOGSIG       = LOGSIG_MAX - LOGSIG_MIN

EXPLOIT_WIN     = 1.0  # s
# EXPLORE_WIN     = 2.0  # s
TAU_BLEND       = EXPLOIT_WIN / 4

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
        "best_v":           None,
        "prev_best_v":      None,
        "best_f":           None,                   # best objective value
        "prev_best_f":      None,                   # previous best objective value
        "smoothed_improv":  1.0,                    # smoothed improvement metric
        "archive":          SortedList(key=lambda _x: _x[1]),       # small archive to represent the sampling phase
        "mode":             "explore",             # current mode: explore / exploit
        "t_mode_switch":    0.0,                    # time of last improvement
        "sigma":            0.25,                    # current spread value
    }

    # DEFINE THE MAIN COMPONENTS
    # --------------------------------------------------------------
    def _mode_update_func(t, _feature):
        if t < INIT_TIME:
            state["mode"] = "explore"
            state["t_mode_switch"] = t
            return 0.0

        time_since_switch = t - state["t_mode_switch"]

        if state["mode"] == "explore":
            # Only switch TO exploit when feature triggers
            if _feature > 0.0:  # Threshold for significant improvement
                state["mode"] = "exploit"
                state["t_mode_switch"] = t
                return 1.0
            return 0.0


        else:  # mode == "exploit"
            if time_since_switch > EXPLOIT_WIN:
                state["mode"] = "explore"
                state["t_mode_switch"] = t

                # OXYGENATE: Keep top 50%, inject 25% random
                if len(state["archive"]) > MU:
                    keep_size = len(state["archive"]) // 2
                    elite = list(state["archive"][:keep_size])
                    state["archive"].clear()

                    # Keep elite
                    for item in elite:
                        state["archive"].add(item)
                    # Inject random samples for diversity
                    for _ in range(keep_size // 2):
                        random_v = np.random.uniform(-0.7, 0.7, NUM_DIMS)
                        random_f = eval_obj_func(random_v)
                        state["archive"].add((random_v, random_f))
                return 0.0
            return 1.0


    mode_gate_node = nengo.Node(
        label       = "Mode Gate",
        output      = _mode_update_func,
        size_in     = 1,
        size_out    = 1,
    )

    # Gate noise by the improvement pulse: B_effective = (1 - gate) * noise
    def _noise_gate_func(t, feature):
        if t < INIT_TIME:
            return np.zeros(NUM_DIMS)

        if state["mode"] == "exploit":
            if len(state["archive"]) >= 4:
                k = min(MU, len(state["archive"]))
                elite = np.array([v for v, _ in state["archive"][:k]])

                mean = np.mean(elite, axis=0)
                centered = elite - mean
                cov = (centered.T @ centered) / (k - 1)

                # Stronger regularization: blend with isotropic
                iso_cov = 0.1 * np.eye(NUM_DIMS)  # 10% isotropic component
                cov = 0.7 * cov + 0.3 * iso_cov  # 70/30 blend

                try:
                    L = np.linalg.cholesky(cov)
                    return L @ np.random.normal(0.0, 1.0, NUM_DIMS)
                except np.linalg.LinAlgError:
                    return 0.3 * np.random.normal(0.0, 1.0, NUM_DIMS)  # Larger fallback
            else:
                return 0.3 * np.random.normal(0.0, 1.0, NUM_DIMS)

        # Exploration: isotropic
        return np.random.normal(0.0, 1.0, NUM_DIMS)

    noise_gate = nengo.Node(
        label       = "Noise Gate",
        output      = _noise_gate_func,
        size_in     = 1,
        size_out    = NUM_DIMS,
    )

    # Spread target node
    def spread_target_func(t, _improv_rate):
        if t < INIT_TIME:
            return np.ones(NUM_DIMS)

        if _improv_rate > 0.0:
            return -1.0 * np.ones(NUM_DIMS)

        spreads = np.clip(_improv_rate, 0.0, 1.0)

        # Adapt spread based on improvement rate
        amgis = 1.0 - 2.0 * spreads ** 2

        return amgis * np.ones(NUM_DIMS)


    spread_target_node = nengo.Node(
        label       = "Spread Target",
        output      = spread_target_func,
        size_in     = 1,
        size_out    = NUM_DIMS,
    )

    def _sigmoid_normalise(x):
        x = np.clip(x, 0.0, 1.0)

        # x = 1.0 if x >= 1e-3 else x

        return 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))

    # Centre selection
    def centre_selector_func(t, vv):
        exploit_centre = vv[:NUM_DIMS]
        explore_centre = vv[NUM_DIMS:]

        time_in_mode = t - state["t_mode_switch"]
        alpha = np.clip(time_in_mode / TAU_BLEND, 0.0, 1.0)

        if state["mode"] == "exploit":
            # Blend FROM explore TO exploit
            return (1.0 - alpha) * explore_centre + alpha * exploit_centre
        else:
            # Stay at explore centre
            return explore_centre


    centre_selector_node = nengo.Node(
        label       = "Centre Selector",
        output      = centre_selector_func,
        size_in     = 2 * NUM_DIMS,
        size_out    = NUM_DIMS,
    )

    # Centre for exploitation node
    def centre_exploit_func(t):
        if state["best_v"] is None:
            return V_INITIAL_GUESS.copy()
        return state["best_v"]

    centre_exploit_node = nengo.Node(
        label       = "Centre for Exploitation",
        output      = centre_exploit_func,
        # size_in     = 1,
        size_out    = NUM_DIMS,
    )

    # Centre for exploration node
    def centre_explore_func(t, improv_rate):
        if t < INIT_TIME or not state["archive"]:
            return V_INITIAL_GUESS.copy()

        k = max(2, min(MU, len(state["archive"]) // 2))
        elite = np.array([v for v, _ in state["archive"][:k]])
        centroid = np.mean(elite, axis=0)

        # OXYGENATE: Stronger, longer-lasting perturbation
        if state["mode"] == "explore":
            time_since_switch = t - state["t_mode_switch"]
            mutation_strength = 0.6 * np.exp(-time_since_switch / 2.0)  # 60%, 2s decay
            mutation = np.random.uniform(-mutation_strength, mutation_strength, NUM_DIMS)
            centroid = np.clip(centroid + mutation, -1.0, 1.0)

        weight = np.clip(improv_rate, 0.0, 1.0)
        return weight * centroid + (1.0 - weight) * V_INITIAL_GUESS

    centre_explore_node = nengo.Node(
        label       = "Centre for Exploration",
        output      = centre_explore_func,
        size_in     = 1,
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
        # if t < INIT_TIME:
        #     v   = V_INITIAL_GUESS.copy()
        #     fv  = eval_obj_func(v)
        #     return  np.concatenate((v, [fv]))

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
        v = vf[:NUM_DIMS]
        fv = float(vf[NUM_DIMS])

        state["archive"].add((v.copy(), fv))
        if len(state["archive"]) > LAMBDA:
            state["archive"].pop()

        # Initialize on first evaluation
        if state["best_f"] is None or fv < state["best_f"]:
            state["prev_best_f"] = state["best_f"]
            state["best_v"] = v.copy()
            state["best_f"] = fv

        return np.concatenate((state["best_v"], [state["best_f"]]))


    selector_node   = nengo.Node(
        label       = "Selector",
        output      = selector_func,
        size_in     = NUM_DIMS + NUM_OBJS,
        size_out    = NUM_DIMS + NUM_OBJS,
    )

    # Features node
    def features_func(t, _difff):
        feature_value = np.clip(_difff, 0.0, 1.0)

        # If improvement occurs DURING exploitation, reset timer
        if feature_value > 0.0 and state["mode"] == "exploit":
            state["t_mode_switch"] = t  # Extend exploitation window

        return feature_value


    features_input_node = nengo.Node(
        label       = "Features Gate",
        output      = features_func,
        size_in     = NUM_OBJS,
        size_out    = 1,
    )

    # init_node = nengo.Node(
    #     label       = "Initialiser",
    #     output      = lambda t: 1.0 if t < INIT_TIME else 0.0,
    # )

    # Features ensemble to extract features for utility computation
    neuromodulator_ens = nengo.Ensemble(
        label       = "Neuromodulator",
        n_neurons   = NEURONS_PER_ENS,
        dimensions  = 1,
        radius      = 1.1,
    )

    # CONNECT THE COMPONENTS
    # --------------------------------------------------------------


    # CENTRE CONTROL LOOP
    # --------------------------------------------------------------
    nengo.Connection(centre_exploit_node, centre_selector_node[:NUM_DIMS], synapse=0.0)
    nengo.Connection(centre_explore_node, centre_selector_node[NUM_DIMS:], synapse=0.0)

    # Compute error between target and current centre
    nengo.Connection(neuromodulator_ens, centre_explore_node, synapse=0.0)
    nengo.Connection(mode_gate_node, spread_target_node[:NUM_DIMS], synapse=0.0)

    nengo.Connection(centre_selector_node, motor_centre, synapse=None)
    nengo.Connection(spread_target_node, motor_amgis, synapse=None,
                     transform=np.eye(NUM_DIMS))

    # CANDIDATE GENERATION
    # --------------------------------------------------------------
    # Noise and scale

    # Direct sigma control
    motor_sigma = nengo.Ensemble(
        label="Motor Sigma",
        n_neurons=NEURONS_PER_DIM,
        dimensions=NUM_DIMS,
        radius=SIG_MAX,
        encoders=nengo.dists.Choice([[1] * NUM_DIMS]),  # Only positive encoders
    )


    def sigma_target_func(t, improv_rate):
        if t < INIT_TIME:
            return 0.5 * np.ones(NUM_DIMS)

        if state["mode"] == "exploit":
            return 0.0005 * np.ones(NUM_DIMS)

        # OXYGENATE: Larger, longer-lasting boost
        time_since_switch = t - state["t_mode_switch"]
        boost_magnitude = 0.5  # 50% boost (was 20%)
        boost_duration = 2.0  # 2s decay (was 0.5s)
        boost = boost_magnitude * np.exp(-time_since_switch / boost_duration)

        # Floor sigma to prevent total collapse
        base_sigma = np.clip(state["sigma"], 0.05, SIG_MAX)  # min 5% spread

        return (base_sigma + boost) * np.ones(NUM_DIMS)

    sigma_target_node = nengo.Node(
        label="Sigma Target",
        output=sigma_target_func,
        size_in=1,
        size_out=NUM_DIMS,
    )


    def mode_probe_func(t):
        return 1.0 if state["mode"] == "exploit" else 0.0


    mode_probe_node = nengo.Node(
        label="Mode Probe",
        output=mode_probe_func,
    )


    nengo.Connection(sigma_target_node, motor_sigma, synapse=TAU_BLEND)
    nengo.Connection(motor_sigma, mult_spread_noise.A, synapse=None)

    # nengo.Connection(neuromodulator_ens, noise_gate, synapse=None)
    nengo.Connection(features_input_node, mode_gate_node, synapse=None)
    nengo.Connection(mode_gate_node, noise_gate, synapse=None)

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
    nengo.Connection(selector_node[-1], features_input_node,
                     synapse=TAU_IMPROV_SLOW, transform=+K_IMPROV)
    nengo.Connection(selector_node[-1], features_input_node,
                     synapse=TAU_IMPROV_FAST, transform=-K_IMPROV)

    # Keep stagnation/other feature on channel 1
    nengo.Connection(features_input_node, neuromodulator_ens, synapse=None)

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

    mode_val = nengo.Probe(mode_probe_node, synapse=0.0)


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

# [centre plot]
# plt.plot(sim.trange(), f_centre - problem.optimum.y, "g", marker=".", markersize=1, alpha=0.25)
# f_centre_smooth = np.convolve(f_centre, np.ones(100)/100, mode='same')
# plt.plot(sim.trange(), f_centre_smooth - problem.optimum.y, "g--", label="Centre error")

# add second axis for centre and spread
ax1 = plt.gca()
plt.xlim(0 - INIT_TIME, SIMULATION_TIME)
plt.xlabel("Time, s")
plt.ylabel("Objective value error")
plt.yscale("log")
plt.legend(loc="lower left")

# Highlight exploitation phases
mode_signal = sim.data[mode_val][:,0]
exploit_mask = mode_signal > 0.5
for i in range(len(exploit_mask) - 1):
    if exploit_mask[i]:
        plt.axvspan(sim.trange()[i], sim.trange()[i+1],
                    color='yellow', alpha=0.05, zorder=-1)

ax2 = ax1.twinx()
ax2.plot(sim.trange(), mode_signal, "y:", linewidth=2.0, alpha=0.3, label="Exploit mode")

# ax2.plot(sim.trange(), np.average(sim.data[centre_val], axis=1), "g--", label="Centre")
# ax2.plot(sim.trange(), np.exp(np.average(sim.data[spread_val], axis=1)), "m--", label="Spread")

NUM_FEATURES = 1
colors_1 = plt.cm.rainbow(np.linspace(0, 1, NUM_FEATURES))
# for i in range(NUM_FEATURES):
improv_rate = sim.data[features_val][:,0]
# succes_rate = sim.data[features_val][:,1]
ax2.plot(sim.trange(), improv_rate, color="magenta", marker=".", markersize=1, linestyle="dashed", alpha=0.5, label="Feature"), #f"Dim {i+1}")
# ax2.plot(sim.trange(), np.convolve(improv_rate, np.ones(100)/100, mode='same'), linestyle="dashed", color=colors_1[i], label=f"Dim {i+1}")
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
for i in range(NUM_FEATURES):
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
