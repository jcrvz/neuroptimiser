#%%
import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.utils.ensemble import sorted_neurons
from nengo.utils.matplotlib import rasterplot

# Transformations
from neuroptimiser.utils import tro2s, trs2o

from ioh import get_problem

#%%

PROBLEM_ID      = 15  # Problem ID from the IOH framework
PROBLEM_INS     = 2  # Problem instance
NUM_DIMENSIONS  = 5  # Number of dimensions for the problem

problem         = get_problem(fid=PROBLEM_ID, instance=PROBLEM_INS, dimension=NUM_DIMENSIONS)
problem.reset()
print(problem)

X_LOWER_BOUND   = problem.bounds.lb
X_UPPER_BOUND   = problem.bounds.ub
X_INITIAL_GUESS = np.random.uniform(X_LOWER_BOUND, X_UPPER_BOUND, NUM_DIMENSIONS)

#%%

# Create the Nengo model
model = nengo.Network(label="nNeurOpti V1", seed=69)
with model:
    # INITIALISATION
    # --------------------------------------------------------------
    v0_state    = tro2s(X_INITIAL_GUESS, X_LOWER_BOUND, X_UPPER_BOUND)
    # Input nodes for the bounds and initial guess
    # input_x_l    = nengo.Node(label="x_l", output=X_LOWER_BOUND)
    # input_x_u    = nengo.Node(label="x_u", output=X_UPPER_BOUND)
    # input_x_0    = nengo.Node(label="x_0", output=X_INITIAL_GUESS)

    # Evaluate the objective function
    def f_obj(t, state_vector):
        """Evaluate the objective function at the given state vector."""
        v       = np.clip(state_vector, -1.0, 1.0)  # Ensure within bounds, consider toroidal space later
        x_vals  = trs2o(v, X_LOWER_BOUND, X_UPPER_BOUND)
        # Implement here the transformation for integer variables if needed
        fv      = problem(x_vals)
        return np.concatenate((v, [fv]))

    # LOCAL STATE
    state = {
        "best_v": v0_state.copy(),
        "best_f": f_obj(None, v0_state)[-1],
    }

    def init_pulse(t):
        if t < 0.01:
            return v0_state.copy()
        return np.zeros(NUM_DIMENSIONS)

    init_node = nengo.Node(output=init_pulse, size_out=NUM_DIMENSIONS)

    # State ensemble
    neurons_per_dim     = 64
    ea             = nengo.networks.EnsembleArray(
        label           = "ens_array",
        n_neurons       = neurons_per_dim,
        n_ensembles     = NUM_DIMENSIONS,
        ens_dimensions  = 1,
        radius          = 1.0,
        # intercepts  = nengo.dists.Uniform(-0.9, 0.9),
        # max_rates   = nengo.dists.Uniform(80,220),
        # encoders    = nengo.dists.UniformHypersphere(surface=True),
        neuron_type     = nengo.LIF(
            # initial_state   = {
            #     "voltage": nengo.dists.Uniform(-1.0, 1.0),
            #     "refractory_time": np.zeros(n_neurons)
            # },
            # min_voltage     = -1.0,
        ),
        # noise=nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 0.5), seed=42),
    )

    nengo.Connection(init_node, ea.input, synapse=None)

    # Recurrent connection to hold the state
    eta = 0.6 # "Learning rate"
    def hs(t, x):
        if state["best_v"] is None:
            return np.zeros_like(x)
        # best_v = state["best_v"]
        return x + eta * (state["best_v"] - x)
        # return x + eta * (best_v - x) + np.random.normal(0, 0.1, size=x.shape)  # Add some noise for exploration

    tau     = 0.1
    # nengo.Connection(
    #     label       = "recurrent",
    #     pre         = motor_ens.output,
    #     post        = motor_ens.input,
    #     synapse     = tau,
    #     function    = hs,
    #     transform   = np.eye(NUM_DIMENSIONS),
    # )
    hs_node = nengo.Node(
        label       = "hs",
        size_in     = NUM_DIMENSIONS,
        size_out    = NUM_DIMENSIONS,
        output      = hs
    )

    # Route state through hs_node, then feed back into the EA input
    nengo.Connection(ea.output, hs_node, synapse=0)
    nengo.Connection(hs_node, ea.input, synapse=tau, transform=np.eye(NUM_DIMENSIONS))

    # Encoding function for the initial guess in state space within [-1, 1]
    # input_v0    = nengo.Node(label="v0_state", output=v0_state.copy())

    # nengo.Connection(input_v0, ens_lif, synapse=None)

    # Connect to the objective function node
    obj_node    = nengo.Node(
        label="f_obj",
        output=f_obj,
        size_in=NUM_DIMENSIONS,
        size_out=NUM_DIMENSIONS+1
    )
    nengo.Connection(
        pre     = ea.output,
        post    = obj_node,
        synapse = 0,
    )

    # ====
    def local_selector(t, x):
        v       = x[:NUM_DIMENSIONS]
        fv      = float(x[NUM_DIMENSIONS])
        if state["best_f"] is None or (fv < state["best_f"] and t >= 0.01):  # Avoid initial transient
            state["best_f"] = fv
            state["best_v"] = v.copy()

        if state["best_v"] is None:
            return np.concatenate((v, [fv]))
        else:
            return np.concatenate((state["best_v"], [state["best_f"]]))

    selector_node   = nengo.Node(
        label="selector",
        output=local_selector,
        size_in=NUM_DIMENSIONS + 1,
        size_out=NUM_DIMENSIONS + 1,
    )
    nengo.Connection(obj_node, selector_node, synapse=0)
    # ====

    # Controller towards the best-so-far
    # eta = 0.2 # Learning rate or movement rate towards the best
    # def controller(t, x):
    #     best_v  = x[:NUM_DIMENSIONS]
    #     curr_v  = x[NUM_DIMENSIONS:]
    #     return eta * (best_v - curr_v)

    # control_node = nengo.Node(
    #     label="controller",
    #     output=controller,
    #     size_in=2 * NUM_DIMENSIONS,
    #     size_out=NUM_DIMENSIONS,
    # )

    # nengo.Connection(selector_node[:NUM_DIMENSIONS], control_node[:NUM_DIMENSIONS], synapse=0)
    # nengo.Connection(ens_lif, control_node[NUM_DIMENSIONS:], synapse=0)      # Current state vector

    # nengo.Connection(control_node, ens_lif, synapse=0.01)  # Apply control to the ensemble

    fbest_only = nengo.Node(
        size_in=NUM_DIMENSIONS+1,
        size_out=1,
        output=lambda t, x: x[-1]
    )
    nengo.Connection(selector_node, fbest_only, synapse=0)

    # input_probe = nengo.Probe(v_0, synapse=0.01)
    ens_lif_val = nengo.Probe(ea.output, synapse=0.01)  # 10ms filter
    obj_val     = nengo.Probe(obj_node[-1], synapse=0)
    fbest_val   = nengo.Probe(fbest_only, synapse=0.01)
    ea_spk      = [nengo.Probe(ens.neurons, synapse=0.01) for ens in ea.ensembles]

#%%
simulation_time = 20.0  # seconds

# Create our simulator
with nengo.Simulator(model) as sim:
    # Run it for 1 second
    sim.run(simulation_time)
#%%
plt.figure()
# plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value")
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")
plt.xlim(0, simulation_time)
plt.yscale("log")
plt.legend()
plt.title(f"Obj. func. err. vs time | fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()
#%%
# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[ens_lif_val], label=[f"x{i}" for i in range(NUM_DIMENSIONS)])
# plt.plot(sim.trange(), sim.data[input_probe], "r", label="Input")
plt.xlim(0, simulation_time)
plt.legend()
plt.show()
#%%
plt.figure()
plt.plot(sim.trange(), sim.data[obj_val] - problem.optimum.y, label="Objective value error")
plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far error")
# plt.hlines(problem.optimum.y, 0, simulation_time, colors="r", linestyles="dashed", label="Optimal value")
# plt.plot(sim.trange(), sim.data[fbest_val] - problem.optimum.y, label="Best-so-far diff") #"Best-so-far value")
plt.xlim(0, simulation_time)
plt.yscale("log")
plt.legend()
plt.title(f"Obj. func. err. vs time | fbest: {sim.data[fbest_val][-1][0]:.2f} :: fopt: {problem.optimum.y}")
plt.show()
#%%
# Plot the spiking output of the ensemble
plt.figure()
spikes_cat = np.hstack([sim.data[p] for p in ea_spk])
rasterplot(sim.trange(), spikes_cat) #, use_eventplot=True)
plt.xlim(0, simulation_time)
plt.show()

#%%
# # For interest's sake, you can also sort by encoder
# indices = sorted_neurons(ens_lif, sim, iterations=250)
# plt.figure()
# rasterplot(sim.trange(), sim.data[ens_lif_spk][:, indices])
# plt.xlim(0, simulation_time)
# plt.show()
#%%
