# neuropt_nengo_runonce.py
# Multi-agent, run-once Neuroptimiser (pure Nengo)
# - Continuous hd: dx = -K(x-g) + u + C*n
# - On-chip epoch clock, proposal sample/hold, selector, gated memories
# - One sim.run(T_total). No outer loop.

import numpy as np
import nengo

# ---------- small blocks ----------

def ring_knn(N, k=2):
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            W[i, (i+j) % N] = 1.0
            W[i, (i-j) % N] = 1.0
    return (W > 0).astype(float)

def piecewise_const(seed, D, scale):
    rng = np.random.RandomState(seed)
    last = rng.randn(D) * scale
    def fn(t):
        # returns last; external gate will resample at pulses
        return last
    # a setter we’ll call from the resample node (kept in closure via list)
    return fn, rng

# Evaluates f(x) inside graph (Sphere here; swap with your function)
def f_eval(x):
    return float(np.dot(x, x))

# ---------- GatedMemory: y <- (1-gate)*y + gate*inp ----------
def gated_memory(net, D, neurons=80, tau=0.1, label="mem"):
    with net:
        y = nengo.networks.EnsembleArray(neurons, D, neuron_type=nengo.LIF(), label=label)
        inp = nengo.Node(size_in=D, label=f"{label}_in")
        gate = nengo.Node(size_in=1, label=f"{label}_gate")

        # y' = y + gate*(inp - y) -> continuous approx of discrete latch
        # Implement: to y.input: (I - tau*1)*y + tau*1*(y + gate*(inp - y))
        # Simpler: use two paths with Elementwise product:
        diff = nengo.Node(size_in=2*D, label=f"{label}_diff",
                          output=lambda t, x: x[:D] - x[D:2*D])
        nengo.Connection(inp, diff[:D], synapse=None)
        nengo.Connection(y.output, diff[D:2*D], synapse=None)

        # Multiply gate * (inp - y) using a Product network
        prod = nengo.networks.Product(n_neurons=neurons, dimensions=D)
        nengo.Connection(gate, prod.input_a, transform=np.ones((D, 1)), synapse=None)            # scalar gate broadcast
        nengo.Connection(diff, prod.input_b, synapse=None)

        # Leaky integrator: y' = y + k*(gate*(inp - y))
        k = 1.0
        nengo.Connection(y.output, y.input, transform=np.eye(D), synapse=tau)
        nengo.Connection(prod.output, y.input, transform=k*np.eye(D), synapse=tau)

    return y, inp, gate

# ---------- Model ----------
def build_runonce(N=6, D=3, T_total=6.0, T_epoch=0.1,
                  neurons_per_dim=64, K=2.0, C_ngb=0.7,
                  u_scale=0.12, tau=0.1, seed=42):

    rng = np.random.RandomState(seed)
    W = ring_knn(N, k=2)

    with nengo.Network(seed=seed, label="Neuropt-RunOnce") as model:
        # Clock: square pulses every T_epoch
        pulse = nengo.Node(lambda t: 1.0 if (t % T_epoch) < 0.01 else 0.0, label="pulse")  # 10ms pulse

        # Smooth gate for better numerics
        gate = nengo.Node(size_in=1, output=lambda t, x: 1.0 if x[0] > 0.5 else 0.0, label="gate")
        nengo.Connection(pulse, gate, synapse=0.005)

        # Global best memory g \in R^D (gated)
        g_net = nengo.Network(label="g_mem")
        with g_net:
            g_mem, g_in, g_gate = gated_memory(g_net, D, neurons=neurons_per_dim, tau=tau, label="gmem")
        nengo.Connection(gate, g_gate, synapse=None)
        model.g = g_mem

        # Per-agent state, proposal S/H, selector (all in-graph)
        agents = []
        P_bus = []   # stacked personal bests
        FP_bus = []  # stacked best fitness
        Xouts = []
        Nouts = []

        # Precompute neighbour mapping (best-neighbour delta) as a Node
        # We’ll compute this each epoch from P_bus/FP_bus via a combine node
        def pfp_bus_out():
            P = np.zeros(N*D); FP = np.ones(N)*np.inf
            return P, FP

        # Mutable buffers exposed through closures
        P_flat = np.zeros(N*D); FP = np.ones(N)*np.inf

        PFP_bus = nengo.Node(lambda t: np.concatenate([P_flat, FP]), size_out=N*D + N, label="PFP_bus")

        # Neighbour summary delta n_i = p_j* - p_i
        def neigh_sum(t, vec):
            P = vec[:N*D].reshape(N, D)
            F = vec[N*D:]
            out = np.zeros((N, D))
            for i in range(N):
                neigh = np.where(W[i] > 0)[0]
                if len(neigh) == 0:
                    continue
                j = neigh[np.argmin(F[neigh])]
                out[i] = P[j] - P[i]
            return out.ravel()

        NBR = nengo.Node(neigh_sum, size_in=N*D + N, size_out=N*D, label="neigh_delta")
        nengo.Connection(PFP_bus, NBR, synapse=None)

        # Global selector helper: scan mini array, choose best index
        best_idx = nengo.Node(size_in=N, size_out=1,
                              output=lambda t, F: float(np.argmin(F)), label="argmin_idx")

        # Projections to update g at pulses: pick row k from P and send to g_in
        def pick_row(i):
            return nengo.Node(size_in=N*D, size_out=D,
                              output=lambda t, flat: flat.reshape(N, D)[i], label=f"pickP_{i}")

        picks = [pick_row(i) for i in range(N)]
        for i in range(N):
            nengo.Connection(PFP_bus[:N*D], picks[i], synapse=None)

        # Selector gate to g: compare FP each epoch and gate g update
        # We’ll just always gate g on pulses with the current global best; g_mem keeps the value.
        nengo.Connection(gate, g_gate, synapse=None)

        # Build per agent
        for i in range(N):
            # x_i ensemble (state)
            x = nengo.networks.EnsembleArray(neurons_per_dim, D, neuron_type=nengo.LIF(), label=f"x_{i}")

            # decoded output of x
            x_out = nengo.Node(size_in=D, label=f"x_out_{i}")
            nengo.Connection(x.output, x_out, synapse=None)
            Xouts.append(x_out)

            # u_i: proposal sample/hold. We resample only at pulses by gating a memory.
            u_mem_net = nengo.Network(label=f"u_mem_{i}")
            with u_mem_net:
                u_mem, u_in, u_gate = gated_memory(u_mem_net, D, neurons=neurons_per_dim, tau=tau, label=f"umem_{i}")
            nengo.Connection(gate, u_gate, synapse=None)   # resample on pulse

            # proposal generator Node (Gaussian) — its output is ALWAYS noise, but only latched on pulse
            _rng = np.random.RandomState(seed + 100 + i)
            prop = nengo.Node(lambda t: u_scale * _rng.randn(D), size_out=D, label=f"prop_{i}")
            nengo.Connection(prop, u_in, synapse=None)

            # neighbour slice for agent i
            n_i = nengo.Node(size_in=N*D, size_out=D,
                             output=lambda t, flat, i=i: flat.reshape(N, D)[i], label=f"n_{i}")
            nengo.Connection(NBR, n_i, synapse=None)

            # hd dynamics: dx = -K(x-g) + u + C*n
            # NEF transforms with lowpass tau:
            # recurrent: I - tau*K
            # inputs: tau*K*g, tau*u, tau*C*n
            nengo.Connection(x.output, x.input, transform=(np.eye(D) - tau*K), synapse=tau)
            nengo.Connection(g_mem.output, x.input, transform=(tau*K)*np.eye(D), synapse=None)
            nengo.Connection(u_mem.output, x.input, transform=(tau*np.eye(D)), synapse=None)
            nengo.Connection(n_i, x.input, transform=(tau*C_ngb)*np.eye(D), synapse=None)

            # ---- per-agent selector (on-chip)
            # Evaluate f(x) via Node. (Replace with your f.)
            f_node = nengo.Node(size_in=D, size_out=1,
                                output=lambda t, xv: f_eval(xv), label=f"f_{i}")
            nengo.Connection(x_out, f_node, synapse=0.05)

            # Personal best memory p_i (gated by accept)
            p_net = nengo.Network(label=f"p_mem_{i}")
            with p_net:
                p_mem, p_in, p_gate = gated_memory(p_net, D, neurons=neurons_per_dim, tau=tau, label=f"pmem_{i}")

            # Fitness memory fp_i (scalar gated memory)
            fp_net = nengo.Network(label=f"fp_mem_{i}")
            with fp_net:
                # 1D “memory”: use an Ensemble with D=1
                fp_ens = nengo.Ensemble(neurons_per_dim, 1, neuron_type=nengo.LIF(), label=f"fp_{i}")
                fp_in = nengo.Node(size_in=1, label=f"fp_in_{i}")
                fp_gate = nengo.Node(size_in=1, label=f"fp_gate_{i}")
                # Leaky integrator with gated increment
                diff = nengo.Node(size_in=2, output=lambda t, x: x[0]-x[1], label=f"fp_diff_{i}")
                # diff input: new_f - old_f
                nengo.Connection(f_node, diff[0], synapse=None)
                nengo.Connection(fp_ens,  diff[1], synapse=None)
                prod = nengo.networks.Product(n_neurons=neurons_per_dim, dimensions=1)
                nengo.Connection(fp_gate, prod.input_a, synapse=None)
                nengo.Connection(diff,    prod.input_b, synapse=None)
                nengo.Connection(fp_ens,  fp_ens, transform=1.0, synapse=tau)
                nengo.Connection(prod.output, fp_ens, transform=1.0, synapse=tau)

            # Accept gate: 1 if new f < fp (or if fp is +inf initially) — computed at pulse
            def accept_fn(t, vec):
                f_new, f_old, pulse = vec
                if pulse < 0.5:
                    return 0.0
                if not np.isfinite(f_old) or f_new < f_old:
                    return 1.0
                return 0.0

            accept = nengo.Node(size_in=3, size_out=1, output=accept_fn, label=f"accept_{i}")
            # Wire inputs: [f_new, f_old, pulse]
            nengo.Connection(f_node, accept[0], synapse=None)
            nengo.Connection(fp_ens, accept[1], synapse=None)
            nengo.Connection(pulse,  accept[2], synapse=None)

            # Gate memories with accept
            nengo.Connection(accept, p_gate, synapse=None)
            nengo.Connection(accept, fp_gate, synapse=None)

            # Provide inputs to memories
            nengo.Connection(x_out, p_in, synapse=None)
            nengo.Connection(f_node, fp_in, synapse=None)  # (not strictly needed; fp_ens reads f_node via diff)

            # Expose for global buses
            agents.append({
                "x_out": x_out,
                "p_mem": p_mem,
                "fp_ens": fp_ens,
                "u_mem": u_mem,
            })

        # ----- Global selector (update g at pulses using argmin over fp_i)
        # Build FP vector and P matrix flows into buses
        def pf_sink(i):
            def sink(t, vec):
                # write p_i, fp_i to global arrays at any time (we sample at pulses logically)
                p = vec[:D]; fpi = vec[-1]
                P_flat[i*D:(i+1)*D] = p
                FP[i] = fpi
            return sink

        # Pack [p_i, fp_i] per agent via nodes
        for i, a in enumerate(agents):
            cat = nengo.Node(size_in=D+1, size_out=D+1,
                             output=lambda t, x: x, label=f"cat_{i}")
            nengo.Connection(a["p_mem"].output, cat[:D], synapse=None)
            nengo.Connection(a["fp_ens"],       cat[D:D+1], synapse=None)
            sink = nengo.Node(pf_sink(i), size_in=D+1, size_out=0, label=f"sink_{i}")
            nengo.Connection(cat, sink, synapse=None)

        # Index of best FP at pulse
        idx_gate = nengo.Node(size_in=1, output=lambda t, x: x[0], label="idx_gate")
        nengo.Connection(pulse, idx_gate, synapse=None)

        # choose best index (we’ll read FP via PFP_bus → best_idx) and route to g_in at pulses
        best = nengo.Node(size_in=1, size_out=D, label="chosenP",
                          output=lambda t, k: np.zeros(D))  # overwritten below via routing nodes

        # Build small router: for each i, on pulse, if i==argmin(FP) gate p_i into g_in
        # Simpler: at pulse, overwrite g_in from pickP[k]; we approximate by always driving g_in with pickP[argmin(F)]
        # We do this by a Node that reads PFP and outputs pick of best row:
        def pick_best(t, vec):
            P = vec[:N*D].reshape(N, D)
            F = vec[N*D:]
            k = int(np.argmin(F))
            return P[k]

        CHOOSE = nengo.Node(pick_best, size_in=N*D + N, size_out=D, label="choose_best")
        nengo.Connection(PFP_bus, CHOOSE, synapse=None)
        # Gate into g_mem: multiply by pulse (on pulse write strongly)
        prod_g = nengo.networks.Product(n_neurons=neurons_per_dim, dimensions=D)
        nengo.Connection(pulse, prod_g.input_a, synapse=None)
        nengo.Connection(CHOOSE, prod_g.input_b, synapse=None)
        nengo.Connection(prod_g.output, g_in, synapse=None)

        # ---------- Probes ----------
        p_g = nengo.Probe(g_mem.output, synapse=0.05)
        p_fg = nengo.Probe(pulse, synapse=None)   # for visualizing pulses
        p_x = [nengo.Probe(a["x_out"], synapse=0.05) for a in agents]

    return model, {"p_g": p_g, "p_x": p_x, "T_total": T_total}

if __name__ == "__main__":
    from ioh import get_problem

    problem_id = 1  # Problem ID from the IOH framework
    problem_ins = 1  # Problem instance
    num_dimensions = 2  # Number of dimensions for the problem

    problem = get_problem(fid=problem_id,
                          instance=problem_ins,
                          dimension=num_dimensions,
                          )
    problem.reset()
    print(problem)

    def f_eval(x):
        val = problem(x)
        return float(val)

    model, H = build_runonce(N=6, D=num_dimensions, T_total=6.0, T_epoch=0.1,
                             neurons_per_dim=64, K=2.0, C_ngb=0.7,
                             u_scale=0.12, tau=0.1, seed=42)
    sim = nengo.Simulator(model, dt=0.001)
    sim.run(H["T_total"])
    g_traj = sim.data[H["p_g"]]
    print("Final best g:", g_traj[-1])

    print("\nf* =", g_traj[-1], " | f_opt =", problem.optimum.y, " | diff =", g_traj[-1] - problem.optimum.y)