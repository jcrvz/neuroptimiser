import numpy as np
import nengo


class LinearCore(nengo.Network):
    """A linear dynamical core: dx/dt = A x + B u, y = x (identity)."""

    def __init__(self, dim, A=None, B=None, **net_kwargs):
        super().__init__(**net_kwargs)
        self.dim = dim
        with self:
            # ensemble representing x
            x = nengo.Ensemble(n_neurons=dim * 50, dimensions=dim)

            # recurrent connection implements A
            if A is not None:
                # use LinearFilter for A
                # Note: synapse = LinearFilter(s) implements dynamics; but here we approximate simple first-order
                nengo.Connection(
                    x, x, transform=A, synapse=0.1  # 0.1 s time constant placeholder
                )

            # input node for u (heuristic proposals)
            u_node = nengo.Node(size_in=dim)
            nengo.Connection(u_node, x, transform=B if B is not None else np.eye(dim), synapse=0.1)

            # output node (just reads x)
            x_out = nengo.Node(size_in=dim)
            nengo.Connection(x, x_out, synapse=None)

        # expose ports
        self.x = x
        self.u = u_node
        self.x_out = x_out


def random_heuristic(dim, scale=0.1):
    """Generate a random Gaussian perturbation for proposal u."""
    return np.random.randn(dim) * scale


def sphere_objective(x):
    """Simple sphere objective; lower is better."""
    return np.sum(x ** 2)


def run_neuroptimiser(
        dim=5,
        A=None,
        B=None,
        n_steps=100,
        dt=0.001,
        proposal_scale=0.1,
        select_every=0.1,
):
    # default A, B = zero (pure integrator with proposals)
    if A is None:
        A = np.zeros((dim, dim))
    if B is None:
        B = np.eye(dim)

    # create network
    net = nengo.Network()
    with net:
        core = LinearCore(dim, A=A, B=B)
        # probe x
        p_x = nengo.Probe(core.x_out)

    sim = nengo.Simulator(net, dt=dt)

    # initial best
    best_x = np.random.randn(dim)
    best_f = sphere_objective(best_x)

    # Write initial best into the network by driving the u input
    # (We trick it: u = best_x - current_x, assuming zero initial state.)
    # Actually, simpler: let simulation warm up, then override via connection hack.

    # run simulation in chunks
    steps_per_select = int(select_every / dt)
    for i in range(n_steps):
        # generate proposal(s)
        u_prop = random_heuristic(dim, scale=proposal_scale)
        # run sim for that many steps while feeding that proposal
        # but here Nengo sim doesn’t support changing Node input mid-run,
        # so we must restart or use a Node with callback.

        # Simpler for prototype: manually simulate step by step
        sim.run_steps(steps_per_select, progress_bar=False)

        # read current x
        x_data = sim.data[p_x][-1]
        f_val = sphere_objective(x_data)
        if f_val < best_f:
            best_f = f_val
            best_x = x_data.copy()
            print(f"Step {i}: new best f = {best_f:.5f}")

            # To reset network state toward best_x, we could reinitialize sim state
            sim.reset(seed=0)
            # or add a corrective u to drive toward best_x (not implemented here)

    # return trajectory
    x_traj = sim.data[p_x]
    return x_traj, best_x, best_f


if __name__ == "__main__":
    traj, x_best, f_best = run_neuroptimiser(dim=2, n_steps=100)
    print("Best x:", np.mean(x_best))
    print("Best f:", f_best)