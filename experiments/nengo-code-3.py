import numpy as np
import nengo


class LinearCore(nengo.Network):
    """Implements dx/dt = A x + B u via a stateful Node.

    The step closure keeps a private x_state array and performs explicit Euler integration every simulator tick dt.

    Exposes:
      - self.input: Node(size_in=dim) for control u(t)
      - self.state: Node(size_in=dim, size_out=dim) emitting x(t)
    """
    def __init__(self, A, B, dt=0.001, x0=None, label="linear_core"):
        super().__init__(label=label)

        # Prepare A, B as float arrays
        A       = np.array(A, dtype=float)
        B       = np.array(B, dtype=float)

        # Read dimensions and, if some, set initial state x0
        dim     = A.shape[0]
        x0      = np.zeros(dim) if x0 is None else np.array(x0, dtype=float).copy()

        with self:
            # Control input u(t)
            self.input = nengo.Node(size_in=dim)

            # Stateful integrator for x(t)
            x_state = x0.copy()

            def step(t, u):
                nonlocal x_state
                dx = A @ x_state + B @ u
                x_state = x_state + dt * dx
                return x_state

            # This Node both *receives* u(t) and *emits* x(t)
            self.state = nengo.Node(step, size_in=dim, size_out=dim)
            nengo.Connection(self.input, self.state, synapse=None)


# ——— Heuristic operators ———

class GaussianProposal(nengo.Process):
    """A process that emits Gaussian proposal deltas u."""

    def __init__(self, dim, scale=0.1, seed=None):
        super().__init__(default_dt=0.001)  # dt is adjustable
        self.dim = dim
        self.scale = scale
        self.rng = np.random.RandomState(seed)

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        # no internal state
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        def step(t):
            # ignore x_in (or optionally allow bias)
            return self.rng.randn(self.dim) * self.scale

        return step


# You can similarly define e.g. coordinate jitter, etc.


# ——— NengoNeuroptimiser class ———

class NengoNeuroptimiser:
    def __init__(
            self,
            dim,
            A=None,
            B=None,
            proposal_processes=None,
            epoch_duration=0.1,  # seconds
            dt=0.001,
            seed=None,
    ):
        self.dim = dim
        self.A = A if A is not None else np.zeros((dim, dim))
        self.B = B if B is not None else np.eye(dim)
        self.epoch_duration = epoch_duration
        self.dt = dt
        self.seed = seed

        # default one Gaussian operator if none provided
        if proposal_processes is None:
            proposal_processes = [GaussianProposal(dim, scale=0.1, seed=seed)]
        self.proposal_processes = proposal_processes

    def _build_model(self, objective_fn, x0):
        m = nengo.Network(seed=self.seed)
        with m:
            # Create the linear core block (pure Nengo)
            sys = (self.A, self.B, np.eye(self.dim), np.zeros((self.dim, self.dim)))  # kept for reference
            lin = LinearCore(self.A, self.B, dt=self.dt, x0=x0, label="linear_core")

            # Control signal u(t) we can change from Python between epochs
            control_vec = np.zeros(self.dim)
            def control_fn(t):
                return control_vec
            control_node = nengo.Node(control_fn, size_out=self.dim)
            nengo.Connection(control_node, lin.input, synapse=None)

            # Proposals: one per operator
            prop_nodes = []
            for i, proc in enumerate(self.proposal_processes):
                prop = nengo.Node(proc, size_in=0, size_out=self.dim)
                prop_nodes.append(prop)

            # Read out x
            x_out = nengo.Node(size_in=self.dim)
            nengo.Connection(lin.state, x_out, synapse=None)

            # Objective evaluator: Node that reads x and outputs scalar
            def obj_fn_wrapper(t, x):
                return objective_fn(x)
            obj_node = nengo.Node(obj_fn_wrapper, size_in=self.dim, size_out=1)
            nengo.Connection(x_out, obj_node, synapse=None)

            # Probes
            p_x = nengo.Probe(x_out)
            p_obj = nengo.Probe(obj_node)
            for i, prop in enumerate(prop_nodes):
                setattr(self, f"p_prop_{i}", nengo.Probe(prop))

        # store references
        self.net = m
        self.lin = lin
        self.prop_nodes = prop_nodes
        self.obj_node = obj_node
        self._control_vec = control_vec  # mutable from Python
        self._control_node = control_node
        self.probes = dict(x=p_x, obj=p_obj)
        for i in range(len(prop_nodes)):
            self.probes[f"prop_{i}"] = getattr(self, f"p_prop_{i}")

    def run(self, objective_fn, x0, n_epochs=50):
        """Run the optimisation loop for n_epochs, each epoch lasting epoch_duration seconds."""
        self._build_model(objective_fn, x0)
        sim = nengo.Simulator(self.net, dt=self.dt)

        # Initialize state by injecting x0: we treat “reset_node” input as B * (x_desired − x_current)
        # (No warmup/reset hack needed; initial state set in LinearCore)

        best_x = x0.copy()
        best_f = objective_fn(best_x)

        n_steps_per_epoch = int(self.epoch_duration / self.dt)
        history = {"x": [], "f": []}

        for ep in range(n_epochs):
            sim.run(self.epoch_duration)

            x_vals = sim.data[self.probes["x"]][-1]
            obj_val = sim.data[self.probes["obj"]][-1]

            # read proposals from each
            prop_vals = [sim.data[self.probes[f"prop_{i}"]][-1] for i in range(len(self.prop_nodes))]

            # choose best among current + proposals:
            # for simplicity compare objective(x_vals + prop)
            cand_x = [x_vals + p for p in prop_vals]
            cand_x.append(x_vals)  # include staying put
            cand_f = [objective_fn(x) for x in cand_x]
            winner_idx = int(np.argmin(cand_f))
            if winner_idx < len(prop_vals):
                chosen = prop_vals[winner_idx]
                # Apply chosen proposal as control input u(t) for the next epoch
                self._control_vec[:] = chosen
            else:
                # No change (stay put)
                self._control_vec[:] = 0.0

            # update best
            if cand_f[winner_idx] < best_f:
                best_f = cand_f[winner_idx]
                best_x = cand_x[winner_idx].copy()

            history["x"].append(best_x.copy())
            history["f"].append(best_f)
            print(f"Epoch {ep} => best f = {best_f:.5f}")

        return history, best_x, best_f


if __name__ == "__main__":
    from ioh import get_problem

    def sphere(x):
        return np.sum(x ** 2)

    problem_id      = 1  # Problem ID from the IOH framework
    problem_ins     = 1  # Problem instance
    num_dimensions  = 2  # Number of dimensions for the problem

    problem = get_problem(fid=problem_id,
                          instance=problem_ins,
                          dimension=num_dimensions,
                          )
    problem.reset()
    print(problem)

    opt = NengoNeuroptimiser(dim=num_dimensions, epoch_duration=0.01, dt=0.01, seed=69)
    history, x_best, f_best = opt.run(problem, x0=np.array([-5.0] * num_dimensions), n_epochs=50)
    print("Best found:", x_best, " f=", f_best)

    import matplotlib.pyplot as plt
    history_x = np.array(history["x"])
    plt.plot(history_x[:, 0], history_x[:, 1], marker='o')
    plt.plot(problem.optimum.x[0], problem.optimum.x[1], 'rx', markersize=10, label='Optimum')
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.title("Trajectory of x during optimisation")
    plt.grid()
    plt.show()

    plt.plot(history["f"])
    plt.hlines(problem.optimum.y, 0, len(history["f"]), colors='r', linestyles='dashed', label='Optimum value')
    plt.xlabel("Epoch")
    plt.ylabel("Best objective value")
    plt.title("Objective value over epochs")
    plt.grid()
    plt.show()
