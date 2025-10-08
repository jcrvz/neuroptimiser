import matplotlib.pyplot as plt
import numpy as np

import nengo

class LTIProcess(nengo.Process):
    def __init__(self, A, B, **kwargs):
        A, B = np.asarray(A), np.asarray(B)

        # check that the matrix shapes are compatible
        assert A.ndim == 2 and A.shape[0] == A.shape[1]
        assert B.ndim == 2 and B.shape[0] == A.shape[0]

        # store the matrices for `make_step`
        self.A = A
        self.B = B

        # pass the default sizes to the Process constructor
        super().__init__(
            default_size_in=B.shape[1], default_size_out=A.shape[0], **kwargs
        )

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        return {"state": np.zeros(self.A.shape[0])}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (self.B.shape[1],)
        assert shape_out == (self.A.shape[0],)
        A, B = self.A, self.B
        s = state["state"]

        def step(t, x):
            s[:] += dt * (np.dot(A, s) + np.dot(B, x))
            return s

        return step


# demonstrate the LTIProcess in action
A = [[-0.1, -1], [1, -0.1]]
B = [[10], [-10]]

with nengo.Network() as model:
    u = nengo.Node(lambda t: 1 if t < 0.1 else 0)
    # we don't need to specify size_in and size_out!
    a = nengo.Node(LTIProcess(A, B))
    nengo.Connection(u, a)
    a_p = nengo.Probe(a)

with nengo.Simulator(model) as sim:
    sim.run(20.0)

plt.figure()
plt.plot(sim.trange(), sim.data[a_p])
plt.xlabel("time [s]")
plt.show()

plt.figure()
plt.plot(sim.data[a_p][:, 0], sim.data[a_p][:, 1])
plt.xlabel("state 0")
plt.ylabel("state 1")
plt.axis("equal")
plt.show()