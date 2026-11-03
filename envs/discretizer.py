"""Uniform state discretization for tabular RL on continuous observations.

Each observation dimension i is cut into bins[i] equal-width intervals inside
[low_i, high_i]. Values outside the range fall into the first or last bin
(np.digitize does this on its own), so the index never goes out of bounds.
The tuple of per-dimension bin indices is flattened into a single integer
state id, so the Q-table is a plain 2-D array Q[state_id, action].

Example (CartPole, 6 bins per dimension):
    disc = UniformDiscretizer(6, CARTPOLE_RANGES)
    disc.n_states          # 6**4 = 1296
    s = disc(obs)          # int in [0, 1296)
"""
import numpy as np

# CartPole-v1 has infinite bounds for the two velocities, so the ranges must be
# chosen by hand. Position and angle use the termination limits (|x| > 2.4,
# |theta| > 12 deg = 0.2095 rad). Velocity ranges are a design choice to justify
# in the report (most visited states fall well inside them).
CARTPOLE_RANGES = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.5, 3.5)]

# MountainCar-v0 has finite bounds: position in [-1.2, 0.6], velocity in [-0.07, 0.07].
MOUNTAINCAR_RANGES = [(-1.2, 0.6), (-0.07, 0.07)]


class UniformDiscretizer:
    def __init__(self, bins, ranges):
        """bins: int (same for every dimension) or list with one int per dimension.
        ranges: list of (low, high) pairs, one per observation dimension."""
        ranges = np.asarray(ranges, dtype=np.float64)
        n_dims = len(ranges)
        if np.isscalar(bins):
            bins = [bins] * n_dims
        self.bins = np.asarray(bins, dtype=np.int64)
        assert len(self.bins) == n_dims, "need one bin count per dimension"
        assert np.all(self.bins >= 1)
        assert np.all(ranges[:, 0] < ranges[:, 1]), "each range needs low < high"
        self.ranges = ranges
        # n bins need only the n-1 interior cut points.
        self.edges = [np.linspace(lo, hi, n + 1)[1:-1] for (lo, hi), n in zip(ranges, self.bins)]
        self.n_states = int(np.prod(self.bins))

    def indices(self, obs):
        """Per-dimension bin index, e.g. (2, 5, 0, 3). Useful for debugging/plots."""
        return tuple(int(np.digitize(x, e)) for x, e in zip(obs, self.edges))

    def __call__(self, obs):
        """Single integer state id in [0, n_states)."""
        return int(np.ravel_multi_index(self.indices(obs), self.bins))

    @classmethod
    def from_space(cls, observation_space, bins):
        """Use the env's own bounds. Only valid when every bound is finite
        (MountainCar yes, CartPole no)."""
        low, high = observation_space.low, observation_space.high
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("observation space has infinite bounds; pass explicit ranges")
        return cls(bins, list(zip(low, high)))
