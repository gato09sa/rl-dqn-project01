"""Tabular Q-learning and SARSA.

The agent works on integer states (use envs.discretizer to map a continuous
observation to a state id). Exploration is epsilon-greedy; epsilon is passed in
on every call so the experiment script owns the schedule.

Update rules (Sutton & Barto, 2018, ch. 6):
    Q-learning (off-policy): Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    SARSA      (on-policy) : Q(s,a) += alpha * [r + gamma * Q(s',a')       - Q(s,a)]
where a' in SARSA is the action the agent will actually take next.

Interface shared with DQNAgent:
    act(state, epsilon) -> action      exploration (training)
    greedy(state)       -> action      epsilon = 0 (evaluation)
    update(...)                        one learning step
    save(path) / load(path)

Parallel runs: each agent owns its RNG (np.random.Generator), so several agents
in different processes never share random state.
"""
import numpy as np


class TabularAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, algo="qlearning",
                 q_init=0.0, seed=None):
        if algo not in ("qlearning", "sarsa"):
            raise ValueError("algo must be 'qlearning' or 'sarsa'")
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma, self.algo = alpha, gamma, algo
        self.rng = np.random.default_rng(seed)
        self.Q = np.full((n_states, n_actions), q_init, dtype=np.float64)
        self.visits = np.zeros(n_states, dtype=np.int64)   # how often each state was updated

    def greedy(self, s):
        q = self.Q[s]
        # Random tie-breaking: in an unvisited state all values are equal and a
        # plain argmax would always return action 0, biasing the policy.
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, s, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def update(self, s, a, r, s_next, terminated, a_next=None):
        """One TD update; returns the TD error.

        terminated: True only for a real terminal state (pole fell, car reached
            the goal). A time-limit truncation is NOT terminal, so pass
            terminated=False there and the update still bootstraps from s_next.
        a_next: next action, required for SARSA, ignored by Q-learning.
        """
        if terminated:
            target = r                                        # no future after a terminal state
        elif self.algo == "qlearning":
            target = r + self.gamma * self.Q[s_next].max()    # best next action (off-policy)
        else:
            if a_next is None:
                raise ValueError("SARSA needs a_next")
            target = r + self.gamma * self.Q[s_next, a_next]  # action actually taken (on-policy)
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        self.visits[s] += 1
        return float(td_error)

    def visited_fraction(self):
        """Share of table rows ever updated; shows how much of the table each bin count uses."""
        return float(np.mean(self.visits > 0))

    def save(self, path):
        np.savez_compressed(path, Q=self.Q, visits=self.visits)

    def load(self, path):
        data = np.load(path)
        self.Q, self.visits = data["Q"], data["visits"]
