"""Phase 5 baselines, from naive to optimised, all returning a Decision for the current network state."""
import numpy as np

from .kpis import Decision
from .reference_policies import optimize_placement, reference_decision

STATIC_SCENARIOS = ("all_edge", "cu_regional_du_edge", "all_regional")


def random_decision(net, rng):
    """Uniformly random association, PRB owners among each RU's users, power split and placement."""
    t, U, M = net.topo, net.users.U, net.cfg.n_prb
    assoc = rng.integers(t.B, size=U)
    owner = -np.ones((t.B, M), int)
    power = np.zeros((t.B, M))
    for b in range(t.B):
        mine = np.flatnonzero(assoc == b)
        if len(mine):
            owner[b] = rng.choice(mine, size=M)
            power[b] = rng.dirichlet(np.ones(M)) * net.cfg.p_max_w
    return Decision.from_compact(assoc, owner, power, rng.integers(t.N, size=t.B),
                                 rng.integers(t.N, size=t.B), t.B, t.N)


def static(net, scenario):
    """ch6 fixed placement scenarios with max-gain association, round-robin PRBs, equal power."""
    return reference_decision(net, scenario)


def static_best(net):
    """Best of the three ch6 static scenarios (an oracle over fixed placements)."""
    return min((reference_decision(net, s) for s in STATIC_SCENARIOS), key=lambda d: _penalised(net, d))


def placement_search(net):
    """Static best, then per-RU coordinate descent on (CU node, DU node)."""
    return optimize_placement(net, static_best(net))[0]


def joint_optimum(net, restarts=4):
    """Full solver (association search, exact power, PRB local search, exact placement)."""
    from .optimal import Solver
    return Solver(net).solve(restarts=restarts).decision


def _penalised(net, d):
    from .reference_policies import penalised_objective
    return penalised_objective(net, d)


BASELINES = {
    "random": lambda net, rng: random_decision(net, rng),
    "all_edge": lambda net, rng: static(net, "all_edge"),
    "cu_regional_du_edge": lambda net, rng: static(net, "cu_regional_du_edge"),
    "all_regional": lambda net, rng: static(net, "all_regional"),
    "static_best": lambda net, rng: static_best(net),
    "placement_search": lambda net, rng: placement_search(net),
    "joint_optimum": lambda net, rng: joint_optimum(net),
}
