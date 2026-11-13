"""Simple, deterministic decisions used to exercise the environment and as baselines.

Placement scenarios mirror ch6: dynamic (learned later), all-edge, CU-regional/DU-edge.
"""
import numpy as np

from .kpis import Decision


def associate_max_gain(G):
    """Each UE joins the RU with the strongest large-scale gain (eq.1 holds by construction)."""
    return G.argmax(axis=0)


def allocate_prbs_round_robin(assoc, B, M):
    """PRBs of each RU dealt round-robin to its users (eq.2, eq.3 hold by construction)."""
    owner = -np.ones((B, M), dtype=int)
    for b in range(B):
        mine = np.flatnonzero(assoc == b)
        if len(mine):
            owner[b] = mine[np.arange(M) % len(mine)]
    return owner


def allocate_prbs_demand(assoc, offered_bps, B, M):
    """PRBs split in proportion to offered load (largest-remainder), at least one each if possible."""
    owner = -np.ones((B, M), dtype=int)
    for b in range(B):
        mine = np.flatnonzero(assoc == b)
        if not len(mine):
            continue
        share = offered_bps[mine] / offered_bps[mine].sum() * M
        n = np.floor(share).astype(int)
        if M >= len(mine):
            n = np.maximum(n, 1)
        while n.sum() > M:
            n[np.argmax(n)] -= 1
        rem = share - np.floor(share)
        for i in np.argsort(-rem)[: M - n.sum()]:
            n[i] += 1
        owner[b, : n.sum()] = np.repeat(mine, n)
    return owner


def equal_power(prb_owner, p_max_w):
    """P_max split equally over the PRBs actually used by each RU."""
    used = prb_owner >= 0
    per = np.where(used.sum(axis=1, keepdims=True) > 0, p_max_w / np.maximum(used.sum(axis=1, keepdims=True), 1), 0)
    return used * per


def nearest_node(topo, nodes):
    """For each RU, the node in `nodes` with the shortest fronthaul route."""
    nodes = np.asarray(nodes)
    return nodes[topo.prop_front[:, nodes].argmin(axis=1)]


def placement(topo, scenario, rng=None):
    edge, reg = topo.edge_nodes, topo.regional_nodes
    if scenario == "all_edge":
        du = nearest_node(topo, edge)
        return du.copy(), du
    if scenario == "cu_regional_du_edge":
        du = nearest_node(topo, edge)
        cu = reg[topo.prop_mid[du][:, reg].argmin(axis=1)]
        return cu, du
    if scenario == "all_regional":
        n = nearest_node(topo, reg)
        return n.copy(), n
    if scenario == "random":
        return rng.integers(topo.N, size=topo.B), rng.integers(topo.N, size=topo.B)
    raise ValueError(f"unknown placement scenario {scenario!r}")


def reference_decision(net, scenario="all_edge", prb="round_robin", rng=None):
    topo, users, cfg = net.topo, net.users, net.cfg
    assoc = associate_max_gain(net.G)
    if prb == "round_robin":
        owner = allocate_prbs_round_robin(assoc, topo.B, cfg.n_prb)
    else:
        owner = allocate_prbs_demand(assoc, users.offered_bps, topo.B, cfg.n_prb)
    power = equal_power(owner, cfg.p_max_w)
    cu, du = placement(topo, scenario, rng)
    return Decision.from_compact(assoc, owner, power, cu, du, topo.B, topo.N)


def penalised_objective(net, dec, penalty=1e3):
    """Objective plus a fixed penalty per violated constraint (inf objective -> 1e6)."""
    r = net.evaluate(dec)
    obj = r["objective"] if np.isfinite(r["objective"]) else 1e6
    return obj + penalty * sum(not v for v in r["constraints"].values())


def optimize_placement(net, dec):
    """Coordinate descent over (CU node, DU node) per RU, other variables fixed.

    Returns the improved Decision and its penalised objective.
    """
    N, B = net.topo.N, net.topo.B
    cu, du = dec.cu_node.copy(), dec.du_node.copy()

    def build(cu, du):
        psi = np.zeros((2, B, N))
        psi[0, np.arange(B), cu] = 1
        psi[1, np.arange(B), du] = 1
        return Decision(dec.alpha, dec.kappa, dec.omega, psi)

    best = penalised_objective(net, dec)
    improved = True
    while improved:
        improved = False
        for b in range(B):
            for c in range(N):
                for d in range(N):
                    cu2, du2 = cu.copy(), du.copy()
                    cu2[b], du2[b] = c, d
                    s = penalised_objective(net, build(cu2, du2))
                    if s < best - 1e-12:
                        best, cu, du, improved = s, cu2, du2, True
    return build(cu, du), best
