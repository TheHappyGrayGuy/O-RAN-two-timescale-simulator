"""KPIs and constraints of the ch4 model for one decision (alpha, kappa, omega, psi).

Equation numbers refer to chapter4.tex labels. Unstable M/M/1 queues yield
delay = inf and a violated stability constraint instead of a negative delay.
"""
from dataclasses import dataclass

import numpy as np

from .channel import prb_rate, sinr
from .config import C_LIGHT, CU, DU

TOL = 1e-9


@dataclass
class Decision:
    alpha: np.ndarray  # (B,U)   UE-RU association alpha_{u_s}^b
    kappa: np.ndarray  # (B,U,M) PRB allocation kappa_{u_s}^{b,m}
    omega: np.ndarray  # (B,U,M) power [W] omega_{u_s}^{b,m}
    psi: np.ndarray    # (2,B,N) psi_b^{l,n}, l in {CU, DU}

    @classmethod
    def from_compact(cls, assoc, prb_owner, prb_power, cu_node, du_node, B, N):
        """assoc (U,) RU per user; prb_owner (B,M) user per PRB or -1;
        prb_power (B,M) W; cu_node, du_node (B,) O-Cloud node per RU."""
        assoc = np.asarray(assoc)
        U, M = len(assoc), prb_owner.shape[1]
        alpha = np.zeros((B, U))
        alpha[assoc, np.arange(U)] = 1
        kappa = np.zeros((B, U, M))
        b_idx, m_idx = np.nonzero(prb_owner >= 0)
        kappa[b_idx, prb_owner[b_idx, m_idx], m_idx] = 1
        omega = kappa * prb_power[:, None, :]
        psi = np.zeros((2, B, N))
        psi[CU, np.arange(B), cu_node] = 1
        psi[DU, np.arange(B), du_node] = 1
        return cls(alpha, kappa, omega, psi)

    @property
    def assoc(self):
        return self.alpha.argmax(axis=0)

    @property
    def cu_node(self):
        return self.psi[CU].argmax(axis=1)

    @property
    def du_node(self):
        return self.psi[DU].argmax(axis=1)


def _mm1(service, arrival):
    """M/M/1 sojourn time 1/(mu - lambda); inf if unstable, 0 if no traffic."""
    service, arrival = np.broadcast_arrays(np.asarray(service, float), np.asarray(arrival, float))
    out = np.full(service.shape, np.inf)
    idle = arrival <= 0
    stable = ~idle & (service > arrival)
    out[idle] = 0.0
    out[stable] = 1.0 / (service[stable] - arrival[stable])
    return out


def erlang_c(c, a):
    """Probability of waiting in M/M/c with c servers and offered load a = lambda/mu (< c)."""
    terms = np.cumprod(np.r_[1.0, a / np.arange(1, c + 1)])  # a^k / k!, k = 0..c
    top = terms[c] * c / (c - a)
    return top / (terms[:c].sum() + top)


def _node_queue_delay(topo, psi, c_tilde, lam_b, qbar_b, core_hz):
    """Per-function sojourn time when every node n is an M/M/c queue (c = n_cores).

    A packet of RU b at layer l needs S = c~^l qbar_b / core_hz seconds of one core.
    All packets of the functions hosted on n share the queue, so placement and load matter.
    Returns d_proc_l (2,B) = W_q(node) + S (RU-mean packet), and W_q per node.
    Per-user sojourn uses the user's own slice packet size (eq.20 uses q_s per service).
    """
    S = c_tilde[:, None] * qbar_b[None, :] / core_hz            # (2,B) service time
    lam_n = np.einsum("lbn,b->n", psi, lam_b)                   # packets/s entering node n
    work_n = np.einsum("lbn,b,lb->n", psi, lam_b, S)            # offered load in cores
    wait = np.zeros(topo.N)
    for n in range(topo.N):
        c, a = topo.n_cores[n], work_n[n]
        if lam_n[n] <= 0:
            continue
        if a >= c:
            wait[n] = np.inf
            continue
        es = a / lam_n[n]                                       # mean service time at n
        wait[n] = erlang_c(c, a) * es / (c - a)
    node_of = psi.argmax(axis=2)                                # (2,B)
    d = np.where(lam_b[None, :] > 0, wait[node_of] + S, 0.0)
    return d, wait


def jain(x):
    x = np.asarray(x, float)
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if (x ** 2).sum() > 0 else 0.0


def evaluate(topo, users, G, H, dec: Decision):
    cfg = topo.cfg
    B, N = topo.B, topo.N
    U = users.U
    lam, q = users.lam, users.q_bits
    a, k, w, psi = dec.alpha, dec.kappa, dec.omega, dec.psi
    assoc, cu, du = dec.assoc, dec.cu_node, dec.du_node
    r = {}

    # ---- rates, eq.(4)-(5)
    rate_bum = prb_rate(sinr(G, H, w * k, cfg), cfg) * k
    rate_bu = rate_bum.sum(axis=2)                     # R_{b,u_s}, eq.4
    rate_u = (a * rate_bu).sum(axis=0)                 # R_{u_s},   eq.5
    r.update(rate_bum=rate_bum, rate_bu=rate_bu, rate_u=rate_u)

    # ---- per-RU traffic
    lam_b = a @ lam                                    # sum_u alpha lambda  [pkt/s]
    load_b = a @ (lam * q)                             # sum_u alpha lambda q [bit/s]
    with np.errstate(invalid="ignore", divide="ignore"):
        qbar_b = np.where(lam_b > 0, load_b / lam_b, 0.0)  # mean packet size at RU b
    ru_rate = (a * rate_bu).sum(axis=1)                # sum_u alpha R_{b,u}
    r.update(lam_b=lam_b, load_b=load_b, qbar_b=qbar_b, ru_rate=ru_rate)

    # ---- power, eq.(a5)
    ru_power = (a[:, :, None] * k * w).sum(axis=(1, 2))
    r["ru_power"] = ru_power

    # ---- CPU demand, eq.(7); node usage, eq.(8)
    c_tilde = np.asarray(cfg.cpu_per_bps)              # (2,) CU, DU
    c_demand = c_tilde[:, None] * load_b[None, :]      # c_b^l (2,B)
    node_cpu = np.einsum("lbn,lb->n", psi, c_demand)   # c_n(t)
    r.update(cpu_demand=c_demand, node_cpu=node_cpu, cpu_util=node_cpu / topo.cpu_capacity)
    # ---- fronthaul/midhaul demand, eq.(9); link loads, eq.(10)-(11)
    w_mid = cfg.w_tilde_mid * ru_rate
    w_front = cfg.w_tilde_front * ru_rate
    split = cu != du
    mid_paths = topo.p_mid[cu, du]                     # (B,E), zero rows when co-located
    front_paths = topo.p_front[np.arange(B), du]       # (B,E)
    link_mid = (split * w_mid) @ mid_paths
    link_front = w_front @ front_paths
    link_load = link_mid + link_front
    r.update(w_mid=w_mid, w_front=w_front, link_load=link_load,
             link_util=link_load / topo.link_capacity)

    # ---- processing delay, eq.(20)-(21)
    if cfg.processing_model == "node_queue":
        d_proc_l, node_wait = _node_queue_delay(topo, psi, c_tilde, lam_b, qbar_b, cfg.core_hz)
        node_of = psi.argmax(axis=2)                                  # (2,B)
        d_proc_u = (node_wait[node_of[:, assoc]] + c_tilde[:, None] * q[None, :] / cfg.core_hz).sum(axis=0)
        r["node_wait"] = node_wait
    else:
        if cfg.processing_model == "thesis":
            c_alloc = c_demand
        elif cfg.processing_model == "proportional":
            c_alloc = c_demand * np.einsum("lbn,n->lb", psi, topo.cpu_capacity / np.maximum(node_cpu, TOL))
        else:
            raise ValueError(cfg.processing_model)
        with np.errstate(invalid="ignore", divide="ignore"):
            upsilon = np.where(qbar_b > 0, c_alloc * cfg.gamma_bits_per_cycle / qbar_b, 0.0)  # (2,B)
        d_proc_l = _mm1(upsilon, lam_b[None, :])
        d_proc_u = d_proc_l.sum(axis=0)[assoc]
    d_proc_b = d_proc_l.sum(axis=0)
    r.update(d_proc_l=d_proc_l, d_proc_b=d_proc_b, d_proc_u=d_proc_u)

    # ---- transport delay, eq.(23)-(24) (fronthaul denominator sign corrected: mu - lambda)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_mid = np.where(qbar_b > 0, w_mid / qbar_b, 0.0)
        mu_front = np.where(qbar_b > 0, w_front / qbar_b, 0.0)
    d_tr_mid = np.where(split, _mm1(mu_mid, lam_b), 0.0)
    d_tr_front = _mm1(mu_front, lam_b)
    d_prop_b = topo.prop_mid[cu, du] + topo.prop_front[np.arange(B), du]
    d_b = d_prop_b + d_proc_b + d_tr_mid + d_tr_front  # eq.13 (RU-mean, for reporting)
    r.update(d_tr_mid=d_tr_mid, d_tr_front=d_tr_front, d_prop_b=d_prop_b, d_b=d_b)

    # ---- radio delay, eq.(14), (26)
    dist = np.linalg.norm(topo.ru_pos[assoc] - users.pos, axis=1)
    d_prop_u = dist / C_LIGHT
    d_tr_u = _mm1(rate_u / q, lam)
    d_u = d_prop_u + d_tr_u
    d_e2e = (d_prop_b + d_tr_mid + d_tr_front)[assoc] + d_proc_u + d_u   # eq.12-13 per user
    r.update(d_prop_u=d_prop_u, d_tr_u=d_tr_u, d_u=d_u, d_e2e=d_e2e)

    # ---- per-slice delay, eq.(29); cost, eq.(30); objective
    S = len(cfg.slices)
    d_slice = np.array([d_e2e[users.slice_idx == s].mean() if np.any(users.slice_idx == s) else 0.0
                        for s in range(S)])
    phi_node = topo.cost_factor * node_cpu / topo.cpu_capacity * cfg.cost_unit_v  # phi_{c,n}
    phi_bar = np.einsum("lbn,n->", psi, phi_node) / B
    delta = np.array([s.delta for s in cfg.slices])
    if cfg.objective_norm == "raw":
        objective = float(np.sum(delta * d_slice * cfg.delay_unit_in_objective + (1 - delta) * phi_bar))
    elif cfg.objective_norm == "qos":
        # both terms dimensionless in [0, ~1]: delay relative to the slice budget, cost relative to
        # the worst case (CU and DU of every RU on a fully loaded edge node)
        d_norm = np.array([(d_e2e / users.d_qos)[users.slice_idx == s].mean()
                           if np.any(users.slice_idx == s) else 0.0 for s in range(S)])
        phi_ref = 2 * topo.cost_factor.max() * cfg.cost_unit_v
        objective = float(np.sum(delta * d_norm + (1 - delta) * phi_bar / phi_ref))
    else:
        raise ValueError(cfg.objective_norm)
    r.update(d_slice=d_slice, phi_node=phi_node, phi_bar=float(phi_bar), objective=objective)

    # ---- constraints
    qos_ok = d_e2e < users.d_qos
    cons = {
        "eq1_single_association": np.allclose(a.sum(axis=0), 1),
        "eq2_prb_needs_association": bool(np.all(k <= a[:, :, None] + TOL)),
        "eq3_prb_exclusive": bool(np.all((a[:, :, None] * k).sum(axis=1) <= 1 + TOL)),
        "eqa5_power": bool(np.all(ru_power <= cfg.p_max_w * (1 + 1e-9))),
        "eq6_single_placement": np.allclose(psi.sum(axis=2), 1),
        "eq8_cpu_capacity": bool(np.all(node_cpu <= topo.cpu_capacity * (1 + 1e-9))),
        "eq10_11_link_capacity": bool(np.all(link_load <= topo.link_capacity * (1 + 1e-9))),
        "eq15_qos": bool(np.all(qos_ok)),
        "eq22_processing_stable": bool(np.all(np.isfinite(d_proc_l))),
        "eq25_transport_stable": bool(np.all(np.isfinite(d_tr_front) & np.isfinite(d_tr_mid))),
        "eq27_radio_stable": bool(np.all(np.isfinite(d_tr_u))),
        "fronthaul_latency": cfg.fronthaul_budget_s is None or bool(np.all(
            topo.prop_front[np.arange(B), du] + d_tr_front <= cfg.fronthaul_budget_s)),
        "sched_slot": bool(np.all(~split | (topo.prop_mid[cu, du] + d_proc_l[CU] + d_tr_mid <= cfg.slot_s))),
    }
    r["constraints"] = cons
    r["feasible"] = all(cons.values())

    # ---- summary KPIs
    satisfied = qos_ok & np.isfinite(d_e2e)
    r.update(
        qos_ok=qos_ok,
        satisfaction_ratio=float(satisfied.mean()),
        satisfaction_per_slice=np.array([satisfied[users.slice_idx == s].mean()
                                         if np.any(users.slice_idx == s) else np.nan for s in range(S)]),
        throughput_bps=float(rate_u.sum()),
        spectral_eff=float(rate_u.sum() / (B * cfg.bandwidth_hz)),
        jain_rate_over_demand=jain(rate_u / users.offered_bps),
        edge_share=float(psi[:, :, topo.edge_nodes].sum() / (2 * B)),
    )
    return r
