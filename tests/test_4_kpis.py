"""Step 4: KPIs and constraints of ch4 (rates, CPU, transport, delays, cost, objective)."""
import numpy as np
import pytest

from oran_sim import CU, DU, Config, Decision, Network, reference_decision
from oran_sim.config import C_LIGHT
from oran_sim.kpis import _mm1, erlang_c, jain
from oran_sim.reference_policies import allocate_prbs_demand, allocate_prbs_round_robin, equal_power


# ---------------------------------------------------------------- helpers
THESIS_LITERAL = dict(processing_model="thesis", objective_norm="raw", fronthaul_budget_s=None)


def tiny_net(**kw):
    """1 RU, 1 edge + 1 regional node, 1 eMBB user, 2 PRBs, no fading."""
    base = dict(n_ru=1, n_edge=1, n_regional=1, n_ue=1, n_prb=2, fading="none", fronthaul_budget_s=None)
    base.update(kw)
    return Network(Config(**base))


def tiny_decision(net, cu=0, du=0, power=None):
    M = net.cfg.n_prb
    power = np.full((1, M), net.cfg.p_max_w / M) if power is None else power
    return Decision.from_compact([0], np.zeros((1, M), int), power, [cu], [du], 1, net.topo.N)


# ---------------------------------------------------------------- unit pieces
def test_mm1():
    np.testing.assert_allclose(_mm1([10, 5, 5, 0], [4, 5, 6, 0]), [1 / 6, np.inf, np.inf, 0])


def test_erlang_c():
    assert erlang_c(1, 0.5) == pytest.approx(0.5)           # M/M/1: P(wait) = rho
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)         # textbook value
    assert erlang_c(10, 1e-6) < 1e-12


def test_jain():
    assert jain([1, 1, 1]) == pytest.approx(1)
    assert jain([1, 0, 0]) == pytest.approx(1 / 3)


def test_decision_compact_roundtrip(net):
    d = reference_decision(net, "cu_regional_du_edge")
    assert d.alpha.shape == (4, net.users.U) and d.kappa.shape == (4, net.users.U, net.cfg.n_prb)
    np.testing.assert_array_equal(d.assoc, net.G.argmax(axis=0))
    assert np.all(net.topo.node_is_edge[d.du_node]) and np.all(~net.topo.node_is_edge[d.cu_node])


# ---------------------------------------------------------------- hand calculation
@pytest.mark.parametrize("model", ["thesis", "node_queue"])
@pytest.mark.parametrize("cu,du", [(0, 0), (1, 0), (1, 1)])
def test_single_user_hand_calc(cu, du, model):
    net = tiny_net(processing_model=model)
    cfg, t, u = net.cfg, net.topo, net.users
    r = net.evaluate(tiny_decision(net, cu, du))

    lam, q = u.lam[0], u.q_bits[0]
    snr = net.G[0, 0] * (cfg.p_max_w / 2) / cfg.noise_per_prb_w
    R = 2 * cfg.prb_bandwidth_hz * np.log2(1 + snr)                  # eq.4/5 over 2 PRBs
    assert r["rate_u"][0] == pytest.approx(R)

    c_cu, c_du = (np.array(cfg.cpu_per_bps) * lam * q)               # eq.7
    if model == "thesis":                                            # eq.20-21 as written
        proc = 1 / (c_cu * cfg.gamma_bits_per_cycle / q - lam) + 1 / (c_du * cfg.gamma_bits_per_cycle / q - lam)
    else:                                                            # one packet stream per node
        S = np.array(cfg.cpu_per_bps) * q / cfg.core_hz              # service time CU, DU
        nodes = [cu, du]

        def sojourn(l):
            n = nodes[l]
            hosted = [m for m in range(2) if nodes[m] == n]
            a = lam * S[hosted].sum()                                # offered load in cores
            c = t.n_cores[n]
            es = S[hosted].mean()
            return erlang_c(c, a) * es / (c - a) + S[l]
        proc = sojourn(0) + sojourn(1)
    front = 1 / (cfg.w_tilde_front * R / q - lam)
    mid = 0.0 if cu == du else 1 / (cfg.w_tilde_mid * R / q - lam)
    prop_fiber = t.prop_mid[cu, du] + t.prop_front[0, du]
    radio = np.linalg.norm(t.ru_pos[0] - u.pos[0]) / C_LIGHT + 1 / (R / q - lam)   # eq.14, 26
    e2e = prop_fiber + proc + front + mid + radio                    # eq.12-13

    assert r["d_proc_b"][0] == pytest.approx(proc)
    assert r["d_tr_front"][0] == pytest.approx(front)
    assert r["d_tr_mid"][0] == pytest.approx(mid)
    assert r["d_e2e"][0] == pytest.approx(e2e)
    assert r["d_slice"][0] == pytest.approx(e2e)                     # eq.29, one eMBB user

    used = np.zeros(2)
    used[cu] += c_cu
    used[du] += c_du
    np.testing.assert_allclose(r["node_cpu"], used)                  # eq.8 LHS
    phi = t.cost_factor * used / t.cpu_capacity * cfg.cost_unit_v
    assert r["phi_bar"] == pytest.approx(phi[cu] + phi[du])          # eq.30 with B = 1
    # link load: fronthaul w_front on route RU->DU node, midhaul w_mid on CU->DU route
    expect_link = cfg.w_tilde_front * R * t.p_front[0, du] + (cu != du) * cfg.w_tilde_mid * R * t.p_mid[cu, du]
    np.testing.assert_allclose(r["link_load"], expect_link)


def test_objective_formula_raw(cfg):
    net = Network(cfg.with_(objective_norm="raw"))
    r = net.evaluate(reference_decision(net))
    delta = np.array([s.delta for s in net.cfg.slices])
    expect = np.sum(delta * r["d_slice"] * 1e3 + (1 - delta) * r["phi_bar"])
    assert r["objective"] == pytest.approx(expect)


def test_objective_formula_qos(net):
    r = net.evaluate(reference_decision(net))
    u = net.users
    delta = np.array([s.delta for s in net.cfg.slices])
    d_norm = np.array([(r["d_e2e"] / u.d_qos)[u.slice_idx == s].mean() for s in range(3)])
    expect = np.sum(delta * d_norm + (1 - delta) * r["phi_bar"] / (2 * 10 * net.cfg.cost_unit_v))
    assert r["objective"] == pytest.approx(expect)


def test_cost_edge_vs_regional_ratio(net):
    """Same load: phi^e/phi^r = (cost 10 / cost 1) * (C_r / C_e = 10) = 100."""
    e = net.evaluate(reference_decision(net, "all_edge"))
    g = net.evaluate(reference_decision(net, "all_regional"))
    assert e["phi_bar"] / g["phi_bar"] == pytest.approx(100, rel=1e-9)


def test_colocated_has_no_midhaul(net):
    r = net.evaluate(reference_decision(net, "all_edge"))
    assert np.all(r["d_tr_mid"] == 0)
    assert np.all(r["link_load"][net.topo.link_tier == "midhaul"] == 0)
    r2 = net.evaluate(reference_decision(net, "cu_regional_du_edge"))
    active = r2["lam_b"] > 0
    assert np.all(r2["d_tr_mid"][active] > 0)
    assert np.all(r2["link_load"][net.topo.link_tier == "midhaul"].sum() > 0)


def test_rate_aggregation(net):
    r = net.evaluate(reference_decision(net))
    np.testing.assert_allclose(r["rate_bu"], r["rate_bum"].sum(axis=2))
    np.testing.assert_allclose(r["rate_u"], r["rate_bu"][net.G.argmax(0), np.arange(net.users.U)])
    assert r["throughput_bps"] == pytest.approx(r["rate_u"].sum())


def test_proportional_cpu_is_faster_than_demand(cfg):
    a = Network(cfg.with_(processing_model="thesis"))
    b = Network(cfg.with_(processing_model="proportional"))
    ra, rb = a.evaluate(reference_decision(a)), b.evaluate(reference_decision(b))
    active = ra["lam_b"] > 0
    assert np.all(rb["d_proc_b"][active] < ra["d_proc_b"][active])


# ---------------------------------------------------------------- constraint detection
def _violated(net, dec):
    return {k for k, v in net.evaluate(dec)["constraints"].items() if not v}


def test_reference_decision_is_feasible(net):
    assert _violated(net, reference_decision(net, "all_edge")) == set()


def test_eq1_double_association(net):
    d = reference_decision(net)
    d.alpha[:, 0] = 1
    assert "eq1_single_association" in _violated(net, d)


def test_eq2_prb_without_association(net):
    d = reference_decision(net)
    b = (d.assoc[0] + 1) % 4
    d.kappa[b, 0, 0] = 1
    assert "eq2_prb_needs_association" in _violated(net, d)


def test_eq3_prb_shared(net):
    d = reference_decision(net)
    b = d.assoc[0]
    others = np.flatnonzero(d.assoc == b)
    if len(others) < 2:
        pytest.skip("RU has a single user")
    d.kappa[b, others, 0] = 1
    assert "eq3_prb_exclusive" in _violated(net, d)


def test_eqa5_power_exceeded(net):
    d = reference_decision(net)
    d.omega *= 1.5
    assert "eqa5_power" in _violated(net, d)


def test_eq6_double_placement(net):
    d = reference_decision(net)
    d.psi[DU, 0, :] = 1
    assert "eq6_single_placement" in _violated(net, d)


def test_eq8_cpu_overload(cfg):
    net = Network(cfg.with_(cpu_edge=1e6))
    assert "eq8_cpu_capacity" in _violated(net, reference_decision(net, "all_edge"))
    assert "eq8_cpu_capacity" not in _violated(net, reference_decision(net, "all_regional"))


def test_eq10_11_link_overload(cfg):
    net = Network(cfg.with_(fronthaul_link_bps=1e6))
    assert "eq10_11_link_capacity" in _violated(net, reference_decision(net))


def test_eq15_qos(cfg):
    from oran_sim.config import DEFAULT_SLICES
    tight = tuple(s.__class__(**{**s.__dict__, "d_qos_s": 1e-6}) for s in DEFAULT_SLICES)
    net = Network(cfg.with_(slices=tight))
    assert "eq15_qos" in _violated(net, reference_decision(net))


def test_eq22_processing_unstable(cfg):
    net = Network(cfg.with_(processing_model="thesis", gamma_bits_per_cycle=1e-3))  # c~*gamma < 1 -> mu < lambda
    v = _violated(net, reference_decision(net))
    assert "eq22_processing_stable" in v and "eq15_qos" in v


def test_node_queue_overload_is_unstable(cfg):
    """With the M/M/c node model, eq.22 (stability) coincides with eq.8 (CPU capacity)."""
    net = Network(cfg.with_(cpu_edge=1e6, core_hz=1e6))
    v = _violated(net, reference_decision(net, "all_edge"))
    assert {"eq8_cpu_capacity", "eq22_processing_stable"} <= v


def test_fronthaul_budget(cfg):
    net = Network(cfg)
    assert "fronthaul_latency" not in _violated(net, reference_decision(net, "all_edge"))
    tight = Network(cfg.with_(fronthaul_budget_s=1e-6))
    assert "fronthaul_latency" in _violated(tight, reference_decision(tight, "all_edge"))
    off = Network(cfg.with_(fronthaul_budget_s=None))
    assert "fronthaul_latency" not in _violated(off, reference_decision(off, "all_regional"))


def test_node_queue_delay_depends_on_placement_and_load(cfg):
    """Decision D2: the processing delay must react to where functions run and how loaded the node is."""
    net = Network(cfg)
    edge = net.evaluate(reference_decision(net, "all_edge"))
    reg = net.evaluate(reference_decision(net, "all_regional"))
    active = edge["lam_b"] > 0
    assert np.all(edge["d_proc_b"][active] > reg["d_proc_b"][active])   # 1 shared core vs 10 cores
    waits = []
    for U in [5, 10, 20, 40]:
        n = Network(cfg.with_(n_ue=U))
        waits.append(n.evaluate(reference_decision(n, "all_edge"))["node_wait"].max())
    assert np.all(np.diff(waits) > 0)                                   # congestion grows with load


def test_node_queue_per_slice_service_time(cfg):
    """Users on the same RU and node differ only by their own packet service time c~ q_s / f."""
    net = Network(cfg.with_(n_ue=12))
    r = net.evaluate(reference_decision(net, "all_edge"))
    u, c = net.users, np.array(cfg.cpu_per_bps)
    b = net.G.argmax(0)
    for i in range(u.U):
        for j in range(u.U):
            if b[i] == b[j]:
                expect = c.sum() * (u.q_bits[i] - u.q_bits[j]) / cfg.core_hz
                assert r["d_proc_u"][i] - r["d_proc_u"][j] == pytest.approx(expect, abs=1e-15)


def test_thesis_model_delay_ignores_placement(cfg):
    """Documents why D2 was needed: eq.20 as written gives the same processing delay on every node."""
    net = Network(cfg.with_(processing_model="thesis"))
    e = net.evaluate(reference_decision(net, "all_edge"))["d_proc_b"]
    g = net.evaluate(reference_decision(net, "all_regional"))["d_proc_b"]
    np.testing.assert_allclose(e, g)


def test_eq27_radio_unstable():
    net = tiny_net(p_max_dbm=-60)
    r = net.evaluate(tiny_decision(net))
    assert not r["constraints"]["eq27_radio_stable"] and np.isinf(r["d_e2e"][0])


def test_literal_kbit_reading_is_infeasible(cfg):
    """Documents open issue #1: ch6 packet sizes read as kilobits overload the O-Cloud."""
    net = Network(cfg.with_(packet_size_unit="kbit"))
    v = _violated(net, reference_decision(net, "all_regional"))
    assert "eq8_cpu_capacity" in v


# ---------------------------------------------------------------- reference policies
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("U", [5, 10, 15, 20])
def test_reference_policies_respect_structure(cfg, seed, U):
    net = Network(cfg.with_(seed=seed, n_ue=U))
    for prb in ["round_robin", "demand"]:
        for sc in ["all_edge", "cu_regional_du_edge", "all_regional"]:
            v = _violated(net, reference_decision(net, sc, prb))
            assert not v & {"eq1_single_association", "eq2_prb_needs_association", "eq3_prb_exclusive",
                            "eqa5_power", "eq6_single_placement"}


def test_demand_prb_split(cfg):
    assoc = np.array([0, 0, 0, 1])
    owner = allocate_prbs_demand(assoc, np.array([10.0, 1.0, 1.0, 5.0]), 2, 10)
    counts = np.bincount(owner[owner >= 0], minlength=4)
    assert counts.sum() == 20 and counts[0] > counts[1] and counts[1] >= 1 and counts[3] == 10
    rr = allocate_prbs_round_robin(assoc, 2, 10)
    assert np.bincount(rr[0], minlength=4)[:3].tolist() == [4, 3, 3]
    p = equal_power(rr, 10.0)
    np.testing.assert_allclose(p.sum(axis=1), 10.0)
