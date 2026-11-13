"""Integration: two-timescale network dynamics + KPIs over time for the ch6 scenarios."""
import numpy as np
import pytest

from oran_sim import Config, Network, reference_decision


def test_small_step_redraws_fading_only(net):
    G, H = net.G.copy(), net.H.copy()
    net.small_step()
    np.testing.assert_array_equal(net.G, G)
    assert not np.allclose(net.H, H)


def test_large_step_every_k_slots(net):
    pos, G = net.users.pos.copy(), net.G.copy()
    for _ in range(net.cfg.slots_per_large_step - 1):
        net.small_step()
    np.testing.assert_array_equal(net.G, G)
    net.small_step()
    assert not np.array_equal(net.users.pos, pos)
    assert not np.allclose(net.G, G, atol=0)


def test_same_seed_same_trajectory(cfg):
    a, b = Network(cfg), Network(cfg)
    for _ in range(25):
        a.small_step()
        b.small_step()
    np.testing.assert_array_equal(a.users.pos, b.users.pos)
    np.testing.assert_array_equal(a.H, b.H)


@pytest.mark.parametrize("U", [5, 10, 15, 20])
@pytest.mark.parametrize("scenario", ["all_edge", "cu_regional_du_edge", "all_regional"])
def test_kpis_over_episode(cfg, U, scenario):
    net = Network(cfg.with_(n_ue=U))
    for _ in range(30):
        r = net.evaluate(reference_decision(net, scenario))
        assert np.all(r["rate_u"] >= 0)
        assert np.all(r["d_e2e"] > 0)
        assert 0 <= r["satisfaction_ratio"] <= 1
        assert r["phi_bar"] >= 0
        net.small_step()


def test_more_power_more_rate(cfg):
    lo, hi = Network(cfg), Network(cfg.with_(p_max_dbm=60))     # ch6 sweep 40 vs 60 dBm
    rl, rh = lo.evaluate(reference_decision(lo)), hi.evaluate(reference_decision(hi))
    assert rh["throughput_bps"] > rl["throughput_bps"]
    assert np.all(rh["d_tr_u"] <= rl["d_tr_u"])


def test_more_users_more_cost(cfg):
    costs = []
    for U in [5, 10, 15, 20]:
        n = Network(cfg.with_(n_ue=U))
        costs.append(n.evaluate(reference_decision(n))["phi_bar"])
    assert np.all(np.diff(costs) > 0)


SCENARIOS = ["all_edge", "cu_regional_du_edge", "all_regional"]


def test_tradeoff_exists_under_default_model(cfg):
    """Guards decision D2: optimised placement must beat every fixed ch6 scenario by a clear margin,
    and URLLC users must be served from the edge more often than other slices."""
    from oran_sim.reference_policies import optimize_placement, penalised_objective
    gains, urllc, other = [], [], []
    for seed in range(4):
        net = Network(cfg.with_(seed=seed, n_ue=5))
        fixed = min(penalised_objective(net, reference_decision(net, s)) for s in SCENARIOS)
        start = min(SCENARIOS, key=lambda s: penalised_objective(net, reference_decision(net, s)))
        dec, best = optimize_placement(net, reference_decision(net, start))
        gains.append((fixed - best) / fixed)
        on_edge = net.topo.node_is_edge[dec.cu_node][dec.assoc]
        urllc.append(on_edge[net.users.slice_idx == 1].mean())
        other.append(on_edge[net.users.slice_idx != 1].mean())
    assert np.mean(gains) > 0.10
    assert np.mean(urllc) > np.mean(other)


def test_optimizer_never_worse_than_start(cfg):
    from oran_sim.reference_policies import optimize_placement, penalised_objective
    net = Network(cfg.with_(n_ue=10))
    for s in SCENARIOS:
        start = reference_decision(net, s)
        _, best = optimize_placement(net, start)
        assert best <= penalised_objective(net, start) + 1e-12
