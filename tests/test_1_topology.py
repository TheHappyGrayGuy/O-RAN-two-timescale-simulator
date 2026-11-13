"""Step 1: telecom environment and topology (RU sites, O-Cloud graph, routes, fiber delays)."""
import networkx as nx
import numpy as np
import pytest

from oran_sim import Config
from oran_sim.topology import build_topology


def test_parameters_from_chapter6(cfg):
    assert cfg.n_ru == 4 and cfg.L_max == 1000
    assert cfg.p_max_dbm == 40 and cfg.bandwidth_hz == 25e6
    assert cfg.pathloss_exp == 3 and cfg.noise_psd_dbm_hz == -174
    assert cfg.cpu_regional == pytest.approx(10 * cfg.cpu_edge)
    assert cfg.cpu_per_bps[1] > cfg.cpu_per_bps[0]          # DU needs more CPU than CU (ch4 sec. 4.2)
    assert cfg.w_tilde_front > cfg.w_tilde_mid               # fronthaul needs more BW than midhaul


def test_derived_units(cfg):
    assert cfg.p_max_w == pytest.approx(10.0)                # 40 dBm
    assert cfg.prb_bandwidth_hz == pytest.approx(1e6)
    assert 10 * np.log10(cfg.noise_per_prb_w) + 30 == pytest.approx(-114.0)


def test_ru_grid_inside_area_and_fixed(cfg):
    t1, t2 = build_topology(cfg), build_topology(cfg.with_(n_ue=20))
    assert t1.ru_pos.shape == (4, 2)
    assert np.all((t1.ru_pos >= 0) & (t1.ru_pos <= cfg.L_max))
    np.testing.assert_allclose(sorted(map(tuple, t1.ru_pos)), [(250, 250), (250, 750), (750, 250), (750, 750)])
    np.testing.assert_array_equal(t1.ru_pos, t2.ru_pos)


@pytest.mark.parametrize("B", [1, 3, 5, 7, 9])
def test_ru_grid_any_count(cfg, B):
    t = build_topology(cfg.with_(n_ru=B))
    assert len(t.ru_pos) == B and len(np.unique(t.ru_pos, axis=0)) == B


def test_ru_random_respects_separation(cfg):
    t = build_topology(cfg.with_(ru_layout="random", n_ru=6))
    d = np.linalg.norm(t.ru_pos[:, None] - t.ru_pos[None], axis=-1) + np.eye(6) * 1e9
    assert d.min() >= cfg.ru_min_separation_m


def test_cloud_node_regions(cfg):
    t = build_topology(cfg)
    edge, reg = t.node_pos[t.node_is_edge], t.node_pos[~t.node_is_edge]
    assert len(edge) == cfg.n_edge and len(reg) == cfg.n_regional
    assert np.all((edge >= cfg.L_max) & (edge <= cfg.L_edge_max))
    assert np.all((reg >= cfg.L_edge_max) & (reg <= cfg.L_regional_max))
    np.testing.assert_array_equal(t.cpu_capacity[t.node_is_edge], cfg.cpu_edge)
    np.testing.assert_array_equal(t.cpu_capacity[~t.node_is_edge], cfg.cpu_regional)
    assert t.cost_factor[t.node_is_edge].min() > t.cost_factor[~t.node_is_edge].max()  # phi^e > phi^r


def test_graph_structure(cfg):
    t = build_topology(cfg)
    B, Ne, Nr = cfg.n_ru, cfg.n_edge, cfg.n_regional
    assert nx.is_connected(t.graph)
    assert t.graph.number_of_nodes() == B + Ne + Nr
    assert len(t.links) == B * Ne + Ne * Nr
    assert np.sum(t.link_tier == "fronthaul") == B * Ne
    np.testing.assert_array_equal(t.link_capacity[t.link_tier == "fronthaul"], cfg.fronthaul_link_bps)
    np.testing.assert_array_equal(t.link_capacity[t.link_tier == "midhaul"], cfg.midhaul_link_bps)


def test_route_indicators_and_delays(cfg):
    t = build_topology(cfg)
    edge, reg = t.edge_nodes, t.regional_nodes
    # RU -> edge: one fronthaul hop; RU -> regional: fronthaul + midhaul hop
    assert np.all(t.p_front[:, edge].sum(axis=2) == 1)
    assert np.all(t.p_front[:, reg].sum(axis=2) == 2)
    # route delay = sum of hop lengths / fiber speed
    for b in range(t.B):
        for n in range(t.N):
            hops = [t.links[e] for e in np.flatnonzero(t.p_front[b, n])]
            length = sum(t.graph.edges[u, v]["length"] for u, v in hops)
            assert t.prop_front[b, n] == pytest.approx(length / cfg.fiber_speed)
    # midhaul: symmetric, zero on diagonal, edge->edge passes through a regional node
    np.testing.assert_allclose(t.prop_mid, t.prop_mid.T)
    assert np.all(np.diag(t.prop_mid) == 0) and np.all(t.p_mid[np.arange(t.N), np.arange(t.N)] == 0)
    assert t.p_mid[edge[0], edge[1]].sum() == 2
    # regional nodes are farther than edge nodes from the RUs
    assert t.prop_front[:, reg].min() > t.prop_front[:, edge].max() * 0.5


def test_topology_reproducible_by_seed(cfg):
    a, b, c = build_topology(cfg), build_topology(cfg), build_topology(cfg.with_(seed=7))
    np.testing.assert_array_equal(a.node_pos, b.node_pos)
    assert not np.allclose(a.node_pos, c.node_pos)
