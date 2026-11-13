"""Step 2: user / RU spatial distributions, slices, traffic and mobility."""
import numpy as np
import pytest
from scipy import stats

from oran_sim import Config
from oran_sim.users import assign_slices, build_users, drop_positions


def test_uniform_drop_is_uniform(cfg, rng):
    p = drop_positions(cfg, rng, n=5000)
    assert np.all((p >= 0) & (p <= cfg.L_max))
    for k in range(2):  # Kolmogorov-Smirnov against U(0, L)
        assert stats.kstest(p[:, k], "uniform", args=(0, cfg.L_max)).pvalue > 0.01


def test_clustered_drop_is_inside_and_concentrated(cfg, rng):
    c = cfg.with_(ue_distribution="clustered")
    p = drop_positions(c, rng, n=3000)
    assert np.all((p >= 0) & (p <= c.L_max))
    hist_c = np.histogram2d(p[:, 0], p[:, 1], bins=10, range=[[0, c.L_max]] * 2)[0]
    hist_u = np.histogram2d(*drop_positions(cfg, rng, n=3000).T, bins=10, range=[[0, c.L_max]] * 2)[0]
    assert hist_c.var() > 5 * hist_u.var()


@pytest.mark.parametrize("U", [5, 10, 15, 20])
def test_balanced_slices_cover_all(cfg, rng, U):
    s = assign_slices(cfg.with_(n_ue=U), rng)
    counts = np.bincount(s, minlength=3)
    assert counts.sum() == U                       # sum_s U_s = U
    assert counts.max() - counts.min() <= 1


def test_random_slices_follow_probs(cfg, rng):
    s = assign_slices(cfg.with_(slice_assignment="random", slice_probs=(0.6, 0.3, 0.1)), rng, n=20000)
    np.testing.assert_allclose(np.bincount(s) / len(s), [0.6, 0.3, 0.1], atol=0.015)


def test_slice_traffic_profiles(cfg, rng):
    u = build_users(cfg, rng)
    for s, prof in enumerate(cfg.slices):
        idx = u.users_of_slice(s)
        assert np.all(u.lam[idx] == prof.arrival_rate)
        assert np.all(u.d_qos[idx] == prof.d_qos_s)
        assert np.all(u.q_bits[idx] == prof.packet_size * 8)   # default unit: byte
    assert np.all(u.offered_bps == u.lam * u.q_bits)


def test_packet_unit_switch(cfg):
    assert cfg.with_(packet_size_unit="kbit").packet_bits_per_slice == (1.5e6, 2e5, 2e4)
    assert cfg.packet_bits_per_slice == (12000, 1600, 160)


def test_poisson_arrivals_mean_and_dispersion(cfg, rng):
    u = build_users(cfg.with_(n_ue=3), rng)       # one user per slice
    counts = np.array([u.poisson_arrivals(0.1, rng) for _ in range(20000)])
    np.testing.assert_allclose(counts.mean(0), u.lam * 0.1, rtol=0.03)
    np.testing.assert_allclose(counts.var(0), u.lam * 0.1, rtol=0.06)  # Poisson: var = mean


def test_mobility_bounded_and_speed(cfg, rng):
    u = build_users(cfg.with_(n_ue=30), rng)
    start = u.pos.copy()
    for _ in range(2000):
        prev = u.pos.copy()
        u.move(1.0, rng)
        assert np.all((u.pos >= 0) & (u.pos <= cfg.L_max))
        assert np.all(np.linalg.norm(u.pos - prev, axis=1) <= u.speed + 1e-9)
    still = u.speed == 0                           # mMTC devices are static
    np.testing.assert_array_equal(u.pos[still], start[still])
    assert np.all(np.linalg.norm(u.pos[~still] - start[~still], axis=1) > 0)


def test_static_mobility(cfg, rng):
    u = build_users(cfg.with_(mobility="static"), rng)
    p = u.pos.copy()
    u.move(10.0, rng)
    np.testing.assert_array_equal(u.pos, p)
