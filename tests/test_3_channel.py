"""Step 3: channel models (path loss, shadowing, fading, SINR, PRB rate)."""
import numpy as np
import pytest
from scipy import stats

from oran_sim import Config
from oran_sim.channel import (distances_2d, fspl_db, large_scale_gain, los_probability_umi,
                              pathloss_log_distance_db, pathloss_umi_db, prb_rate, sinr,
                              small_scale_fading)


def test_fspl_known_value():
    # FSPL(1 m, 3.5 GHz) = 20 log10(4 pi f / c) = 43.32 dB
    assert fspl_db(1.0, 3.5e9) == pytest.approx(43.32, abs=0.01)


def test_log_distance_slope_is_10n_per_decade(cfg):
    pl = pathloss_log_distance_db(np.array([10.0, 100.0, 1000.0]), cfg)
    np.testing.assert_allclose(np.diff(pl), 30.0)          # n = 3
    assert pl[0] == pytest.approx(43.32 + 30, abs=0.01)


def test_distance_clipping(cfg):
    d = distances_2d(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [3.0, 4.0], [30.0, 40.0]]), 10.0)
    np.testing.assert_allclose(d, [[10.0, 10.0, 50.0]])


def test_umi_reference_values(cfg):
    # TR 38.901 UMi-SC, fc = 3.5 GHz, hBS = 10 m, hUT = 1.5 m, d2D = 100 m
    d3d = np.hypot(100, 8.5)
    los = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(3.5)
    nlos = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(3.5)
    assert pathloss_umi_db(np.array(100.0), cfg, True) == pytest.approx(los)
    assert pathloss_umi_db(np.array(100.0), cfg, False) == pytest.approx(max(los, nlos))
    assert pathloss_umi_db(np.array(300.0), cfg, False) >= pathloss_umi_db(np.array(300.0), cfg, True)
    assert los_probability_umi(10.0) == 1.0
    assert 0 < los_probability_umi(200.0) < los_probability_umi(50.0) < 1


def test_large_scale_gain_no_shadowing_is_deterministic(cfg, rng):
    ru, ue = np.array([[0.0, 0.0]]), np.array([[100.0, 0.0], [200.0, 0.0]])
    g = large_scale_gain(ru, ue, cfg, rng)
    np.testing.assert_allclose(-10 * np.log10(g), pathloss_log_distance_db(np.array([[100.0, 200.0]]), cfg))


def test_shadowing_statistics(cfg, rng):
    c = cfg.with_(shadowing_std_db=8.0)
    ru, ue = np.zeros((1, 2)), np.tile([[100.0, 0.0]], (20000, 1))
    loss_db = -10 * np.log10(large_scale_gain(ru, ue, c, rng)) - pathloss_log_distance_db(np.array(100.0), c)
    assert abs(loss_db.mean()) < 0.2 and loss_db.std() == pytest.approx(8.0, rel=0.03)


def test_umi_gain_runs(cfg, rng):
    g = large_scale_gain(np.zeros((2, 2)), rng.uniform(0, 1000, (50, 2)), cfg.with_(pathloss_model="3gpp_umi"), rng)
    assert g.shape == (2, 50) and np.all((g > 0) & (g < 1))


@pytest.mark.parametrize("fading", ["rayleigh", "rician", "none"])
def test_fading_unit_mean(cfg, rng, fading):
    h = small_scale_fading((200000,), cfg.with_(fading=fading), rng)
    assert h.mean() == pytest.approx(1.0, rel=0.01)


def test_rayleigh_power_is_exponential(cfg, rng):
    h = small_scale_fading((20000,), cfg, rng)
    assert stats.kstest(h, "expon").pvalue > 0.01


def test_fading_independent_across_prbs(cfg, rng):
    h = small_scale_fading((5000, 2), cfg, rng)
    assert abs(np.corrcoef(h.T)[0, 1]) < 0.05


def test_snr_and_shannon_hand_calc(cfg):
    G = np.array([[1e-10]])
    H = np.ones((1, 1, 1))
    omega = np.full((1, 1, 1), 0.4)
    snr = sinr(G, H, omega, cfg)
    expected = 0.4 * 1e-10 / cfg.noise_per_prb_w
    assert snr[0, 0, 0] == pytest.approx(expected)
    assert prb_rate(snr, cfg)[0, 0, 0] == pytest.approx(1e6 * np.log2(1 + expected))


def test_interference_reduces_sinr(cfg):
    G = np.array([[1e-9, 1e-11], [1e-11, 1e-9]])  # two RUs, two users, cross-gain small
    H = np.ones((2, 2, 1))
    omega = np.zeros((2, 2, 1))
    omega[0, 0, 0] = omega[1, 1, 0] = 1.0          # both RUs reuse PRB 0
    s_off = sinr(G, H, omega, cfg)
    s_on = sinr(G, H, omega, cfg.with_(interference=True))
    expected = 1e-9 / (cfg.noise_per_prb_w + 1e-11)
    assert s_on[0, 0, 0] == pytest.approx(expected)
    assert s_on[0, 0, 0] < s_off[0, 0, 0]
    # orthogonal PRBs -> no interference
    omega2 = np.zeros((2, 2, 2))
    omega2[0, 0, 0] = omega2[1, 1, 1] = 1.0
    np.testing.assert_allclose(sinr(G, np.ones((2, 2, 2)), omega2, cfg.with_(interference=True)),
                               sinr(G, np.ones((2, 2, 2)), omega2, cfg))


def test_link_budget_sane(net):
    """Cell-edge SNR with equal power should be positive (dB) for the default deployment."""
    cfg = net.cfg
    snr_db = 10 * np.log10(net.G.max(axis=0) * cfg.p_max_w / cfg.n_prb / cfg.noise_per_prb_w)
    assert snr_db.min() > 0 and snr_db.max() < 80
