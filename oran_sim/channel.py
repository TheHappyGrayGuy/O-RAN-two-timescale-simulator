"""Wireless channel: path loss, shadowing, small-scale fading, SINR and PRB rates.

Large-scale gain G[b,u] is redrawn on the large time scale (user positions,
shadowing); small-scale fading H[b,u,m] is redrawn every slot, independently per PRB.
Rate per PRB follows ch6 eq. (shanon): R_{b,u,m} = B_prb log2(1 + SINR).
"""
import numpy as np

from .config import Config, C_LIGHT


def distances_2d(ru_pos, ue_pos, d_min=0.0):
    d = np.linalg.norm(ru_pos[:, None, :] - ue_pos[None, :, :], axis=-1)
    return np.maximum(d, d_min)


def fspl_db(d, fc_hz):
    return 20 * np.log10(4 * np.pi * d * fc_hz / C_LIGHT)


def pathloss_log_distance_db(d, cfg: Config):
    """PL(d) = FSPL(d0) + 10 n log10(d/d0), n = cfg.pathloss_exp (ch6: n = 3)."""
    d0 = cfg.ref_distance_m
    return fspl_db(d0, cfg.carrier_hz) + 10 * cfg.pathloss_exp * np.log10(d / d0)


def los_probability_umi(d2d):
    """3GPP TR 38.901 Table 7.4.2-1, UMi-Street Canyon."""
    d2d = np.asarray(d2d, dtype=float)
    p = 18 / np.maximum(d2d, 1e-9) + np.exp(-d2d / 36) * (1 - 18 / np.maximum(d2d, 1e-9))
    return np.where(d2d <= 18, 1.0, p)


def pathloss_umi_db(d2d, cfg: Config, los):
    """3GPP TR 38.901 Table 7.4.1-1, UMi-Street Canyon (LOS / NLOS)."""
    fc = cfg.carrier_hz / 1e9
    h_bs, h_ut = cfg.ru_height_m, cfg.ue_height_m
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    d_bp = 4 * (h_bs - 1) * (h_ut - 1) * cfg.carrier_hz / C_LIGHT
    pl1 = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc)
    pl2 = 32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2)
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos = np.maximum(pl_los, 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (h_ut - 1.5))
    return np.where(los, pl_los, pl_nlos)


def large_scale_gain(ru_pos, ue_pos, cfg: Config, rng):
    """Linear power gain G[b,u] = 10^(-(PL + shadowing)/10)."""
    d = distances_2d(ru_pos, ue_pos, cfg.min_distance_m)
    if cfg.pathloss_model == "log_distance":
        pl = pathloss_log_distance_db(d, cfg)
        sigma = cfg.shadowing_std_db
    elif cfg.pathloss_model == "3gpp_umi":
        los = rng.random(d.shape) < los_probability_umi(d)
        pl = pathloss_umi_db(d, cfg, los)
        sigma = np.where(los, 4.0, 7.82)
    else:
        raise ValueError(f"unknown pathloss_model {cfg.pathloss_model!r}")
    shadow = rng.normal(0, 1, d.shape) * sigma
    return 10 ** (-(pl + shadow) / 10)


def small_scale_fading(shape, cfg: Config, rng):
    """Power fading |h|^2 with E[|h|^2] = 1."""
    if cfg.fading == "none":
        return np.ones(shape)
    if cfg.fading == "rayleigh":
        return rng.exponential(1.0, shape)
    if cfg.fading == "rician":
        K = 10 ** (cfg.rician_k_db / 10)
        los = np.sqrt(K / (K + 1))
        nlos = np.sqrt(1 / (2 * (K + 1))) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
        return np.abs(los + nlos) ** 2
    raise ValueError(f"unknown fading {cfg.fading!r}")


def sinr(G, H, omega, cfg: Config):
    """SINR[b,u,m] if RU b serves u on PRB m with power omega[b,u,m].

    G: (B,U), H: (B,U,M), omega: (B,U,M) [W] (zero where PRB not allocated).
    Interference on PRB m at user u comes from every other RU's total power on m.
    """
    gain = G[:, :, None] * H                   # (B,U,M)
    signal = omega * gain
    noise = cfg.noise_per_prb_w
    if not cfg.interference:
        return signal / noise
    p_ru = omega.sum(axis=1)                   # (B,M) power each RU radiates on PRB m
    rx_all = np.einsum("bm,bum->um", p_ru, gain)  # (U,M) total received from all RUs
    own = p_ru[:, None, :] * gain              # (B,U,M) received from RU b itself
    return signal / (noise + rx_all[None] - own)


def prb_rate(sinr_bum, cfg: Config):
    """R_{b,u,m} [bit/s] via Shannon on one PRB."""
    return cfg.prb_bandwidth_hz * np.log2(1 + sinr_bum)
