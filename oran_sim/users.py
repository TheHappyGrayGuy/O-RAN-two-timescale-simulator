"""User equipment: spatial drop, slice membership, traffic and mobility."""
from dataclasses import dataclass

import numpy as np

from .config import Config


def drop_positions(cfg: Config, rng, n=None):
    n = cfg.n_ue if n is None else n
    L = cfg.L_max
    if cfg.ue_distribution == "uniform":
        return rng.uniform(0, L, (n, 2))
    if cfg.ue_distribution == "clustered":
        # Thomas-type process: hotspot centres uniform, users Gaussian around them
        centres = rng.uniform(0.1 * L, 0.9 * L, (cfg.n_clusters, 2))
        which = rng.integers(cfg.n_clusters, size=n)
        return np.clip(centres[which] + rng.normal(0, cfg.cluster_std_m, (n, 2)), 0, L)
    raise ValueError(f"unknown ue_distribution {cfg.ue_distribution!r}")


def assign_slices(cfg: Config, rng, n=None):
    n = cfg.n_ue if n is None else n
    S = len(cfg.slices)
    if cfg.slice_assignment == "balanced":
        return np.arange(n) % S
    if cfg.slice_assignment == "random":
        return rng.choice(S, size=n, p=np.asarray(cfg.slice_probs))
    raise ValueError(f"unknown slice_assignment {cfg.slice_assignment!r}")


@dataclass
class Users:
    cfg: Config
    pos: np.ndarray        # (U, 2) L_u
    slice_idx: np.ndarray  # (U,) s(u)
    heading: np.ndarray    # (U,) rad

    @property
    def U(self):
        return len(self.pos)

    def _per_slice(self, values):
        return np.asarray(values, dtype=float)[self.slice_idx]

    @property
    def lam(self):        # lambda_{u_s} [packets/s]
        return self._per_slice([s.arrival_rate for s in self.cfg.slices])

    @property
    def q_bits(self):     # q_{u_s} [bit]
        return self._per_slice(self.cfg.packet_bits_per_slice)

    @property
    def d_qos(self):      # D^QoS_{u_s} [s]
        return self._per_slice([s.d_qos_s for s in self.cfg.slices])

    @property
    def speed(self):
        return self._per_slice([s.speed_mps for s in self.cfg.slices])

    @property
    def offered_bps(self):
        return self.lam * self.q_bits

    def users_of_slice(self, s):
        return np.flatnonzero(self.slice_idx == s)

    def move(self, dt, rng):
        """Random-direction walk with heading jitter; reflects at the area border."""
        if self.cfg.mobility == "static":
            return
        self.heading = self.heading + rng.normal(0, 0.3, self.U)
        step = (self.speed * dt)[:, None] * np.c_[np.cos(self.heading), np.sin(self.heading)]
        p = self.pos + step
        L = self.cfg.L_max
        for k in range(2):
            low, high = p[:, k] < 0, p[:, k] > L
            p[low, k] = -p[low, k]
            p[high, k] = 2 * L - p[high, k]
            self.heading[low | high] = (np.pi - self.heading[low | high]) if k == 0 else -self.heading[low | high]
        self.pos = np.clip(p, 0, L)

    def poisson_arrivals(self, dt, rng):
        """Packets arriving per user in an interval dt (Poisson, mean lambda*dt)."""
        return rng.poisson(self.lam * dt)


def build_users(cfg: Config, rng) -> Users:
    return Users(cfg, drop_positions(cfg, rng), assign_slices(cfg, rng),
                 rng.uniform(0, 2 * np.pi, cfg.n_ue))
