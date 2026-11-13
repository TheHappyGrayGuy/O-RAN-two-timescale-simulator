"""Network environment with the two time scales of ch4/ch5.

large step (placement + association agent): users move, large-scale gain redrawn
small slot (PRB + power agent): small-scale fading redrawn
"""
import numpy as np

from .channel import large_scale_gain, small_scale_fading
from .config import Config
from .kpis import evaluate
from .topology import build_topology
from .users import build_users


class Network:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # topology gets its own stream so it stays fixed when UE count/channel settings change
        self.topo = build_topology(cfg, np.random.default_rng([cfg.seed, 1]))
        self.users = build_users(cfg, self.rng)
        self.t_slot = 0
        self._redraw_large()
        self._redraw_small()

    def _redraw_large(self):
        self.G = large_scale_gain(self.topo.ru_pos, self.users.pos, self.cfg, self.rng)

    def _redraw_small(self):
        self.H = small_scale_fading((self.topo.B, self.users.U, self.cfg.n_prb), self.cfg, self.rng)

    def large_step(self):
        dt = self.cfg.slot_s * self.cfg.slots_per_large_step
        self.users.move(dt, self.rng)
        self._redraw_large()
        self._redraw_small()

    def small_step(self):
        self.t_slot += 1
        if self.t_slot % self.cfg.slots_per_large_step == 0:
            self.large_step()
        else:
            self._redraw_small()

    def evaluate(self, decision):
        return evaluate(self.topo, self.users, self.G, self.H, decision)
