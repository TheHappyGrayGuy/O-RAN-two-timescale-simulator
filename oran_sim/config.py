"""Simulation parameters for the Chapter 4 O-RAN system model.

All quantities are SI internally (m, s, Hz, W, bit, cycles/s).
Defaults follow the Chapter 6 parameter table (Table_parameters_s); every value
that the thesis does not specify is marked ASSUMPTION and listed in ROADMAP.md.
"""
from dataclasses import dataclass, replace

import numpy as np

SLICE_NAMES = ("eMBB", "URLLC", "mMTC")
CU, DU = 0, 1  # layer index l in {CU, DU}
C_LIGHT = 3e8


@dataclass(frozen=True)
class SliceProfile:
    name: str
    d_qos_s: float        # D^QoS (ch6: 25, 5, 1000 ms)
    packet_size: float    # q, in units of Config.packet_size_unit (ch6: 1500, 200, 20)
    arrival_rate: float   # lambda, packets/s (ch6: 100, 50, 1)
    delta: float          # priority weight delta_s in the objective (ASSUMPTION)
    speed_mps: float      # mean UE speed (ASSUMPTION)


DEFAULT_SLICES = (
    SliceProfile("eMBB", 25e-3, 1500, 100, 0.5, 1.5),
    SliceProfile("URLLC", 5e-3, 200, 50, 0.9, 1.5),
    SliceProfile("mMTC", 1000e-3, 20, 1, 0.1, 0.0),
)

_PACKET_UNIT_BITS = {"bit": 1.0, "byte": 8.0, "kbit": 1e3}


@dataclass(frozen=True)
class Config:
    seed: int = 0

    # --- geometry (ch4: L_u, L_r in [0,Lmax]^2; edge in [Lmax,L'max]^2; regional in [L'max,L''max]^2)
    L_max: float = 1000.0            # ch6: 1000 m square
    L_edge_max: float = 2000.0       # L'max   (ASSUMPTION)
    L_regional_max: float = 20000.0  # L''max  (ASSUMPTION)

    # --- radio units
    n_ru: int = 4                    # ch6
    ru_layout: str = "grid"          # "grid" | "random"
    ru_min_separation_m: float = 200.0
    ru_height_m: float = 10.0        # used by 3GPP model (ASSUMPTION)

    # --- O-Cloud
    n_edge: int = 4                  # ch6 "2, 4" read as N^r=2, N^e=4 (ASSUMPTION on order)
    n_regional: int = 2
    cpu_edge: float = 3.4e9          # ch6 [cycles/s]
    cpu_regional: float = 10 * 3.4e9
    cpu_per_bps: tuple = (0.00175 * 3.4e9 / 1e6, 0.04 * 3.4e9 / 1e6)  # (CU, DU) cycles/s per bit/s, ch6
    core_hz: float = 3.4e9           # ch6 "10*3.4 GHz" read as cores x clock: edge 1 core, regional 10
    # processing delay model (ROADMAP decision D2):
    #   "node_queue": each O-Cloud node is an M/M/c queue shared by all hosted CU/DU functions;
    #                 a packet of q bits needs c~^l q cycles on one core (default)
    #   "thesis":     eq.20 as written, CPU allocated = demand (delay independent of placement)
    #   "proportional": eq.20 with node capacity split in proportion to demand
    processing_model: str = "node_queue"
    gamma_bits_per_cycle: float = 1.0  # gamma, used only by "thesis"/"proportional" (not given)
    fronthaul_link_bps: float = 100e9  # ch6 "100 GHz, 10 GHz" -> RU-edge / edge-regional (ASSUMPTION on order)
    midhaul_link_bps: float = 10e9
    w_tilde_mid: float = 1.0         # ch6: 1 Mbps per 1 Mbps
    w_tilde_front: float = 10.0      # ch6: 10 Mbps per 1 Mbps
    fiber_speed: float = 2e8         # m/s
    fiber_route_factor: float = 1.0  # fiber length / straight-line distance
    cost_unit_v: float = 1.0         # v in phi_c (ASSUMPTION)
    edge_cost_factor: float = 10.0   # phi^e = 10 * c/C * v (ch4)
    regional_cost_factor: float = 1.0
    fronthaul_budget_s: float = 100e-6  # one-way RU-DU latency budget, O-RAN split 7.2x (D2); None disables

    # --- PHY
    carrier_hz: float = 3.5e9        # ASSUMPTION (needed for reference path loss)
    bandwidth_hz: float = 25e6       # ch6
    n_prb: int = 25                  # M (ASSUMPTION; varied in ch6)
    p_max_dbm: float = 40.0          # ch6 (40 / 60 dBm in sweeps)
    noise_psd_dbm_hz: float = -174.0  # ch6
    noise_figure_db: float = 0.0     # ASSUMPTION (thesis has none)
    pathloss_model: str = "log_distance"  # "log_distance" (ch6) | "3gpp_umi"
    pathloss_exp: float = 3.0        # ch6
    ref_distance_m: float = 1.0
    shadowing_std_db: float = 0.0    # log_distance only; 3GPP uses its own sigma
    fading: str = "rayleigh"         # ch6 | "none" | "rician"
    rician_k_db: float = 6.0
    interference: bool = False       # ch6 uses SNR (Shannon P/N); True adds inter-RU co-channel interference
    min_distance_m: float = 10.0
    ue_height_m: float = 1.5

    # --- users and traffic
    n_ue: int = 10                   # ch6: 5, 10, 15, 20
    ue_distribution: str = "uniform"  # ch6 "uniform" | "clustered"
    n_clusters: int = 3
    cluster_std_m: float = 80.0
    slice_assignment: str = "balanced"  # "balanced" (round robin) | "random"
    slice_probs: tuple = (1 / 3, 1 / 3, 1 / 3)
    slices: tuple = DEFAULT_SLICES
    packet_size_unit: str = "byte"   # ch6 table says "Kb"; bytes chosen, see ROADMAP.md decision D1
    mobility: str = "random_walk"    # "static" | "random_walk"

    # --- time (two-timescale framework)
    slot_s: float = 1e-3             # theta = one NR slot at 15 kHz SCS = small-timescale step (D3)
    slots_per_large_step: int = 10   # ASSUMPTION
    delay_unit_in_objective: float = 1e3  # objective_norm="raw": delay expressed in ms (ASSUMPTION)
    objective_norm: str = "qos"      # "qos": D/D^QoS and phi/phi_max (D2) | "raw": ch4 as written

    # ---------- derived ----------
    @property
    def n_nodes(self):
        return self.n_edge + self.n_regional

    @property
    def prb_bandwidth_hz(self):
        return self.bandwidth_hz / self.n_prb

    @property
    def p_max_w(self):
        return dbm_to_w(self.p_max_dbm)

    @property
    def noise_per_prb_w(self):
        return dbm_to_w(self.noise_psd_dbm_hz + 10 * np.log10(self.prb_bandwidth_hz) + self.noise_figure_db)

    @property
    def packet_bits_per_slice(self):
        k = _PACKET_UNIT_BITS[self.packet_size_unit]
        return tuple(s.packet_size * k for s in self.slices)

    def with_(self, **kw):
        return replace(self, **kw)


def dbm_to_w(dbm):
    return 10 ** ((dbm - 30) / 10)


def w_to_dbm(w):
    return 10 * np.log10(w) + 30

