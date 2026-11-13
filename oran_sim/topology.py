"""Telecom topology: RU sites and the O-Cloud graph G = (N, E) of ch4.

O-Cloud nodes are indexed n = 0..N-1 with the N^e edge nodes first, then the
N^r regional nodes. Graph vertices are ("ru", b), ("edge", i), ("reg", j).
Links: RU-edge (fronthaul tier) and edge-regional (midhaul tier), full bipartite
in each tier; routes are shortest fiber paths.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .config import Config


def ru_positions(cfg: Config, rng):
    B, L = cfg.n_ru, cfg.L_max
    if cfg.ru_layout == "grid":
        cols = int(np.ceil(np.sqrt(B)))
        rows = int(np.ceil(B / cols))
        xs = (np.arange(cols) + 0.5) * L / cols
        ys = (np.arange(rows) + 0.5) * L / rows
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:B]
    if cfg.ru_layout == "random":
        pts = []
        for _ in range(10000):
            p = rng.uniform(0, L, 2)
            if all(np.hypot(*(p - q)) >= cfg.ru_min_separation_m for q in pts):
                pts.append(p)
                if len(pts) == B:
                    return np.array(pts)
        raise ValueError("cannot place RUs with the requested minimum separation")
    raise ValueError(f"unknown ru_layout {cfg.ru_layout!r}")


def cloud_positions(cfg: Config, rng):
    edge = rng.uniform(cfg.L_max, cfg.L_edge_max, (cfg.n_edge, 2))
    reg = rng.uniform(cfg.L_edge_max, cfg.L_regional_max, (cfg.n_regional, 2))
    return edge, reg


@dataclass
class Topology:
    cfg: Config
    ru_pos: np.ndarray        # (B, 2)
    node_pos: np.ndarray      # (N, 2)
    node_is_edge: np.ndarray  # (N,) bool
    cpu_capacity: np.ndarray  # (N,) C_n
    n_cores: np.ndarray       # (N,) C_n / core clock
    cost_factor: np.ndarray   # (N,) 10 for edge, 1 for regional (phi_c multiplier)
    graph: nx.Graph
    links: list               # link e -> (u, v) graph vertices
    link_capacity: np.ndarray  # (E,) W_{p,p'}
    link_tier: np.ndarray     # (E,) "fronthaul" | "midhaul"
    p_front: np.ndarray       # (B, N, E) 1 if link e on route RU b <-> node n
    p_mid: np.ndarray         # (N, N, E) 1 if link e on route node n <-> node n'
    prop_front: np.ndarray    # (B, N) fiber delay RU b <-> node n [s]
    prop_mid: np.ndarray      # (N, N) fiber delay node n <-> node n' [s]

    @property
    def B(self):
        return len(self.ru_pos)

    @property
    def N(self):
        return len(self.node_pos)

    @property
    def edge_nodes(self):
        return np.flatnonzero(self.node_is_edge)

    @property
    def regional_nodes(self):
        return np.flatnonzero(~self.node_is_edge)


def _vertex(n, n_edge):
    return ("edge", n) if n < n_edge else ("reg", n - n_edge)


def build_topology(cfg: Config, rng=None) -> Topology:
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    ru = ru_positions(cfg, rng)
    edge, reg = cloud_positions(cfg, rng)
    node_pos = np.vstack([edge, reg])
    Ne, N, B = cfg.n_edge, cfg.n_nodes, cfg.n_ru

    G = nx.Graph()
    pos = {("ru", b): ru[b] for b in range(B)}
    pos.update({_vertex(n, Ne): node_pos[n] for n in range(N)})
    for v, p in pos.items():
        G.add_node(v, pos=p)

    links, cap, tier = [], [], []

    def add(u, v, c, t):
        length = cfg.fiber_route_factor * float(np.hypot(*(pos[u] - pos[v])))
        G.add_edge(u, v, length=length, capacity=c, tier=t, idx=len(links))
        links.append((u, v))
        cap.append(c)
        tier.append(t)

    for b in range(B):
        for i in range(Ne):
            add(("ru", b), ("edge", i), cfg.fronthaul_link_bps, "fronthaul")
    for i in range(Ne):
        for j in range(cfg.n_regional):
            add(("edge", i), ("reg", j), cfg.midhaul_link_bps, "midhaul")
    E = len(links)

    def route(src, dst):
        path = nx.shortest_path(G, src, dst, weight="length")
        inc = np.zeros(E)
        length = 0.0
        for u, v in zip(path, path[1:]):
            d = G.edges[u, v]
            inc[d["idx"]] = 1
            length += d["length"]
        return inc, length / cfg.fiber_speed

    p_front = np.zeros((B, N, E))
    prop_front = np.zeros((B, N))
    for b in range(B):
        for n in range(N):
            p_front[b, n], prop_front[b, n] = route(("ru", b), _vertex(n, Ne))
    p_mid = np.zeros((N, N, E))
    prop_mid = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            if n != m:
                p_mid[n, m], prop_mid[n, m] = route(_vertex(n, Ne), _vertex(m, Ne))

    is_edge = np.arange(N) < Ne
    return Topology(
        cfg=cfg, ru_pos=ru, node_pos=node_pos, node_is_edge=is_edge,
        cpu_capacity=np.where(is_edge, cfg.cpu_edge, cfg.cpu_regional),
        n_cores=np.maximum(1, np.round(np.where(is_edge, cfg.cpu_edge, cfg.cpu_regional) / cfg.core_hz)).astype(int),
        cost_factor=np.where(is_edge, cfg.edge_cost_factor, cfg.regional_cost_factor),
        graph=G, links=links, link_capacity=np.array(cap), link_tier=np.array(tier),
        p_front=p_front, p_mid=p_mid, prop_front=prop_front, prop_mid=prop_mid,
    )
