"""Phase 5: optimal solutions of the ch4 problem and certified lower bounds.

Structure used (valid when cfg.interference is False, the ch6 setting):
  J(alpha, kappa, omega, psi) = sum_b V_b(kappa_b, omega_b; alpha, split_b) + I(psi; alpha)
where V_b is the rate-dependent part of RU b (radio delay T_u and fronthaul/midhaul
transmission delay, which only sees whether CU and DU of b are split) and I is the
rate-independent part (fiber/air propagation, node-queue processing, cost).

Blocks and how each is solved:
  power omega_b | kappa_b      convex (1/(x - lambda) convex decreasing, rate concave) -> Clarabel, global
  PRBs kappa_b | alpha         exhaustive for small RUs; greedy + local search otherwise
  placement psi | alpha        exhaustive vectorized enumeration of all N^(2B) placements
  association alpha            exhaustive (tiny) or multi-start local search (full size)
Lower bounds: fractional PRB/association with perspective rate (convex) for the radio part,
and a per-user relaxation of queueing for the infrastructure part.
"""
import itertools
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .config import C_LIGHT, CU, DU
from .kpis import Decision, erlang_c

LN2 = np.log(2)
INF = np.inf
LB_SAFETY = 1e-5   # relative margin for solver tolerance when a convex optimum is used as a lower bound


def objective_weights(net):
    """J = sum_u w_u D_e2e_u + K phi_bar, matching kpis.evaluate."""
    cfg, u = net.cfg, net.users
    S = len(cfg.slices)
    delta = np.array([s.delta for s in cfg.slices])
    counts = np.bincount(u.slice_idx, minlength=S)
    if cfg.objective_norm == "qos":
        w = delta[u.slice_idx] / (counts[u.slice_idx] * u.d_qos)
        K = np.sum(1 - delta) / (2 * net.topo.cost_factor.max() * cfg.cost_unit_v)
    else:
        w = delta[u.slice_idx] * cfg.delay_unit_in_objective / counts[u.slice_idx]
        K = np.sum(1 - delta)
    return w, K


def _solve(prob):
    """Clarabel first; retry with tighter tolerances, then SCS, if not accurate."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({"solver": cp.CLARABEL},
                   {"solver": cp.CLARABEL, "tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "max_iter": 500},
                   {"solver": cp.SCS, "eps": 1e-9, "max_iters": 200000}):
            try:
                prob.solve(**kw)
            except cp.SolverError:
                continue
            if prob.status == "optimal":
                return prob.status
    return prob.status


def _queue_terms(x, lam, w):
    """w/(x - lam) written as (w/lam) * 1/(x/lam - 1) for better conditioning."""
    return (w / lam) * cp.inv_pos(x / lam - 1)


def _min_delay_plus_linear(wq, lamq, nu):
    """min_R  wq/(R - lamq) + nu R  over R > lamq  =  2 sqrt(wq nu) + nu lamq  (nu > 0)."""
    return 2 * np.sqrt(wq * nu) + nu * lamq


def _min_transport_plus_linear(W, qbar, lam_b, wf, wm, split, nu0):
    """min_Z  W/(wf Z/qbar - lam_b) + split W/(wm Z/qbar - lam_b) + nu0 Z; returns (value, argmin Z)."""
    if W == 0:
        return 0.0, 0.0
    zmin = lam_b * qbar / (wm if split else wf)            # stability limit of the binding term
    f = lambda z: W / (wf * z / qbar - lam_b) + (W / (wm * z / qbar - lam_b) if split else 0.0) + nu0 * z
    df = lambda z: (-W * wf / qbar / (wf * z / qbar - lam_b) ** 2
                    - (W * wm / qbar / (wm * z / qbar - lam_b) ** 2 if split else 0.0) + nu0)
    lo, hi = zmin * (1 + 1e-12) + 1e-30, max(zmin * 2, 1.0)
    while df(hi) < 0:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if df(mid) < 0 else (lo, mid)
    z = 0.5 * (lo + hi)
    return f(z), z


def radio_dual_bound(a, allowed, q, lam, w, Bp, transport=None, init=None, restarts=2):
    """Certified lower bound (weak duality) on

        min sum_i w_i/(R_i/q_i - lam_i) [+ transport(sum_i R_i)]
        s.t. R_i <= sum_{b,m} X_bim Bp log2(1 + a_bim S_bim / X_bim),
             sum_i X_bim <= 1 (each PRB), sum_{i,m} S_bim <= 1 (each RU's power), X, S >= 0,
             X_bim = 0 where not allowed[b,i],
             sum_{b,m} X_bim >= 1 (valid cut: a user without a whole PRB has an unstable queue)

    which relaxes integer PRBs, integer association and power jointly. The dual function has a
    closed form; any multipliers give a valid bound. It is maximized with L-BFGS on a soft-min
    smoothing (soft-min <= min, so the smoothed dual is itself a valid bound) with an annealed
    temperature, and the exact dual is evaluated at every iterate.
    a: (B,n,M) = P_b * gain / noise. transport: dict(W, qbar, lam_b, wf, wm, split) for a single RU.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    B, n, M = a.shape
    c = Bp / LN2
    wq, lamq = w * q, lam * q
    has_tr = transport is not None and transport["W"] > 0
    mask = allowed[:, :, None] & (a > 0)
    inv_a = np.where(mask, 1.0 / np.where(mask, a, 1.0), np.inf)
    a0 = np.where(mask, a, 0.0)

    def parts(theta):
        theta = np.clip(theta, -700, 700)
        nu = np.exp(theta[:n])
        nu0 = np.exp(theta[n]) if has_tr else 0.0
        mu = np.exp(theta[n + 1:n + 1 + B])
        eta = eta_scale[0] * np.maximum(theta[n + 1 + B:], 0.0)       # direct (not log) so it can leave 0
        nue = nu + nu0
        t = np.maximum(0.0, (nue * c)[None, :, None] / mu[:, None, None] - inv_a)          # optimal power density
        lg = np.log1p(a0 * t)
        phi = mu[:, None, None] * t - (nue * c)[None, :, None] * lg - eta[None, :, None]
        phi = np.where(mask, phi, np.inf)
        const = _min_delay_plus_linear(wq, lamq, nu).sum() - mu.sum() + eta.sum()
        tr_val, z = _min_transport_plus_linear(nu0=nu0, **transport) if has_tr else (0.0, 0.0)
        return nu, nu0, mu, eta, t, lg, phi, const + tr_val, z

    def exact(theta):
        *_, phi, const, _z = parts(theta)
        return float(const + np.minimum(phi.min(axis=1), 0.0).sum())

    def smooth(theta, tau):
        nu, nu0, mu, eta, t, lg, phi, const, z = parts(theta)
        stack = np.concatenate([np.zeros((B, 1, M)), -phi / tau], axis=1)                  # option 0 = leave PRB idle
        lse = logsumexp(stack, axis=1)                                                     # (B,M)
        val = const - tau * lse.sum()
        p = np.exp(stack[:, 1:, :] - lse[:, None, :])                                      # soft argmin weights
        p = np.where(mask, p, 0.0)
        d_nue = -(p * c * lg).sum(axis=(0, 2))                                             # (n,)
        g_nu = np.sqrt(wq / nu) + lamq + d_nue
        g_mu = -1 + (p * t).sum(axis=(1, 2))
        g_eta = 1 - p.sum(axis=(0, 2))
        grad = np.r_[g_nu * nu, [(z + d_nue.sum()) * nu0 if has_tr else 0.0], g_mu * mu, g_eta * eta_scale[0]]
        return val, grad

    old = np.seterr(all="ignore")
    rng = np.random.default_rng(0)
    eta_scale = [1.0]
    base = np.r_[np.log(np.maximum(wq, 1e-300)) - 2 * np.log(np.maximum(lamq, 1.0)) - 2, [-20.0],
                 np.zeros(B), np.zeros(n)]
    starts = [] if init is None else [np.r_[init, np.zeros(n)] if len(init) == n + 1 + B else init]
    bounds = [(None, None)] * (n + 1 + B) + [(0.0, None)] * n
    starts += [base] + [base + rng.normal(0, 1.5, base.shape) for _ in range(restarts)]
    best = -INF
    for th in starts:
        th = np.asarray(th, float)
        eta_scale[0] = 1.0
        th[n + 1 + B:] = 0.0
        scale = max(abs(exact(th)), 1e-12)
        eta_scale[0] = scale / M                                          # natural size of a PRB's dual value
        best = max(best, exact(th))
        for k in range(9):
            tau = scale * 10.0 ** (-1 - k)
            res = minimize(lambda x: tuple(-v for v in smooth(x, tau)), th, jac=True, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 2000, "gtol": 1e-12, "ftol": 1e-15})
            th = res.x
            e = exact(th)
            if np.isfinite(e):
                best = max(best, e)
            scale = max(abs(best), 1e-12)
    np.seterr(**old)
    return best * (1 - 1e-9) if best > 0 else best


def _inv(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 1.0 / np.where(x > 0, x, 1.0), INF)


# ============================================================== radio block (one RU)
@dataclass
class RadioSolution:
    value: float          # V_b
    owner: np.ndarray     # (M,) global user index per PRB, -1 unused
    power: np.ndarray     # (M,) W
    rate: np.ndarray      # (|users|,) bit/s
    t_radio: np.ndarray   # (|users|,) 1/(R/q - lambda)
    tr_front: float
    tr_mid: float         # 0 when not split


class RadioRU:
    def __init__(self, net, b, users, w):
        cfg = net.cfg
        self.cfg, self.b = cfg, b
        self.users = np.asarray(users, int)
        u = net.users
        self.q, self.lam, self.w = u.q_bits[self.users], u.lam[self.users], w[self.users]
        self.M, self.P, self.Bp = cfg.n_prb, cfg.p_max_w, cfg.prb_bandwidth_hz
        self.snr1 = (net.G[b, self.users, None] * net.H[b][self.users]) / cfg.noise_per_prb_w  # per W
        self.W = self.w.sum()
        self.lam_b = self.lam.sum()
        self.qbar = (self.lam * self.q).sum() / self.lam_b if len(self.users) else 0.0
        self.wf, self.wm = cfg.w_tilde_front, cfg.w_tilde_mid

    # ---- value from per-user rates
    def value_from_rates(self, R, split):
        t = _inv(R / self.q - self.lam)
        Rb = R.sum()
        tf = _inv(self.wf * Rb / self.qbar - self.lam_b)
        tm = _inv(self.wm * Rb / self.qbar - self.lam_b) if split else 0.0
        return float(self.w @ t + self.W * (tf + tm)), t, float(tf), float(tm)

    def rates(self, owner_local, power):
        r = np.zeros(len(self.users))
        used = owner_local >= 0
        m = np.flatnonzero(used)
        np.add.at(r, owner_local[m], self.Bp * np.log2(1 + power[m] * self.snr1[owner_local[m], m]))
        return r

    def _solution(self, owner_local, power, split):
        R = self.rates(owner_local, power)
        v, t, tf, tm = self.value_from_rates(R, split)
        owner = np.where(owner_local >= 0, self.users[np.maximum(owner_local, 0)], -1)
        return RadioSolution(v, owner, power, R, t, tf, tm)

    def empty(self):
        return RadioSolution(0.0, -np.ones(self.M, int), np.zeros(self.M), np.zeros(0), np.zeros(0), 0.0, 0.0)

    # ---- exact power for a fixed PRB assignment
    def optimal_power_cvx(self, owner_local, split):
        """Reference implementation with cvxpy/Clarabel (slow; used to validate optimal_power)."""
        n = len(self.users)
        p = cp.Variable(self.M, nonneg=True)                     # fraction of P_max
        x = []                                                   # R_u / q_u  [packets/s]
        for i in range(n):
            ms = np.flatnonzero(owner_local == i)
            if len(ms) == 0:
                return None
            coef = self.Bp / (LN2 * self.q[i])
            x.append(coef * cp.sum(cp.log(1 + cp.multiply(self.P * self.snr1[i, ms], p[ms]))))
        obj = sum(_queue_terms(x[i], self.lam[i], self.w[i]) for i in range(n))
        rb_over_qbar = sum(self.q[i] * x[i] for i in range(n)) / self.qbar
        obj += _queue_terms(self.wf * rb_over_qbar, self.lam_b, self.W)
        if split:
            obj += _queue_terms(self.wm * rb_over_qbar, self.lam_b, self.W)
        prob = cp.Problem(cp.Minimize(obj), [cp.sum(p) <= 1, p[owner_local < 0] == 0] if np.any(owner_local < 0)
                          else [cp.sum(p) <= 1])
        status = _solve(prob)
        if p.value is None or status not in ("optimal", "optimal_inaccurate"):
            return None
        power = np.clip(p.value, 0, None)
        power = power / max(power.sum(), 1.0) * self.P
        return self._solution(owner_local, power, split)

    def values_batch(self, Rs, split):
        """V_b for many rate vectors at once. Rs: (K, n) bit/s."""
        with np.errstate(divide="ignore", invalid="ignore"):
            x = Rs / self.q - self.lam
            t = np.where(x > 0, 1 / np.where(x > 0, x, 1), INF)
            Rb = Rs.sum(axis=1)
            v = t @ self.w
            for wt, on in ((self.wf, True), (self.wm, bool(split))):
                if on:
                    y = wt * Rb / self.qbar - self.lam_b
                    v = v + self.W * np.where(y > 0, 1 / np.where(y > 0, y, 1), INF)
        return v

    def optimal_power(self, owner_local, split, max_iter=400, tol=1e-13, golden_iters=60):
        """Exact power allocation for a fixed PRB assignment.

        The objective is convex in p. Each iteration linearizes it in the user rates, solves the
        resulting weighted sum-rate problem exactly (water-filling with user-specific weights,
        bisection on the power price) and does an exact line search towards that point. A fixed
        point satisfies the KKT conditions p_m = [kappa_o(m) c / mu - 1/a_m]^+, hence is optimal.
        """
        n = len(self.users)
        ms = np.flatnonzero(owner_local >= 0)
        o = owner_local[ms]
        if n == 0 or len(np.unique(o)) < n:
            return None
        a = self.P * self.snr1[o, ms]
        c = self.Bp / LN2

        def R_of(pf):
            return np.bincount(o, weights=c * np.log1p(a * pf), minlength=n)

        def F(pf):
            return self.values_batch(R_of(pf)[None], split)[0]

        def kappa(R):
            x = R / self.q - self.lam
            k = self.w / self.q / x ** 2
            Rb = R.sum()
            for wt, on in ((self.wf, True), (self.wm, bool(split))):
                if on:
                    k = k + self.W * wt / self.qbar / (wt * Rb / self.qbar - self.lam_b) ** 2
            return k

        def waterfill(weights):
            """Exact: p_m = max(0, l_m x - 1/a_m), sum p = 1, with l = weights*c and x = 1/mu.
            Sort PRBs by breakpoint 1/(a l); for the first k active, x = (1 + sum 1/a)/sum l."""
            l = weights * c
            order = np.argsort(1 / (a * l))
            bp = (1 / (a * l))[order]
            x_k = (1 + np.cumsum(1 / a[order])) / np.cumsum(l[order])
            k = np.flatnonzero(np.r_[x_k[:-1] < bp[1:], True])[0]     # first k whose level stays below next breakpoint
            pw = np.maximum(0.0, l * x_k[k] - 1 / a)
            return pw / pw.sum()

        onehot = np.eye(n)[o]                                          # (|ms|, n)

        def F_batch(P_rows):                                           # many power vectors at once
            return self.values_batch(np.log1p(a * P_rows) * c @ onehot, split)

        pf = np.full(len(ms), 1 / len(ms))
        f = F(pf)
        if not np.isfinite(f):
            pf = waterfill(np.ones(len(ms)))              # max sum-rate start
            f = F(pf)
            if not np.isfinite(f):
                return None
        for _ in range(max_iter):
            pw = waterfill(kappa(R_of(pf))[o])
            d = pw - pf
            if np.abs(d).max() < 1e-12:
                break
            lo_t, hi_t = 0.0, 1.0                          # batched zoom line search on the convex 1-D function
            for _ in range(3):
                ts = np.linspace(lo_t, hi_t, 65)
                v = F_batch(pf[None] + ts[:, None] * d[None])
                k = int(np.argmin(v))
                step = ts[1] - ts[0]
                lo_t, hi_t = max(0.0, ts[k] - step), min(1.0, ts[k] + step)
            t_best = ts[k]
            f_new = F(pf + t_best * d)
            if f_new >= f - tol * abs(f):
                if F(pw) < f:
                    pf, f = pw, F(pw)
                    continue
                break
            pf, f = pf + t_best * d, f_new
        power = np.zeros(self.M)
        power[ms] = pf * self.P
        return self._solution(owner_local, power, split)

    def equal_power(self):
        return np.full(self.M, self.P / self.M)

    # ---- PRB assignment
    def assign_exhaustive(self, split):
        """All |users|^M assignments (every PRB is used: an unused PRB never helps), exact power each."""
        n = len(self.users)
        if n == 0:
            return self.empty()
        best = None
        for combo in itertools.product(range(n), repeat=self.M):
            owner = np.array(combo)
            if len(set(combo)) < n:          # a user without PRB has infinite delay
                continue
            sol = self.optimal_power(owner, split)
            if sol is not None and (best is None or sol.value < best.value):
                best = sol
        return best

    def assign_local_search(self, split, power=None, owner=None):
        """Greedy marginal assignment, then best-improvement single-PRB moves (vectorized)."""
        n, M = len(self.users), self.M
        if n == 0:
            return -np.ones(M, int)
        power = self.equal_power() if power is None else power
        r = self.Bp * np.log2(1 + power[None, :] * self.snr1)       # (n, M)
        eye = np.eye(n)
        if owner is None:
            owner = -np.ones(M, int)
            R = np.zeros(n)
            free = set(range(M))
            for i in np.argsort(-self.w):                           # every user gets its best free PRB
                m = max(free, key=lambda m: r[i, m])
                owner[m] = i
                R[i] += r[i, m]
                free.discard(m)
            for m in sorted(free, key=lambda m: -r[:, m].max()):
                i = int(np.argmin(self.values_batch(R[None] + eye * r[:, m][:, None], split)))
                owner[m] = i
                R[i] += r[i, m]
        owner = owner.copy()
        R = np.bincount(owner, weights=r[owner, np.arange(M)], minlength=n).astype(float)
        cur = self.values_batch(R[None], split)[0]
        mm, jj = np.meshgrid(np.arange(M), np.arange(n), indexing="ij")
        mm, jj = mm.ravel(), jj.ravel()
        while True:
            ii = owner[mm]
            Rs = np.repeat(R[None], len(mm), axis=0)
            Rs[np.arange(len(mm)), ii] -= r[ii, mm]
            Rs[np.arange(len(mm)), jj] += r[jj, mm]
            v = self.values_batch(Rs, split)
            v[ii == jj] = INF
            k = int(np.argmin(v))
            if not v[k] < cur - 1e-15 * abs(cur):
                return owner
            owner[mm[k]], R, cur = jj[k], Rs[k], v[k]

    def refine(self, sol, split, top_k=12, max_rounds=200):
        """Exact-move improvement of a PRB assignment.

        Candidate single-PRB moves and pairwise PRB swaps are ranked with the current per-PRB
        powers; the top_k are re-evaluated with exact power and the best real improvement is
        accepted. Stops when no candidate improves.
        """
        n, M = len(self.users), self.M
        if n < 2 or sol is None or not np.isfinite(sol.value):
            return sol
        best = sol
        mm, jj = (a.ravel() for a in np.meshgrid(np.arange(M), np.arange(n), indexing="ij"))
        m1, m2 = (a.ravel() for a in np.meshgrid(np.arange(M), np.arange(M), indexing="ij"))
        keep = m1 < m2
        m1, m2 = m1[keep], m2[keep]
        for _ in range(max_rounds):
            own = np.searchsorted(self.users, best.owner)
            r = self.Bp * np.log2(1 + best.power[None, :] * self.snr1)
            R = np.bincount(own, weights=r[own, np.arange(M)], minlength=n).astype(float)
            cands, vals = [], []
            ii = own[mm]                                             # single moves
            Rs = np.repeat(R[None], len(mm), axis=0)
            Rs[np.arange(len(mm)), ii] -= r[ii, mm]
            Rs[np.arange(len(mm)), jj] += r[jj, mm]
            v = self.values_batch(Rs, split)
            v[ii == jj] = INF
            for k in np.argsort(v)[:top_k]:
                if np.isfinite(v[k]):
                    o2 = own.copy()
                    o2[mm[k]] = jj[k]
                    cands.append(o2)
                    vals.append(v[k])
            i1, i2 = own[m1], own[m2]                                # swaps
            Rs = np.repeat(R[None], len(m1), axis=0)
            ar = np.arange(len(m1))
            Rs[ar, i1] += r[i1, m2] - r[i1, m1]
            Rs[ar, i2] += r[i2, m1] - r[i2, m2]
            v = self.values_batch(Rs, split)
            v[i1 == i2] = INF
            for k in np.argsort(v)[:top_k]:
                if np.isfinite(v[k]):
                    o2 = own.copy()
                    o2[m1[k]], o2[m2[k]] = own[m2[k]], own[m1[k]]
                    cands.append(o2)
                    vals.append(v[k])
            improved = None
            for o2 in cands:
                sol2 = self.optimal_power(o2, split)
                if sol2 is not None and sol2.value < (improved or best).value * (1 - 1e-12):
                    improved = sol2
            if improved is None:
                return best
            best = improved
        return best

    def solve(self, split, exhaustive_limit=5000, refine=False):
        n = len(self.users)
        if n == 0:
            return self.empty()
        if n > self.M:
            return RadioSolution(INF, -np.ones(self.M, int), np.zeros(self.M), np.zeros(n), np.full(n, INF), INF, INF)
        if n ** self.M <= exhaustive_limit:
            return self.assign_exhaustive(split)
        owner = self.assign_local_search(split)
        best = self.optimal_power(owner, split)
        for _ in range(3):                                          # alternate PRB moves and exact power
            if best is None:
                break
            owner2 = self.assign_local_search(split, best.power, owner)
            if np.array_equal(owner2, owner):
                break
            sol = self.optimal_power(owner2, split)
            if sol is None or sol.value >= best.value - 1e-15:
                break
            owner, best = owner2, sol
        if best is None:
            return self._solution(owner, self.equal_power(), split)
        return self.refine(best, split) if refine else best

    def relaxation_lb(self, split, init_from=None):
        """Certified lower bound on V_b (fractional PRBs + free power), via the dual of the relaxation."""
        n = len(self.users)
        if n == 0:
            return 0.0
        a = (self.P * self.snr1)[None]
        tr = dict(W=self.W, qbar=self.qbar, lam_b=self.lam_b, wf=self.wf, wm=self.wm, split=bool(split))
        init = None
        if init_from is not None and np.all(np.isfinite(init_from.t_radio)):
            R = init_from.rate
            nu = self.w * self.q / np.maximum(R - self.lam * self.q, 1e-9) ** 2
            init = np.r_[np.log(nu), [np.log(nu.mean() * 1e-2)], [np.log(nu.mean() * self.Bp / LN2)]]
        return radio_dual_bound(a, np.ones((1, n), bool), self.q, self.lam, self.w, self.Bp, tr, init)


# ============================================================== placement block
class PlacementEvaluator:
    """Vectorized value of every placement (cu, du) for a fixed association and radio solutions."""

    def __init__(self, net, assoc, radio, w, K):
        cfg, t, u = net.cfg, net.topo, net.users
        self.cfg, self.t, self.B, self.N, self.U = cfg, t, t.B, t.N, u.U
        self.assoc, self.w, self.K = np.asarray(assoc), w, K
        self.V = np.array([[radio[b][s].value for s in (0, 1)] for b in range(self.B)])       # (B,2)
        self.trf = np.array([[radio[b][s].tr_front for s in (0, 1)] for b in range(self.B)])
        self.trm = np.array([[radio[b][s].tr_mid for s in (0, 1)] for b in range(self.B)])
        self.t_u = np.zeros((2, self.U))
        for b in range(self.B):
            idx = np.flatnonzero(self.assoc == b)
            for s in (0, 1):
                if len(idx):
                    self.t_u[s, idx] = radio[b][s].t_radio
        c = np.asarray(cfg.cpu_per_bps)
        self.S_u = c[:, None] * u.q_bits[None, :] / cfg.core_hz                               # (2,U)
        onehot = np.eye(self.B)[self.assoc].T                                                 # (B,U)
        self.lam_b = onehot @ u.lam
        self.load_b = onehot @ (u.lam * u.q_bits)
        self.work_lb = (onehot @ (u.lam[:, None] * self.S_u.T)).T                             # (2,B) cores
        self.qbar_b = np.divide(self.load_b, self.lam_b, out=np.zeros(self.B), where=self.lam_b > 0)
        self.air = np.linalg.norm(t.ru_pos[self.assoc] - u.pos, axis=1) / C_LIGHT
        self.d_qos = u.d_qos

    def evaluate(self, cu, du, radio_constraints=True):
        """cu, du: (C,B) int. Returns (value (C,), feasible (C,) bool)."""
        cfg, t, B, N = self.cfg, self.t, self.B, self.N
        cu, du = np.atleast_2d(cu), np.atleast_2d(du)
        C = len(cu)
        split = (cu != du).astype(int)
        ar = np.arange(B)
        radio = np.take_along_axis(np.broadcast_to(self.V, (C, B, 2)), split[..., None], 2)[..., 0].sum(1)
        trf = self.trf[ar, split]
        trm = self.trm[ar, split]
        prop_b = t.prop_mid[cu, du] + t.prop_front[ar, du]                                    # (C,B)
        nodes = np.stack([cu, du], axis=1)                                                    # (C,2,B)
        work = np.zeros((C, N))
        lam_n = np.zeros((C, N))
        for l in (CU, DU):
            for b in range(B):
                np.add.at(work, (np.arange(C), nodes[:, l, b]), self.work_lb[l, b])
                np.add.at(lam_n, (np.arange(C), nodes[:, l, b]), self.lam_b[b])
        wait = np.zeros((C, N))
        stable = np.ones(C, bool)
        for n in range(N):
            c, a = t.n_cores[n], work[:, n]
            bad = a >= c
            stable &= ~bad
            ok = ~bad & (lam_n[:, n] > 0)
            if np.any(ok):
                aa = a[ok]
                terms = np.cumprod(np.c_[np.ones(len(aa)), aa[:, None] / np.arange(1, c + 1)], axis=1)
                top = terms[:, c] * c / (c - aa)
                pc = top / (terms[:, :c].sum(1) + top)
                wait[ok, n] = pc * (aa / lam_n[ok, n]) / (c - aa)
            wait[bad, n] = INF
        ub = self.assoc
        node_u = nodes[:, :, ub]                                                              # (C,2,U)
        wait_u = np.take_along_axis(wait, node_u[:, CU], 1) + np.take_along_axis(wait, node_u[:, DU], 1)
        proc_u = wait_u + self.S_u.sum(0)
        infra = (self.w * (prop_b[:, ub] + proc_u + self.air)).sum(1)
        phi_n = t.cost_factor * work * cfg.core_hz / t.cpu_capacity * cfg.cost_unit_v           # (C,N)
        phi_bar = (np.take_along_axis(phi_n, cu, 1) + np.take_along_axis(phi_n, du, 1)).sum(1) / B
        value = radio + infra + self.K * phi_bar
        feas = stable.copy()
        fh = t.prop_front[ar, du] + (trf if radio_constraints else 0.0)
        if cfg.fronthaul_budget_s is not None:
            feas &= np.all(fh <= cfg.fronthaul_budget_s, axis=1)
        if radio_constraints:
            with np.errstate(invalid="ignore"):
                cu_soj = np.where(self.lam_b > 0, np.take_along_axis(wait, cu, 1)
                                  + np.asarray(cfg.cpu_per_bps)[CU] * self.qbar_b / cfg.core_hz, 0.0)
            sched = (split == 0) | (t.prop_mid[cu, du] + cu_soj + trm <= cfg.slot_s)
            feas &= np.all(sched, axis=1)
            e2e = ((prop_b + trf + trm)[:, ub] + proc_u + self.air
                   + self.t_u[split[:, ub], np.arange(self.U)])
            feas &= np.all(e2e < self.d_qos, axis=1)
        value = np.where(np.isfinite(value), value, INF)
        return value, feas

    def best_exhaustive(self, chunk=200_000, radio_constraints=True):
        B, N = self.B, self.N
        total = N ** (2 * B)
        best = (INF, None, None)
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total))
            digits = np.stack(np.unravel_index(idx, (N,) * (2 * B)), axis=1)
            cu, du = digits[:, :B], digits[:, B:]
            v, f = self.evaluate(cu, du, radio_constraints)
            v = np.where(f, v, INF)
            i = int(np.argmin(v))
            if v[i] < best[0]:
                best = (float(v[i]), cu[i].copy(), du[i].copy())
        return best

    def best_coordinate(self, cu, du, radio_constraints=True):
        cu, du = np.array(cu), np.array(du)
        v0, f0 = self.evaluate(cu[None], du[None], radio_constraints)
        cur = v0[0] if f0[0] else INF
        pairs = np.array(list(itertools.product(range(self.N), repeat=2)))
        improved = True
        while improved:
            improved = False
            for b in range(self.B):
                CU_, DU_ = np.tile(cu, (len(pairs), 1)), np.tile(du, (len(pairs), 1))
                CU_[:, b], DU_[:, b] = pairs[:, 0], pairs[:, 1]
                v, f = self.evaluate(CU_, DU_, radio_constraints)
                v = np.where(f, v, INF)
                i = int(np.argmin(v))
                if v[i] < cur - 1e-15:
                    cur, cu, du, improved = float(v[i]), CU_[i].copy(), DU_[i].copy(), True
        return cur, cu, du


# ============================================================== full solver
@dataclass
class Result:
    value: float
    decision: Decision
    assoc: np.ndarray
    cu: np.ndarray
    du: np.ndarray
    lower_bound: float = -INF
    certified: bool = False
    info: dict = field(default_factory=dict)

    @property
    def gap(self):
        return (self.value - self.lower_bound) / self.value if np.isfinite(self.lower_bound) else INF


class Solver:
    def __init__(self, net, exhaustive_limit=5000, refine_in_search=False):
        if net.cfg.interference:
            raise ValueError("decomposition requires interference=False")
        self.net = net
        self.w, self.K = objective_weights(net)
        self.lim = exhaustive_limit
        self.refine = refine_in_search
        self._radio = {}

    def radio(self, b, users):
        key = (b, tuple(users))
        if key not in self._radio:
            ru = RadioRU(self.net, b, users, self.w)
            self._radio[key] = (ru.solve(0, self.lim, self.refine), ru.solve(1, self.lim, self.refine))
        return self._radio[key]

    def polish(self, assoc):
        """Replace the cached radio solutions of an association by exact-move refined ones."""
        for b in range(self.net.topo.B):
            users = tuple(np.flatnonzero(np.asarray(assoc) == b))
            ru = RadioRU(self.net, b, users, self.w)
            old = self.radio(b, users)
            self._radio[(b, users)] = tuple(
                min((ru.refine(old[sp], sp), old[sp]), key=lambda x: x.value) for sp in (0, 1))

    def radio_all(self, assoc):
        return [self.radio(b, tuple(np.flatnonzero(assoc == b))) for b in range(self.net.topo.B)]

    def evaluator(self, assoc):
        return PlacementEvaluator(self.net, assoc, self.radio_all(assoc), self.w, self.K)

    def decision(self, assoc, cu, du):
        t = self.net.topo
        radio = self.radio_all(assoc)
        split = cu != du
        owner = np.stack([radio[b][int(split[b])].owner for b in range(t.B)])
        power = np.stack([radio[b][int(split[b])].power for b in range(t.B)])
        return Decision.from_compact(assoc, owner, power, cu, du, t.B, t.N)

    # ---- tiny instances: certified global optimum by full enumeration
    def solve_exact(self):
        t, U = self.net.topo, self.net.users.U
        relaxed, best = INF, None
        for assoc in itertools.product(range(t.B), repeat=U):
            assoc = np.array(assoc)
            ev = self.evaluator(assoc)
            v, _, _ = ev.best_exhaustive(radio_constraints=False)        # radio-coupled constraints relaxed
            relaxed = min(relaxed, v)                                     # -> valid lower bound
            v, cu, du = ev.best_exhaustive(radio_constraints=True)
            if cu is not None and (best is None or v < best[0]):
                best = (v, assoc, cu, du)
        v, assoc, cu, du = best
        res = Result(v, self.decision(assoc, cu, du), assoc, cu, du, lower_bound=relaxed)
        r = self.net.evaluate(res.decision)
        # certified when the relaxation bound is attained by a feasible solution
        res.certified = bool(r["feasible"]) and v <= relaxed * (1 + 1e-9)
        res.info["kpis"] = r
        return res

    # ---- full size: multi-start association search + exact placement
    def _assoc_value(self, assoc, cu, du):
        ev = self.evaluator(assoc)
        return ev.best_coordinate(cu, du)

    def solve(self, restarts=8, seed=0, candidates=3, exact_placement=True, verbose=False):
        net, t, U = self.net, self.net.topo, self.net.users.U
        rng = np.random.default_rng(seed)
        order = np.argsort(-net.G, axis=0)[:candidates]                  # candidate RUs per user
        from .reference_policies import placement
        starts = [order[0].copy()]
        for k in range(restarts - 1):
            depth = min(2 if k % 2 == 0 else candidates, len(order))         # alternate top-2 and top-k starts
            starts.append(order[rng.integers(0, depth, U), np.arange(U)])
        best = (INF, None, None, None)
        history = []
        for s_i, assoc in enumerate(starts):
            cu0, du0 = placement(t, "all_edge")
            v, cu, du = self._assoc_value(assoc, cu0, du0)
            for sc in ("cu_regional_du_edge", "all_regional"):
                c2, d2 = placement(t, sc)
                v2, c2, d2 = self._assoc_value(assoc, c2, d2)
                if v2 < v:
                    v, cu, du = v2, c2, d2
            improved = True
            while improved:
                improved = False
                for u in rng.permutation(U):
                    for b in order[:, u]:
                        if b == assoc[u]:
                            continue
                        a2 = assoc.copy()
                        a2[u] = b
                        v2, c2, d2 = self._assoc_value(a2, cu, du)
                        if v2 < v - 1e-12:
                            assoc, v, cu, du, improved = a2, v2, c2, d2, True
            history.append(v)
            if verbose:
                print(f"  start {s_i}: {v:.6f}")
            if v < best[0]:
                best = (v, assoc.copy(), cu.copy(), du.copy())
        v, assoc, cu, du = best
        self.polish(assoc)
        v, cu, du = self.evaluator(assoc).best_coordinate(cu, du)
        if exact_placement:
            ve, ce, de = self.evaluator(assoc).best_exhaustive()
            if ce is not None and ve <= v:
                v, cu, du = ve, ce, de
        res = Result(v, self.decision(assoc, cu, du), assoc, cu, du, info={"restart_values": history})
        res.info["kpis"] = net.evaluate(res.decision)
        return res

    # ---- lower bounds
    def radio_lower_bound(self, primal=None):
        """Fractional association + fractional PRBs + free power: certified LB on sum_u w_u T_u.

        Transport delays (>= 0) are dropped; air propagation is accounted in infra_lower_bound.
        """
        net, cfg = self.net, self.net.cfg
        u = net.users
        a = cfg.p_max_w * net.G[:, :, None] * net.H / cfg.noise_per_prb_w           # (B,U,M)
        allowed = np.ones(a.shape[:2], bool)
        return radio_dual_bound(a, allowed, u.q_bits, u.lam, self.w, cfg.prb_bandwidth_hz,
                                init=self._kkt_multipliers(a, primal))

    def _kkt_multipliers(self, a, primal):
        """Dual warm start from a primal solution: nu_u = -df_u/dR_u, mu_b = nu c a/(1 + a p) on used PRBs."""
        if primal is None:
            return None
        u, B = self.net.users, a.shape[0]
        R = primal.info["kpis"]["rate_u"]
        x = np.maximum(R / u.q_bits - u.lam, 1e-12)
        nu = self.w / u.q_bits / x ** 2
        c = self.net.cfg.prb_bandwidth_hz / LN2
        owner = primal.decision.kappa.argmax(axis=1)                                 # (B,M)
        used = primal.decision.kappa.sum(axis=1) > 0
        pf = primal.decision.omega.sum(axis=1) / self.net.cfg.p_max_w               # (B,M)
        mu = np.ones(B)
        for b in range(B):
            ms = np.flatnonzero(used[b] & (pf[b] > 0))
            if len(ms):
                ab = a[b, owner[b, ms], ms]
                mu[b] = np.median(nu[owner[b, ms]] * c * ab / (1 + ab * pf[b, ms]))
            else:
                mu[b] = np.median(nu) * c
        return np.r_[np.log(nu), [-20.0], np.log(mu)]

    def infra_lower_bound(self, chunk=100_000):
        """min over psi of sum_u min_b [w_u (fiber + service + air) + K L_u coef_b(psi)], queueing waits >= 0.

        Exact in psi; relaxes association per user and drops queueing and transport delays.
        phi_bar is linear in per-RU load for fixed psi, so the cost term splits per user.
        """
        net, cfg = self.net, self.net.cfg
        t, u = net.topo, net.users
        B, N = t.B, t.N
        c = np.asarray(cfg.cpu_per_bps)
        service = (c[:, None] * u.q_bits[None, :] / cfg.core_hz).sum(0)              # (U,)
        air = np.linalg.norm(t.ru_pos[:, None] - u.pos[None], axis=-1) / C_LIGHT     # (B,U)
        L = u.lam * u.q_bits
        unit = t.cost_factor / t.cpu_capacity * cfg.cost_unit_v                        # phi_n per cycle/s
        best = INF
        total = N ** (2 * B)
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total))
            digits = np.stack(np.unravel_index(idx, (N,) * (2 * B)), axis=1)
            cu, du = digits[:, :B], digits[:, B:]
            C = len(idx)
            ok = np.ones(C, bool)
            if cfg.fronthaul_budget_s is not None:
                ok &= np.all(t.prop_front[np.arange(B), du] <= cfg.fronthaul_budget_s, axis=1)
            k = np.zeros((C, N))
            for b in range(B):
                np.add.at(k, (np.arange(C), cu[:, b]), 1)
                np.add.at(k, (np.arange(C), du[:, b]), 1)
            ku = k * unit                                                               # (C,N)
            coef = (c[CU] * np.take_along_axis(ku, cu, 1) + c[DU] * np.take_along_axis(ku, du, 1)) / B  # (C,B)
            fiber = t.prop_mid[cu, du] + t.prop_front[np.arange(B), du]                 # (C,B)
            per = (self.w[None, None, :] * (fiber[:, :, None] + service[None, None, :] + air[None])
                   + self.K * coef[:, :, None] * L[None, None, :])                       # (C,B,U)
            val = per.min(axis=1).sum(1)
            val = np.where(ok, val, INF)
            best = min(best, float(val.min()))
        return best

    def lower_bound(self, primal=None):
        """Global certified lower bound (association, PRBs, power and placement all relaxed)."""
        return self.radio_lower_bound(primal) + self.infra_lower_bound()

    def conditional_lower_bound(self, assoc):
        """Certified lower bound over every PRB assignment, power and placement for a fixed association.

        Per RU and split flag the radio block is bounded by its dual relaxation; placement is then
        enumerated exactly with these bounds, with the radio-coupled constraints relaxed.
        """
        net = self.net
        ev = self.evaluator(assoc)
        V_lb = np.zeros((net.topo.B, 2))
        for b in range(net.topo.B):
            users = tuple(np.flatnonzero(np.asarray(assoc) == b))
            sols = self.radio(b, users)
            ru = RadioRU(net, b, users, self.w)
            for sp in (0, 1):
                V_lb[b, sp] = min(ru.relaxation_lb(sp, init_from=sols[sp]), sols[sp].value)
        ev.V = V_lb
        v, _, _ = ev.best_exhaustive(radio_constraints=False)
        return v
