"""Phase 5: optimal solver, certified bounds and baselines."""
import itertools

import numpy as np
import pytest

from oran_sim import Config, Decision, Network
from oran_sim.baselines import BASELINES, STATIC_SCENARIOS
from oran_sim.optimal import RadioRU, Solver, objective_weights
from oran_sim.reference_policies import penalised_objective, reference_decision

TINY = dict(n_ru=2, n_edge=1, n_regional=1, n_ue=3, n_prb=3)


@pytest.fixture(scope="module")
def net10():
    return Network(Config(n_ue=10))


@pytest.fixture(scope="module")
def solved10(net10):
    s = Solver(net10)
    return s, s.solve(restarts=2)


# ---------------------------------------------------------------- decomposition is exact
def test_objective_weights_reproduce_kpis(net10):
    w, K = objective_weights(net10)
    for sc in STATIC_SCENARIOS:
        r = net10.evaluate(reference_decision(net10, sc))
        assert w @ r["d_e2e"] + K * r["phi_bar"] == pytest.approx(r["objective"], rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_placement_evaluator_matches_kpis(seed):
    net = Network(Config(seed=seed, n_ue=10))
    s = Solver(net)
    assoc = net.G.argmax(0)
    ev = s.evaluator(assoc)
    rng = np.random.default_rng(seed)
    for _ in range(25):
        cu, du = rng.integers(net.topo.N, size=4), rng.integers(net.topo.N, size=4)
        v, f = ev.evaluate(cu[None], du[None])
        r = net.evaluate(s.decision(assoc, cu, du))
        assert v[0] == pytest.approx(r["objective"], rel=1e-10)
        assert bool(f[0]) == r["feasible"]


# ---------------------------------------------------------------- power block
def _rus(net, w):
    assoc = net.G.argmax(0)
    return [RadioRU(net, b, np.flatnonzero(assoc == b), w) for b in range(net.topo.B)
            if np.any(assoc == b)]


def test_fast_power_never_worse_than_cvxpy(net10):
    w, _ = objective_weights(net10)
    for ru in _rus(net10, w):
        for sp in (0, 1):
            owner = ru.assign_local_search(sp)
            fast, ref = ru.optimal_power(owner, sp), ru.optimal_power_cvx(owner, sp)
            if ref is not None:
                assert fast.value <= ref.value * (1 + 1e-6)


def test_power_kkt_conditions(net10):
    """Water-filling KKT: kappa_o c a/(1 + a p) = mu on PRBs with p > 0 and <= mu where p = 0."""
    w, _ = objective_weights(net10)
    for ru in _rus(net10, w):
        owner = ru.assign_local_search(0)
        sol = ru.optimal_power(owner, 0)
        ms = np.flatnonzero(owner >= 0)
        o, pf = owner[ms], sol.power[ms] / ru.P
        a = ru.P * ru.snr1[o, ms]
        x = sol.rate / ru.q - ru.lam
        kap = ru.w / ru.q / x ** 2 + ru.W * ru.wf / ru.qbar / (ru.wf * sol.rate.sum() / ru.qbar - ru.lam_b) ** 2
        marg = kap[o] * a / (1 + a * pf)
        on = pf > 1e-9
        mu = np.median(marg[on])
        np.testing.assert_allclose(marg[on], mu, rtol=1e-4)
        assert np.all(marg[~on] <= mu * (1 + 1e-4))
        assert pf.sum() == pytest.approx(1.0)


def test_power_beats_random_feasible_powers(net10):
    w, _ = objective_weights(net10)
    rng = np.random.default_rng(0)
    for ru in _rus(net10, w):
        owner = ru.assign_local_search(0)
        best = ru.optimal_power(owner, 0).value
        for _ in range(200):
            p = rng.dirichlet(np.ones(ru.M) * 0.5) * ru.P
            assert ru._solution(owner, p, 0).value >= best * (1 - 1e-9)


# ---------------------------------------------------------------- certified bounds
def test_radio_dual_bound_is_valid_and_tight(net10):
    w, _ = objective_weights(net10)
    for ru in _rus(net10, w):
        for sp in (0, 1):
            sol = ru.refine(ru.solve(sp), sp)
            lb = ru.relaxation_lb(sp, sol)
            assert lb <= sol.value
            assert (sol.value - lb) / sol.value < 0.01          # certified: within 1% of the radio optimum
            if len(ru.users) <= 2:
                assert (sol.value - lb) / sol.value < 1e-4


def test_lower_bounds_bracket_the_solution(net10, solved10):
    s, res = solved10
    glob = s.lower_bound(res)
    cond = s.conditional_lower_bound(res.assoc)
    assert glob <= cond <= res.value
    assert (res.value - cond) / res.value < 0.01


# ---------------------------------------------------------------- placement block
def test_exhaustive_placement_matches_kpis_bruteforce():
    net = Network(Config(**{**TINY, "n_ue": 4, "n_prb": 4}))
    s = Solver(net)
    assoc = net.G.argmax(0)
    v, cu, du = s.evaluator(assoc).best_exhaustive()
    brute = INF = np.inf
    for c in itertools.product(range(net.topo.N), repeat=2):
        for d in itertools.product(range(net.topo.N), repeat=2):
            r = net.evaluate(s.decision(assoc, np.array(c), np.array(d)))
            if r["feasible"]:
                brute = min(brute, r["objective"])
    assert brute < INF and v == pytest.approx(brute, rel=1e-10)


# ---------------------------------------------------------------- global optimum on tiny instances
def _bruteforce_tiny(net):
    """Independent enumeration through kpis.evaluate: every association, PRB owner map,
    placement; exact power per RU for each (owner map, split)."""
    t, U, M = net.topo, net.users.U, net.cfg.n_prb
    w, _ = objective_weights(net)
    best = np.inf
    for assoc in itertools.product(range(t.B), repeat=U):
        assoc = np.array(assoc)
        rus = [RadioRU(net, b, np.flatnonzero(assoc == b), w) for b in range(t.B)]
        options = []
        for ru in rus:
            n = len(ru.users)
            if n == 0:
                options.append([(None, {0: np.zeros(M), 1: np.zeros(M)})])
                continue
            opts = []
            for combo in itertools.product(range(n), repeat=M):
                if len(set(combo)) < n:
                    continue
                own = np.array(combo)
                pw = {}
                for sp in (0, 1):
                    sol = ru.optimal_power(own, sp)
                    pw[sp] = None if sol is None else sol.power
                opts.append((own, pw))
            options.append(opts)
        for c in itertools.product(range(t.N), repeat=t.B):
            for d in itertools.product(range(t.N), repeat=t.B):
                split = np.array(c) != np.array(d)
                for choice in itertools.product(*options):
                    owner = -np.ones((t.B, M), int)
                    power = np.zeros((t.B, M))
                    ok = True
                    for b, (own, pw) in enumerate(choice):
                        if own is None:
                            continue
                        p = pw[int(split[b])]
                        if p is None:
                            ok = False
                            break
                        owner[b] = rus[b].users[own]
                        power[b] = p
                    if not ok:
                        continue
                    r = net.evaluate(Decision.from_compact(assoc, owner, power, np.array(c), np.array(d), t.B, t.N))
                    if r["feasible"]:
                        best = min(best, r["objective"])
    return best


@pytest.mark.parametrize("seed", range(3))
def test_exact_solver_equals_bruteforce(seed):
    net = Network(Config(seed=seed, **{**TINY, "n_prb": 2, "n_ue": 2}))
    res = Solver(net).solve_exact()
    assert res.lower_bound <= res.value * (1 + 1e-12)
    assert res.value == pytest.approx(_bruteforce_tiny(net), rel=1e-7)
    if res.certified:
        assert res.value == pytest.approx(res.lower_bound, rel=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_scalable_solver_finds_tiny_optimum(seed):
    net = Network(Config(seed=seed, **TINY))
    exact = Solver(net).solve_exact()
    found = Solver(net, exhaustive_limit=0, refine_in_search=True).solve(restarts=10)  # no PRB enumeration
    assert found.value == pytest.approx(exact.value, rel=1e-6)


# ---------------------------------------------------------------- baselines
def test_baselines_valid_and_ordered(net10, solved10):
    _, res = solved10
    rng = np.random.default_rng(0)
    vals = {}
    for name, fn in BASELINES.items():
        if name == "joint_optimum":
            vals[name] = res.value
            continue
        d = fn(net10, rng)
        r = net10.evaluate(d)
        assert r["constraints"]["eq1_single_association"] and r["constraints"]["eq6_single_placement"]
        vals[name] = penalised_objective(net10, d)
    assert vals["static_best"] == min(vals[s] for s in STATIC_SCENARIOS)
    assert vals["placement_search"] <= vals["static_best"]
    assert vals["joint_optimum"] <= vals["placement_search"]
    assert vals["random"] > vals["static_best"]
