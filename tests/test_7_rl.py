"""Phase 6-8: two-timescale environment, TD3/SAC agents, the three methods."""
import json

import numpy as np
import pytest
import torch

from oran_sim import Config, Network
from oran_sim.agents import SAC, TD3, AgentConfig, Replay, soft_update
from oran_sim.marl_env import EnvConfig, TwoTimescaleEnv
from oran_sim.rl_methods import METHODS, Method, convex_small_decision, train

SMALL_AGENT = AgentConfig(actor_hidden=(32,), critic_hidden=(32,), warmup=10, batch_size=16, buffer_size=500)
SHORT = EnvConfig(large_steps_per_episode=2)


@pytest.fixture(scope="module")
def env():
    return TwoTimescaleEnv(Config(n_ue=10), SHORT)


# ---------------------------------------------------------------- environment
def test_dimensions(env):
    dl, ds = env.obs_dims
    assert env.large_dim == 10 * 4 + 2 * 4 * 6 and env.small_dim == 20
    assert ds == dl + env.U * env.M
    o = env.reset(0)
    assert o["large"].dtype == np.float32 and np.all(np.isfinite(o["small"]))


def test_episode0_is_phase5_snapshot(env):
    env.reset(0)
    ref = Network(Config(n_ue=10))
    np.testing.assert_array_equal(env.net.G, ref.G)
    np.testing.assert_array_equal(env.net.H, ref.H)
    env.reset(3)
    assert not np.allclose(env.net.users.pos, ref.users.pos)                  # training drops differ
    np.testing.assert_array_equal(env.net.topo.node_pos, ref.topo.node_pos)  # topology fixed


def test_random_actions_are_structurally_valid(env):
    rng = np.random.default_rng(0)
    env.reset(1)
    for _ in range(40):
        o, r, d, info = env.step(rng.uniform(-1, 1, env.small_dim), rng.uniform(-1, 1, env.large_dim))
        c = env.last["constraints"]
        for k in ("eq1_single_association", "eq2_prb_needs_association", "eq3_prb_exclusive",
                  "eqa5_power", "eq6_single_placement"):
            assert c[k], k
        dec = info["decision"]
        for b in range(env.B):
            users = np.flatnonzero(env.assoc == b)
            if 0 < len(users) <= env.M:
                assert np.all(dec.kappa[b][users].sum(axis=1) >= 1)          # every user gets a PRB
                assert dec.omega[b].sum() == pytest.approx(env.cfg.p_max_w)
        if d:
            env.reset(2)


def test_large_action_only_every_K_slots(env):
    rng = np.random.default_rng(1)
    env.reset(1)
    a0 = rng.uniform(-1, 1, env.large_dim)
    env.step(rng.uniform(-1, 1, env.small_dim), a0)
    held = env.assoc.copy(), env.cu.copy()
    for _ in range(env.K - 1):
        env.step(rng.uniform(-1, 1, env.small_dim), rng.uniform(-1, 1, env.large_dim))
        np.testing.assert_array_equal(env.assoc, held[0])
        np.testing.assert_array_equal(env.cu, held[1])
    assert env.is_large_step


def test_reward_definition(env):
    env.reset(0)
    rng = np.random.default_rng(2)
    _, r, _, info = env.step(rng.uniform(-1, 1, env.small_dim), rng.uniform(-1, 1, env.large_dim))
    J = info["objective"] if np.isfinite(info["objective"]) else env.ecfg.reward_clip * info["J_ref"]
    expect = -min((J + info["n_viol"] * info["J_ref"]) / info["J_ref"], env.ecfg.reward_clip)
    assert r == pytest.approx(expect)


def test_episode_length_and_determinism():
    acts = np.random.default_rng(3)
    A = [(acts.uniform(-1, 1, 20), acts.uniform(-1, 1, 88)) for _ in range(20)]
    out = []
    for _ in range(2):
        e = TwoTimescaleEnv(Config(n_ue=10), SHORT)
        e.reset(5)
        rs, n = [], 0
        while not e.done:
            _, r, _, _ = e.step(*A[n])
            rs.append(r)
            n += 1
        out.append(rs)
        assert n == e.K * SHORT.large_steps_per_episode
    assert out[0] == out[1]


def test_convex_small_beats_random_small(env):
    rng = np.random.default_rng(4)
    better = 0
    for ep in range(1, 6):
        env.reset(ep)
        env.assoc, env.cu, env.du = env.decode_large(rng.uniform(-1, 1, env.large_dim))
        net = env.net
        from oran_sim.kpis import Decision
        o_c, p_c = convex_small_decision(env)
        o_r, p_r = env.decode_small(rng.uniform(-1, 1, env.small_dim), env.assoc)
        jc = net.evaluate(Decision.from_compact(env.assoc, o_c, p_c, env.cu, env.du, env.B, env.N))["objective"]
        jr = net.evaluate(Decision.from_compact(env.assoc, o_r, p_r, env.cu, env.du, env.B, env.N))["objective"]
        better += jc <= jr
    assert better == 5


def test_zero_action_with_prior_is_best_static():
    from oran_sim.reference_policies import penalised_objective, reference_decision
    e = TwoTimescaleEnv(Config(n_ue=10), EnvConfig(large_steps_per_episode=1, large_prior=1.0))
    for ep in (0, 4):
        e.reset(ep)
        refs = {s: reference_decision(e.net, s) for s in ("all_edge", "cu_regional_du_edge", "all_regional")}
        best = min(refs.values(), key=lambda d: penalised_objective(e.net, d))
        assoc, cu, du = e.decode_large(np.zeros(e.large_dim))
        np.testing.assert_array_equal(assoc, best.assoc)
        np.testing.assert_array_equal(cu, best.cu_node)
        np.testing.assert_array_equal(du, best.du_node)


# ---------------------------------------------------------------- agents
def test_replay_fifo():
    b = Replay(2, 1, 3)
    for i in range(5):
        b.add([i, i], [i], i, [i, i], 0)
    assert b.n == 3 and sorted(b.r.tolist()) == [2, 3, 4]


def test_soft_update():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = b.weight.detach().clone()
    soft_update(b, a, 0.1)
    np.testing.assert_allclose(b.weight.detach(), 0.9 * before + 0.1 * a.weight.detach(), rtol=1e-6)


@pytest.mark.parametrize("cls", [TD3, SAC])
def test_agents_learn_a_bandit(cls):
    """Constant observation, reward -(a - 0.5)^2 per dimension: the policy must move to 0.5."""
    cfg = AgentConfig(actor_hidden=(32,), critic_hidden=(32,), warmup=64, batch_size=64, buffer_size=4000,
                      actor_lr=3e-3, critic_lr=3e-3, gamma=0.0, tau=0.05)
    ag = cls(1, 2, cfg, seed=0)
    o = np.zeros(1, np.float32)
    for _ in range(1500):
        a = ag.act(o)
        ag.buf.add(o, a, -float(((a - 0.5) ** 2).sum()), o, 1.0)
        ag.update()
    np.testing.assert_allclose(ag.act(o, explore=False), [0.5, 0.5], atol=0.15)


def test_td3_target_smoothing_is_clipped():
    cfg = AgentConfig()
    noise = (torch.randn(100000, 3) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
    assert noise.abs().max() <= cfg.noise_clip and noise.std() == pytest.approx(0.2, rel=0.05)


# ---------------------------------------------------------------- methods end to end
@pytest.mark.parametrize("method", METHODS)
def test_methods_run_and_log(method, tmp_path):
    evals = train(method, n_ue=5, seed=0, episodes=3, out_dir=tmp_path, eval_every=2,
                  agent_cfg=SMALL_AGENT, env_cfg=SHORT)
    assert len(evals) == 2
    data = json.loads((tmp_path / "eval.json").read_text())
    assert data["method"] == method and np.isfinite(data["evals"][-1]["first_objective"])
    assert (tmp_path / "log.csv").exists() and (tmp_path / "final.pt").exists()


@pytest.mark.parametrize("method", ["marl_td3", "single_sac"])
def test_training_resumes_from_checkpoint(method, tmp_path, monkeypatch):
    """A run killed after episode 3 continues at episode 4 with the saved networks, not from scratch."""
    from oran_sim import rl_methods
    from oran_sim.agents import checkpoint_state
    kw = dict(n_ue=5, seed=0, episodes=6, out_dir=tmp_path, eval_every=2, agent_cfg=SMALL_AGENT,
              env_cfg=SHORT, ckpt_every=3)
    orig = rl_methods.Method.run_episode
    saved = {}

    def crash_after_3(self, episode, explore=True, learn=True):
        if episode == 4 and learn:
            saved.update({k: checkpoint_state(a) for k, a in self.agents().items()})
            raise KeyboardInterrupt
        return orig(self, episode, explore, learn)

    monkeypatch.setattr(rl_methods.Method, "run_episode", crash_after_3)
    with pytest.raises(KeyboardInterrupt):
        train(method, **kw)
    assert (tmp_path / "ckpt.pt").exists()
    st = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert st["episode"] == 3 and len(st["rows"]) == 3
    for k, a in saved.items():                                   # checkpoint = state at the kill
        for name, w in a["actor"].items():
            torch.testing.assert_close(st["agents"][k]["actor"][name], w)

    seen = []

    def record(self, episode, explore=True, learn=True):
        if learn:
            seen.append(episode)
            assert all(a.resumed for a in self.agents().values())
        return orig(self, episode, explore, learn)

    monkeypatch.setattr(rl_methods.Method, "run_episode", record)
    evals = train(method, **kw)
    assert seen == [4, 5, 6]
    assert [e["episode"] for e in evals] == [2, 4, 6]
    rows = (tmp_path / "log.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 6 and not (tmp_path / "ckpt.pt").exists()
