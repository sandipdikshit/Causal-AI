import numpy as np
import pytest
import torch

from causal_hedging.data import parse_csv
from causal_hedging.ppo import Policy, clipped_loss, gae, train
from causal_hedging.scm import Config, TrainingEnv, features, make_world, procurement_outcome


def test_intervention_preserves_exogenous_world():
    absent, liquidating = make_world(42, intervention="none"), make_world(42, intervention="q4")
    np.testing.assert_array_equal(absent.noise, liquidating.noise)
    np.testing.assert_array_equal(absent.shock, liquidating.shock)
    expected = -8 * (np.arange(520) % 52 + 1 >= 40)
    np.testing.assert_allclose(liquidating.price - absent.price, expected)


def test_entry_is_calendar_year_nine_and_shock_is_held_out():
    world = make_world(42)
    assert world.liquidation[7 * 52 + 39] == 1
    assert world.liquidation[8 * 52 + 39] == 0
    assert world.liquidation[8 * 52 + 26] == 1
    assert np.all(world.shock[: 9 * 52] == 0)
    assert world.shock[9 * 52] == 12


def test_features_cannot_see_future():
    world = make_world(42)
    before = features(world, 200).copy()
    world.price[201:] += 1000
    np.testing.assert_array_equal(before, features(world, 200))
    assert before.shape == (104, 3)


def test_training_targets_stay_before_entry():
    env = TrainingEnv(True, 42)
    env.reset()
    done = False
    while not done:
        assert env.t + env.config.horizon < env.config.entry_year * 52
        _, _, done = env.step(0.5)
    assert env.t == 412


def test_ppo_rejects_original_broadcast_bug():
    with pytest.raises(ValueError, match="one-dimensional"):
        clipped_loss(torch.zeros(64), torch.zeros(64, 1), torch.ones(64))
    assert clipped_loss(torch.zeros(4), torch.zeros(4), torch.ones(4)).item() == -1


def test_gae_terminal_blocks_next_episode():
    advantage, returns = gae([1, 2], [0, 0], [True, True], 999)
    np.testing.assert_allclose(advantage, [1, 2])
    np.testing.assert_allclose(returns, [1, 2])


def test_policy_likelihood_is_stable_and_bounded():
    torch.set_num_threads(1)
    policy = Policy()
    x = torch.zeros(3, 104, 3)
    p1, value = policy(x)
    p2, _ = policy(x)
    torch.testing.assert_close(p1.probs, p2.probs)
    assert p1.log_prob(p1.sample()).shape == (3,)
    assert value.shape == (3,)
    torch.testing.assert_close(p1.probs.sum(-1), torch.ones(3))


def test_full_hedge_removes_spot_change_exposure():
    config, world = Config(), make_world(42)
    for t in [200, 411, 467]:
        cost, _ = procurement_outcome(world, t, 1.0, config)
        assert cost == config.hedge_fee


def test_csv_missing_values_and_negative_prices():
    data = parse_csv(
        b"observation_date,DCOILWTICO\n2020-04-20,-36.98\n2020-04-21,.\n2020-04-22,13.64\n"
    )
    assert len(data) == 2
    assert data.price.iloc[0] == -36.98
    with pytest.raises(ValueError):
        parse_csv(b"x,y\n1,2\n")


def test_training_smoke_exact_step_count():
    torch.set_num_threads(1)
    policy, log = train(False, 42, steps=17, batch_steps=8)
    assert log[-1]["steps"] == 17
    assert all(torch.isfinite(p).all() for p in policy.parameters())
