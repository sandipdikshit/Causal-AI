"""Validate the experiment directly from its single source: the executed notebook."""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "Causal_AI_Experiment.ipynb").read_text())


def load_definitions():
    # Load imports and definition cells only: no full training or network fetch on import.
    module = ModuleType("notebook_under_test")
    sys.modules[module.__name__] = module
    setup = next(c for c in NOTEBOOK["cells"] if c["cell_type"] == "code")
    source = "".join(setup["source"])
    parsed = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
    imports = [
        node
        for node in parsed.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and getattr(node, "module", "") != "IPython.display"
    ]
    exec(
        compile(ast.Module(body=imports, type_ignores=[]), "notebook-imports", "exec"),
        module.__dict__,
    )
    for cell in NOTEBOOK["cells"]:
        if cell["metadata"].get("experiment_definitions"):
            exec(compile("".join(cell["source"]), "notebook-definitions", "exec"), module.__dict__)
    return module


core = load_definitions()


def test_intervention_preserves_exogenous_world():
    absent, liquidating = (
        core.make_world(42, intervention="none"),
        core.make_world(42, intervention="q4"),
    )
    np.testing.assert_array_equal(absent.noise, liquidating.noise)
    np.testing.assert_array_equal(absent.shock, liquidating.shock)
    expected = -8 * (np.arange(520) % 52 + 1 >= 40)
    np.testing.assert_allclose(liquidating.price - absent.price, expected)


def test_entry_is_calendar_year_nine_and_shock_is_held_out():
    world = core.make_world(42)
    assert world.liquidation[7 * 52 + 39] == 1
    assert world.liquidation[8 * 52 + 39] == 0
    assert world.liquidation[8 * 52 + 26] == 1
    assert np.all(world.shock[: 9 * 52] == 0)
    assert world.shock[9 * 52] == 12


def test_features_cannot_see_future():
    world = core.make_world(42)
    before = core.features(world, 200).copy()
    world.price[201:] += 1000
    np.testing.assert_array_equal(before, core.features(world, 200))
    assert before.shape == (104, 3)


def test_training_targets_stay_before_entry():
    env = core.TrainingEnv(True, 42)
    env.reset()
    done = False
    while not done:
        assert env.t + env.config.horizon < env.config.entry_year * 52
        _, _, done = env.step(0.5)
    assert env.t == 412


def test_ppo_rejects_original_broadcast_bug():
    with pytest.raises(ValueError, match="one-dimensional"):
        core.clipped_loss(torch.zeros(64), torch.zeros(64, 1), torch.ones(64))
    assert core.clipped_loss(torch.zeros(4), torch.zeros(4), torch.ones(4)).item() == -1


def test_gae_terminal_blocks_next_episode():
    advantage, returns = core.gae([1, 2], [0, 0], [True, True], 999)
    np.testing.assert_allclose(advantage, [1, 2])
    np.testing.assert_allclose(returns, [1, 2])


def test_policy_likelihood_is_stable_and_bounded():
    torch.set_num_threads(1)
    policy = core.Policy()
    x = torch.zeros(3, 104, 3)
    p1, value = policy(x)
    p2, _ = policy(x)
    torch.testing.assert_close(p1.probs, p2.probs)
    assert p1.log_prob(p1.sample()).shape == (3,)
    assert value.shape == (3,)
    torch.testing.assert_close(p1.probs.sum(-1), torch.ones(3))


def test_full_hedge_removes_spot_change_exposure():
    config, world = core.Config(), core.make_world(42)
    for t in [200, 411, 467]:
        cost, _ = core.procurement_outcome(world, t, 1.0, config)
        assert cost == config.hedge_fee


def test_csv_missing_values_and_negative_prices():
    data = core.parse_csv(
        b"observation_date,DCOILWTICO\n2020-04-20,-36.98\n2020-04-21,.\n2020-04-22,13.64\n"
    )
    assert len(data) == 2
    assert data.price.iloc[0] == -36.98
    with pytest.raises(ValueError):
        core.parse_csv(b"x,y\n1,2\n")


def test_training_smoke_exact_step_count():
    torch.set_num_threads(1)
    policy, log = core.train(False, 42, steps=17, batch_steps=8)
    assert log[-1]["steps"] == 17
    assert all(torch.isfinite(p).all() for p in policy.parameters())


def test_notebook_outputs_are_complete():
    code = [c for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(range(1, len(code) + 1))
    assert not any(o["output_type"] == "error" for c in code for o in c["outputs"])
    assert sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"]) == 5
    assert sum(c["metadata"].get("experiment_definitions", False) for c in code) == 6


def test_market_snapshot_matches_provenance():
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    assert (
        hashlib.sha256((ROOT / "data/wti_daily.csv").read_bytes()).hexdigest()
        == provenance["sha256"]
    )
