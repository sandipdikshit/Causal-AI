"""Paired, held-out stress evaluation; never choose a seed to obtain a desired result."""

import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .ppo import HEDGES, probabilities, train
from .scm import Config, features, make_world, procurement_outcome


def run(output: Path, seeds: list[int], steps: int, test_paths: int = 8) -> dict:
    if not seeds or test_paths < 1:
        raise ValueError("At least one training seed and test path are required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = Config()
    output.mkdir(parents=True, exist_ok=True)
    rows, training = [], []
    for seed in seeds:
        policies = {}
        for label, causal in [("Observational PPO", False), ("Interventional PPO", True)]:
            print(f"Training {label}: seed={seed}, steps={steps}", flush=True)
            policy, log = train(causal, seed, steps)
            policies[label] = policy
            training.extend({"seed": seed, "policy": label, **entry} for entry in log)
        for path in range(test_paths):
            # New noise realizations, paired across all policies. Fixed stress parameters.
            world = make_world(10000 + path, config)
            times = np.arange(config.entry_year * 52, config.years * 52 - config.horizon)
            states = np.stack([features(world, int(t), config.history) for t in times])
            for label in [*policies, "Always hedge", "Never hedge", "50% hedge"]:
                if label in policies:
                    probs = probabilities(policies[label], states)
                    hedges = probs @ HEDGES  # deterministic deployable fractional hedge
                    under = probs[:, HEDGES < 0.5].sum(1)
                else:
                    value = {"Always hedge": 1.0, "Never hedge": 0.0, "50% hedge": 0.5}[label]
                    hedges = np.full(len(times), value)
                    under = np.full(len(times), float(value < 0.5))
                for t, hedge, probability in zip(times, hedges, under):
                    cost, utility = procurement_outcome(world, int(t), float(hedge), config)
                    rows.append(
                        {
                            "seed": seed,
                            "path": path,
                            "policy": label,
                            "t": int(t),
                            "year": int(t // 52 + 1),
                            "week": int(t % 52 + 1),
                            "price": world.price[t],
                            "hedge": hedge,
                            "p_under_half": probability,
                            "cost": cost,
                            "utility": utility,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "evaluation.csv", index=False)
    pd.DataFrame(training).to_csv(output / "training.csv", index=False)
    per_run = (
        frame.groupby(["seed", "path", "policy"])
        .agg(
            total_cost=("cost", "sum"),
            total_utility=("utility", "sum"),
            worst_tranche_cost=("cost", "max"),
            mean_hedge=("hedge", "mean"),
        )
        .reset_index()
    )
    per_run.to_csv(output / "per_run.csv", index=False)
    # Aggregate over test paths first, so training seeds remain the replication unit.
    per_seed = per_run.groupby(["seed", "policy"]).mean(numeric_only=True).reset_index()
    summary = (
        per_seed.groupby("policy")
        .agg(
            mean_cost=("total_cost", "mean"),
            sd_across_training_seeds=("total_cost", "std"),
            mean_utility=("total_utility", "mean"),
            mean_hedge=("mean_hedge", "mean"),
            mean_worst_tranche=("worst_tranche_cost", "mean"),
        )
        .fillna(0)
    )
    summary.to_csv(output / "summary.csv")
    paired = per_seed.pivot(index="seed", columns="policy", values="total_cost")
    improvement = paired["Observational PPO"] - paired["Interventional PPO"]
    result = {
        "config": asdict(config),
        "training_seeds": seeds,
        "steps_per_policy": steps,
        "test_paths_per_seed": test_paths,
        "evaluation_decisions_per_path": len(times),
        "paired_cost_reduction_mean": float(improvement.mean()),
        "paired_cost_reduction_by_seed": {str(k): float(v) for k, v in improvement.items()},
        "positive_reduction_means": "Interventional policy had lower procurement cost",
        "underhedging_definition": "Probability of categorical hedge below 0.5; not mean hedge",
        "evaluation_action": "Probability-weighted mean hedge; training samples categorical actions",
        "market_data_used_for_training": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "interpretation": "Known synthetic SCM stress experiment, not empirical causal identification",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
