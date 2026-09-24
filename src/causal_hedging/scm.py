"""Known synthetic SCM. Effects are assumptions, not identified from market prices."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    years: int = 10
    history: int = 104
    entry_year: int = 8  # zero based: beginning of calendar year 9
    horizon: int = 4
    liquidation_effect: float = 8.0
    shock_size: float = 12.0
    hedge_fee: float = 0.15
    risk_aversion: float = 0.04


DEFAULT_CONFIG = Config()


@dataclass
class World:
    price: np.ndarray
    seasonal: np.ndarray
    noise: np.ndarray
    liquidation: np.ndarray
    shock: np.ndarray


def make_world(seed: int, config: Config = DEFAULT_CONFIG, intervention: str = "observed") -> World:
    """Same seed preserves exogenous disturbances across liquidation interventions.

    `observed`: company liquidates in Q4 until competitor entry, then Q3.
    `none`, `q3`, `q4`: replace the entire company liquidation equation.
    Competitor entry changes the company's schedule; no direct competitor price effect.
    """
    if intervention not in {"observed", "none", "q3", "q4"}:
        raise ValueError(f"Unknown intervention: {intervention}")
    rng = np.random.default_rng(seed)
    t = np.arange(config.years * 52)
    week, year = t % 52 + 1, t // 52
    seasonal = 1.5 * np.sin(2 * np.pi * (week - 1) / 52)
    noise = rng.normal(0, 0.7, t.size)
    if intervention == "observed":
        liquidation = np.where(year < config.entry_year, week >= 40, (week >= 27) & (week <= 39))
    elif intervention == "none":
        liquidation = np.zeros(t.size, dtype=bool)
    elif intervention == "q3":
        liquidation = (week >= 27) & (week <= 39)
    else:
        liquidation = week >= 40
    # Held-out stress: January of calendar year 10. Never used in training.
    elapsed = t - (config.entry_year + 1) * 52
    shock = np.where(elapsed >= 0, config.shock_size * np.exp(-np.maximum(elapsed, 0) / 8), 0)
    price = 100 + seasonal + noise - config.liquidation_effect * liquidation + shock
    return World(price, seasonal, noise, liquidation.astype(float), shock)


def features(world: World, t: int, history: int = 104) -> np.ndarray:
    """Information available at decision time t; no future prices or hidden effects."""
    if t < history - 1 or t >= len(world.price):
        raise ValueError("Insufficient history or decision outside world")
    idx = np.arange(t - history + 1, t + 1)
    angle = 2 * np.pi * (idx % 52) / 52
    return np.stack(((world.price[idx] - 100) / 10, np.sin(angle), np.cos(angle)), -1).astype(
        np.float32
    )


def procurement_outcome(world: World, t: int, hedge: float, config: Config) -> tuple[float, float]:
    """Independent four-week procurement tranche; a hedge locks today's spot proxy.

    Positive cost = worse than locking all volume at today's spot, before fees.
    Utility penalizes positive cost quadratically. No inventory/futures basis model.
    """
    delta = world.price[t + config.horizon] - world.price[t]
    cost = (1 - hedge) * delta + config.hedge_fee * hedge
    utility = -cost - config.risk_aversion * max(cost, 0) ** 2
    return float(cost), float(utility)


class TrainingEnv:
    """Replay eight historical years, or sample explicit liquidation interventions."""

    def __init__(self, causal: bool, seed: int, config: Config = DEFAULT_CONFIG):
        self.causal, self.seed, self.config = causal, seed, config
        self.rng = np.random.default_rng(seed)
        self.t = config.history - 1
        self.world = make_world(seed, config)

    def reset(self) -> np.ndarray:
        schedule = self.rng.choice(["observed", "none", "q3", "q4"]) if self.causal else "observed"
        self.world = make_world(self.seed, self.config, str(schedule))
        self.t = self.config.history - 1
        return features(self.world, self.t, self.config.history)

    def step(self, hedge: float) -> tuple[np.ndarray, float, bool]:
        _, utility = procurement_outcome(self.world, self.t, hedge, self.config)
        self.t += 1
        # Last training target is strictly before entry: no future regime leakage.
        done = self.t + self.config.horizon >= self.config.entry_year * 52
        return features(self.world, self.t, self.config.history), utility / 10, done
