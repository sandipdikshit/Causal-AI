"""Publication-friendly figures and an offline HTML demo generated from actual outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scm import Config, make_world

COLORS = {
    "Observational PPO": "#ee8065",
    "Interventional PPO": "#33c6b5",
    "Always hedge": "#a8b6ce",
    "Never hedge": "#ceaaef",
    "50% hedge": "#e8c77a",
}
BG, FG, MUTED = "#0e1726", "#f2f5fa", "#a8b6ce"


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": FG,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#344057",
            "grid.color": "#344057",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def finish(fig, path, note):
    fig.text(0.06, 0.035, note, color=MUTED, fontsize=10)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def render(root: Path):
    style()
    assets = root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    config = Config()
    observed = make_world(10000, config)
    absent = make_world(10000, config, "none")
    q4 = make_world(10000, config, "q4")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(top=0.76, bottom=0.19, wspace=0.24, left=0.06, right=0.97)
    fig.text(0.06, 0.91, "The pattern was partly our own intervention.", fontsize=25, weight="bold")
    fig.text(
        0.06,
        0.85,
        "01 / CAUSAL MECHANISM     •     Synthetic prices, known structural equations",
        color=MUTED,
    )
    for schedule, world, color in [
        ("do(liquidation = 0)", absent, "#33c6b5"),
        ("do(Q4 liquidation = 1)", q4, "#ee8065"),
    ]:
        axes[0].plot(
            np.arange(1, 53), world.price[7 * 52 : 8 * 52], label=schedule, color=color, lw=2
        )
    axes[0].axvspan(40, 52, color="#ee8065", alpha=0.1)
    axes[0].set(
        title="Same exogenous shocks. Different liquidation decision.",
        xlabel="Week of year",
        ylabel="Synthetic price index",
    )
    axes[0].legend(loc="lower left", frameon=False, labelcolor=FG)
    t = np.arange(7 * 52, 10 * 52)
    axes[1].plot(t / 52 + 1, observed.price[t], color="#a8b6ce", lw=1.8)
    axes[1].axvline(9, color="#33c6b5", ls="--")
    axes[1].axvline(10, color="#ee8065", ls="--")
    axes[1].text(9.04, 112, "Entry → Q3\nliquidation", color="#33c6b5")
    axes[1].text(10.04, 112, "January\nsupply shock", color="#ee8065")
    axes[1].set(
        title="A policy change removes the Q4 liquidation discount",
        xlabel="Calendar year (1-based)",
        ylabel="Synthetic price index",
        ylim=(87, 117),
    )
    for ax in axes:
        ax.grid(alpha=0.35)
    finish(
        fig,
        assets / "01_mechanism",
        "Illustration, not a claim about WTI: the company responds to competitor entry by moving liquidation to Q3.",
    )

    frame = pd.read_csv(root / "results/evaluation.csv")
    metrics = json.loads((root / "results/metrics.json").read_text())
    per_run = pd.read_csv(root / "results/per_run.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(top=0.76, bottom=0.23, left=0.06, right=0.96, wspace=0.25)
    fig.text(
        0.06, 0.91, "Does intervention training change the decision?", fontsize=25, weight="bold"
    )
    fig.text(
        0.06,
        0.85,
        f"02 / MEASURED RESULTS     •     {len(metrics['training_seeds'])} training seeds × {metrics['test_paths_per_seed']} paired test paths",
        color=MUTED,
    )
    for label in ["Observational PPO", "Interventional PPO"]:
        annual = frame[frame.policy == label].groupby("week").hedge.mean()
        axes[0].plot(annual.index, annual.values, label=label, color=COLORS[label], lw=2.5)
    axes[0].axvspan(40, 52, color="#a8b6ce", alpha=0.08)
    axes[0].set(
        title="Mean deployed hedge across held-out decisions",
        xlabel="Week of year",
        ylabel="Hedge fraction",
        ylim=(0, 1),
    )
    axes[0].legend(frameon=False, labelcolor=FG, loc="lower left")
    seed_costs = per_run.groupby(["policy", "seed"]).total_cost.mean()
    order = list(COLORS)
    means = [seed_costs.loc[p].mean() for p in order]
    axes[1].barh(np.arange(5), means, color=[COLORS[p] for p in order], height=0.55)
    for i, label in enumerate(order):
        dots = seed_costs.loc[label].values
        axes[1].scatter(dots, np.full(len(dots), i), c=FG, s=25, zorder=3, edgecolors=BG)
    axes[1].set_yticks(np.arange(5), [p.replace(" ", "\n", 1) for p in order])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color=MUTED, lw=0.8)
    axes[1].set(
        title="Procurement cost relative to locking spot; lower is better",
        xlabel="Sum of tranche cost differences + hedge fees (index units)",
    )
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    reduction = metrics["paired_cost_reduction_mean"]
    finish(
        fig,
        assets / "02_results",
        f"Paired mean cost reduction, observational minus interventional: {reduction:+.2f}. Dots = training-seed means.\nKnown-SCM stress test; policies optimize risk-penalized utility. No empirical causal or investment-performance claim.",
    )

    if (root / "data/wti_daily.csv").exists():
        data = pd.read_csv(root / "data/wti_daily.csv", parse_dates=["date"])
        provenance = json.loads((root / "data/provenance.json").read_text())
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.subplots_adjust(top=0.73, bottom=0.20, left=0.07, right=0.96)
        fig.text(
            0.07,
            0.91,
            "Real market context. A separate source of evidence.",
            fontsize=25,
            weight="bold",
        )
        fig.text(
            0.07,
            0.85,
            f"03 / WTI SPOT     •     Latest observation: {provenance['last_observation']}     •     ${data.price.iloc[-1]:.2f}/barrel",
            color="#33c6b5",
        )
        ax.plot(data.date, data.price, color="#33c6b5", lw=1.3)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set(
            ylabel="USD per barrel",
            title="Ten years of daily published WTI observations • EIA via FRED",
        )
        ax.grid(alpha=0.3)
        finish(
            fig,
            assets / "03_market",
            "Public market data is not used to infer company liquidation effects or train these synthetic policies.\nSource: FRED DCOILWTICO / U.S. EIA. Daily observations may be delayed or revised; this is not an intraday feed.",
        )
    write_html(root, metrics)


def write_html(root: Path, metrics: dict):
    market = (
        '<img src="assets/03_market.svg" alt="Ten-year WTI daily price history">'
        if (root / "docs/assets/03_market.svg").exists()
        else "<p>Run causal-hedging fetch to add real market context.</p>"
    )
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Causal AI • Commodity Hedging</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0e1726;color:#f2f5fa;font:17px/1.6 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:48px 28px}h1{font-size:clamp(36px,6vw,68px);line-height:1.08;max-width:950px;margin:24px 0}h2{font-size:29px}p{max-width:900px;color:#a8b6ce}.eyebrow{color:#33c6b5;letter-spacing:3px;font-size:13px;font-weight:700}.tag{display:inline-block;padding:6px 12px;border:1px solid #344057;border-radius:30px;font-size:13px;margin-right:8px}section{margin:65px 0}img{width:100%;height:auto;border:1px solid #344057;border-radius:14px}a{color:#33c6b5}label{display:block;margin:16px 0}input{accent-color:#33c6b5;width:260px;max-width:100%}.lab{border:1px solid #344057;padding:24px;border-radius:14px;background:#142034}svg{max-width:100%;height:auto}output{color:#33c6b5;font-weight:700}button{background:#33c6b5;border:0;border-radius:8px;padding:10px 20px;font:inherit;color:#0e1726;cursor:pointer}:focus-visible{outline:3px solid #e8c77a;outline-offset:5px}.footer{border-top:1px solid #344057;padding-top:22px;font-size:14px}</style>
<main><span class="eyebrow">CAUSAL AI / REPRODUCIBLE EXPERIMENT</span><h1>When the pattern you learn<br>is the policy you used.</h1><p>A Transformer sees a Q4 discount. But part of that discount comes from the company’s own liquidation schedule. Change that schedule, and the decision problem changes with it.</p><span class="tag">104 weekly observations</span><span class="tag">Transformer + corrected PPO</span><span class="tag">Explicit do-interventions</span>
<section><h2>01 · Intervene on the mechanism</h2><p>The simulator separates an exogenous seasonal component, random disturbances, liquidation and a supply shock. Paired worlds keep the disturbances fixed and replace the liquidation equation.</p><img src="assets/01_mechanism.svg" alt="Paired liquidation interventions and the held-out regime shift"></section>
<section class="lab"><span class="eyebrow">INTERACTIVE STRUCTURAL EQUATION</span><h2>What if we do not liquidate in Q4?</h2><p>Change the assumed liquidation effect. The green path has no liquidation; the coral path adds Q4 liquidation. This controls the known simulator, not the trained policies or a real market estimate.</p><label for="effect">Assumed liquidation price effect: <output id="amount">8</output> index points</label><input id="effect" type="range" min="0" max="16" value="8" step="0.5"><button id="reset" type="button">Reset</button><svg id="plot" viewBox="0 0 1000 280" role="img" aria-label="Paired synthetic price paths, with and without Q4 liquidation"><rect x="748" y="20" width="212" height="220" fill="#ee8065" opacity="0.09"/><text x="765" y="44" fill="#a8b6ce">Q4 liquidation</text><path id="no" fill="none" stroke="#33c6b5" stroke-width="3"/><path id="yes" fill="none" stroke="#ee8065" stroke-width="3"/><text x="35" y="268" fill="#a8b6ce">Week 1</text><text x="894" y="268" fill="#a8b6ce">Week 52</text></svg><div aria-live="polite">Q4 intervention contrast: <output id="contrast">−8</output> price-index points. Identical exogenous path.</div></section>
<section><h2>02 · Evaluate the policies on the same worlds</h2><p>Training uses years 1–8, with a two-year input warmup. Years 9–10 are held out. Both policies face the same test paths, including the changed liquidation schedule and an unseen January supply shock. Baselines make the comparison interpretable.</p><img src="assets/02_results.svg" alt="Measured hedge decisions and paired procurement cost comparisons"><p>RESULT_TEXT</p><p>Under-hedging means a sampled hedge below 50%. The saved CSV reports its exact categorical probability; deployment uses the probability-weighted mean hedge. No 94% result is assumed.</p></section>
<section><h2>03 · Keep real data and causal assumptions distinct</h2><p>Free WTI spot observations provide market context. Refresh with <code>causal-hedging fetch</code>. Public prices alone do not identify your company’s liquidation effect.</p>MARKET<p><a href="https://fred.stlouisfed.org/series/DCOILWTICO">FRED / EIA source and updates</a></p></section>
<section><h2>What this demonstrates</h2><p>Policy-dependent patterns can fail under a regime shift. A known structural model lets us ask explicit intervention questions and test a different training distribution. Whether that produces a better hedge is an empirical result of this experiment—not a guarantee of causal RL.</p><p>This is a synthetic procurement demo. It does not model owned-inventory P&amp;L, futures basis, margin or market impact from the hedge itself. The causal agent is given the correct simulator; it does not discover the SCM from WTI prices.</p></section><p class="footer">Causal AI · <a href="https://github.com/sandipdikshit/Causal-AI">Source, tests and reproduction instructions</a> · Figures generated from saved experiment outputs.</p></main>
<script>const effect=document.querySelector('#effect');function draw(){const e=Number(effect.value);document.querySelector('#amount').textContent=e;document.querySelector('#contrast').textContent=(-e).toFixed(1);function path(on){return Array.from({length:52},(_,i)=>{const p=100+1.5*Math.sin(2*Math.PI*i/52)-(on&&i>=39?e:0);return `${i?'L':'M'}${35+i*925/51},${45+(103-p)*9}`}).join(' ')}document.querySelector('#no').setAttribute('d',path(false));document.querySelector('#yes').setAttribute('d',path(true))}effect.addEventListener('input',draw);document.querySelector('#reset').addEventListener('click',()=>{effect.value=8;draw()});draw();</script></html>"""
    result = f"Measured paired cost reduction (observational minus interventional): {metrics['paired_cost_reduction_mean']:+.2f} index units. Positive values favor interventional training; see all seeds in results/metrics.json."
    (root / "docs/index.html").write_text(
        html.replace("MARKET", market).replace("RESULT_TEXT", result)
    )
