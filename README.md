# Causal AI · Executed experiments

Four executed notebooks present the experiments in one place each:

- **[Commodity hedging and liquidation interventions](RL_Experiment.ipynb):** Transformer-PPO in a known structural model, with paired counterfactual paths.
- **[RLHF pricing and a causal preference firewall](RLHF_Pricing_Causal_Preference_Firewall.ipynb):** a synthetic reward model trained on biased pairwise advice preferences, checked against randomized pricing, an encouragement instrument and a threshold design.
- **[Propensity versus causal uplift](Propensity_vs_Causal_Uplift.ipynb):** a real randomized email-campaign dataset, fixed-budget targeting policies, held-out incremental conversions, bootstrap uncertainty, and explicit cost scenarios.
- **[Attention after the fact](attention_after_the_fact.ipynb):** a public-retail-calibrated transformer post-mortem showing how standard attention routes through a downstream mediator, and how a causal feature mask changes the learned representation and counterfactual effect estimate.

All three contain code, explanations, tables and inline figures. Saved outputs render on GitHub.

The commodity notebook brings together its structural model, training, results and five inline charts. The pricing notebook independently demonstrates how preference accuracy can disagree with contribution margin.

The commodity experiment asks whether a policy trained on historical liquidation patterns behaves differently after that schedule changes, and what explicit intervention training changes. It is a **synthetic procurement experiment with a known SCM**. Real WTI prices provide separate market context; they do not identify a company's causal liquidation effect.

## Run the notebooks

Python 3.11+; CPU is sufficient. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab RL_Experiment.ipynb
# Or open RLHF_Pricing_Causal_Preference_Firewall.ipynb in JupyterLab
```

Choose **Restart Kernel → Run All** in any notebook. The commodity run trains two policies across three seeds, using 104-week inputs and eight paired test paths per seed. The pricing notebook runs a small synthetic preference model and three simulated causal designs. The commodity notebook regenerates detailed CSVs in the ignored `notebook-runs/` directory; the pricing and uplift notebooks display their results inline. The uplift notebook downloads and verifies the public [MineThatData dataset](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) on first run; internet access is needed once.

WTI observations and their provenance are in `data/`. Set `REFRESH_WTI=True` in the notebook to download current published daily observations from [FRED/EIA](https://fred.stlouisfed.org/series/DCOILWTICO). This is not an intraday feed.

## Reproducibility

- [`requirements-reference.txt`](requirements-reference.txt) preserves the exact model-library versions used for the saved outputs; install it after `requirements.txt` to match those versions. Cross-platform numerical differences remain possible.
- [`results/metrics.json`](results/metrics.json) preserves configuration, seeds and paired results; [`results/summary.csv`](results/summary.csv) preserves the compact benchmark table.
- Tests load the commodity model definitions **directly from its notebook**, checking interventions, information boundaries, PPO likelihoods, terminal handling and a short training run. They also check the pricing notebook’s saved execution and reported designs.

```bash
pip install 'pytest>=8,<10' 'ruff>=0.9,<1'
pytest -q
ruff check .
```

In the saved run, interventional training reduced paired mean procurement cost by **1.60 synthetic index units**. The unhedged baseline had lower raw cost but worse risk-penalized utility. These are results of a specified simulation, not investment returns or proof of universal causal-RL superiority. No 94% under-hedging result is assumed.

Each notebook is the single source of code for its experiment. Earlier package, visual-demo and detailed-results versions remain recoverable in [Git history](https://github.com/sandipdikshit/Causal-AI/commits/main/).

The pricing notebook is **synthetic evidence of a mechanism**, not a verification of live RLHF deployment or a 35% observed margin loss. It uses a transparent four-recommendation preference model, not a fine-tuned LLM. The instrument identifies a simulated complier effect, and the threshold design identifies a local effect.

The uplift notebook retains a negative/uncertain comparison: at a predeclared 30% contact budget, its simple T-learner did not beat response-propensity targeting in the held-out sample (paired 95% bootstrap interval includes zero). The public randomized data support a methodological demonstration, not a claim that uplift always wins or an observed profit improvement.
