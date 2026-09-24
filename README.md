# Causal AI · Commodity Hedging

**[Read the executed experiment notebook](Causal_AI_Experiment.ipynb)**

A single notebook brings together the explanation, structural model, Transformer-PPO code, training, results and five inline charts. Saved outputs let you read it directly on GitHub.

The experiment asks whether a policy trained on historical liquidation patterns behaves differently after that schedule changes, and what explicit intervention training changes. It is a **synthetic procurement experiment with a known SCM**. Real WTI prices provide separate market context; they do not identify a company's causal liquidation effect.

## Run the notebook

Python 3.11+; CPU is sufficient. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab Causal_AI_Experiment.ipynb
```

Choose **Restart Kernel → Run All**. The default run trains two policies across three seeds, using 104-week inputs and eight paired test paths per seed. Detailed outputs are regenerated in the ignored `notebook-runs/` directory.

WTI observations and their provenance are in `data/`. Set `REFRESH_WTI=True` in the notebook to download current published daily observations from [FRED/EIA](https://fred.stlouisfed.org/series/DCOILWTICO). This is not an intraday feed.

## Reproducibility

- [`requirements-reference.txt`](requirements-reference.txt) preserves the exact model-library versions used for the saved outputs; install it after `requirements.txt` to match those versions. Cross-platform numerical differences remain possible.
- [`results/metrics.json`](results/metrics.json) preserves configuration, seeds and paired results; [`results/summary.csv`](results/summary.csv) preserves the compact benchmark table.
- Tests load the model definitions **directly from the notebook**, checking interventions, information boundaries, PPO likelihoods, terminal handling and a short training run.

```bash
pip install 'pytest>=8,<10' 'ruff>=0.9,<1'
pytest -q
ruff check .
```

In the saved run, interventional training reduced paired mean procurement cost by **1.60 synthetic index units**. The unhedged baseline had lower raw cost but worse risk-penalized utility. These are results of a specified simulation, not investment returns or proof of universal causal-RL superiority. No 94% under-hedging result is assumed.

The notebook is the single source of experiment code. Earlier package, visual-demo and detailed-results versions remain recoverable in [Git history](https://github.com/sandipdikshit/Causal-AI/commits/main/).
