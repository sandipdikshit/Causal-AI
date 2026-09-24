# Three-minute demo

1. Open `docs/index.html`. Explain the business exposure: future purchases, not already-owned inventory. An unhedged buyer benefits if prices fall.
2. Show the two paired worlds. The liquidation effect is specified in this simulator. Move the slider to zero: the intervention gap vanishes while exogenous seasonality stays.
3. Show the regime shift. The company moves liquidation to Q3 after competitor entry; the Q4 discount disappears. January year 10 adds an unseen supply shock.
4. Show measured policy results and the fully hedged baseline. Explain that both learned policies see identical held-out paths, and the interventional policy receives the known SCM through randomized intervention rollouts. Report the actual result; do not promise improvement.
5. Show WTI. Explain that it is the latest downloaded daily public series. It makes the market context tangible, but cannot establish this company's causal effect.

The original “94%” figure is not part of this experiment. Exact under-hedging probabilities are saved in `results/evaluation.csv` and refer to the categorical training policy; the deployed hedge is its probability-weighted mean.
