Subject: two inference-only runs, no retraining — guide Task 0.2

Jyotir, Omkar,

Pull first. Two decisions since the last message, then one task.

Decision 1: the operating-point / matched-selectivity contribution is removed from the paper in full — abstract paragraph, intro paragraph, contribution 3, both run-ins, the selectivity figure, and every single-seed row of Table 2. The paper now reports one protocol: five-seed ensembles for the MLP and transformers, single fits for LightGBM and the logistic. Ground rule 8b in the guide lists everything that must never come back. Your Task 0.1 artifacts stay committed as the internal record; do not delete them.

Decision 2: to preempt the referee question "do the rankings depend on how many trades each model takes?", the paper will report every model under two prespecified rules: the committed probability rule (calibrated p >= 0.5) and a coverage rule (each model's top N = 14,000 candidates by calibrated probability). Two rules, reported side by side, no commentary.

The task is guide Task 0.2. It is inference-only: every input is a committed prediction file (your script 67 saved the MLP-5 population predictions; the transformer, LightGBM, and logistic files already exist). No model is retrained. In outline:

1. Reproduce each family's committed p>=0.5 row first (gates listed in the guide). A failed gate stops the task.
2. Compute the coverage rule: one action per episode, ties toward T1, top 14,000 by calibrated probability, full metrics() row per model. Apply the same one-action convention to LightGBM — that also settles the ~5% selection-file discrepancy (old Task 3.1).
3. Recompute from the five-seed ensemble predictions: the strategy-level T1/K2 breakdown, the year-by-year bands, and the classification-metrics MLP row (currently single-seed).
4. Commit two_rule_comparison.json and fiveseed_breakdowns.json, then fill the marked tex slots — grep "TODO-RERUN[two-rule]" and "RERUN-REQUIRED[fiveseed-reanchor]". Templates are in the comments. Both variants compile to at most 8 pages, zero undefined references, artifact and tex in one commit.

Voice rule as always: short sentences, no evaluative adjectives, and no sentence about thresholds, operating points, or ranking reversals — the two rules are just two evaluations.

This is a few hours of work off existing files. Ping me with the gate numbers when step 1 passes.

Pawel
