# SMAP/MSL benchmark

This is an honest check of the L2 idea (learn a next-step predictor, then threshold its errors) on NASA's
labeled spacecraft telemetry anomalies (Hundman et al., KDD 2018). We score it exactly the way that paper
scores Telemanom. Every number below comes from `results/bench_summary*.json`, which
`scripts/run_bench.py` writes.

## Results (event-level, 105 labeled anomalies)

HEADLINE_TABLE

**Telemanom baseline:** Hundman et al., KDD 2018, [arXiv:1802.04431](https://arxiv.org/abs/1802.04431),
Table 2, row "Non-Parametric w/ Pruning (p = 0.13)". The khundman/telemanom README repeats the same
precision and recall. The paper reports precision, recall and F0.5 but **not F1**, so the F1 values marked
\* are computed from its precision and recall. We don't compare on F0.5 because the paper's printed F0.5
values don't follow from its own printed precision and recall. For example, SMAP has P = R = 85.5%,
which gives F0.5 = 0.855, but the paper prints 0.71.

ABLATION_SECTION

## Method

| Stage | What we do | Code |
|---|---|---|
| Data | 82 rows of telemanom's `labeled_anomalies.csv`: 55 SMAP rows (69 anomalies) and 27 MSL rows (36 anomalies), the same counts as the paper's Table 1. Column 0 is telemetry, the rest are command one-hots (25 / 55 inputs). | `vbw/bench/data.py` |
| Predictor | One model per channel: 2x80 LSTM, dropout 0.3, input 250 steps, predicts the next 10 values and is scored on the first. Adam, batch 64, up to 35 epochs, early stopping on a 20% random validation split of *train* (patience 10, min_delta 3e-4), best weights restored. These match the paper's Model Parameters table. | `vbw/bench/model.py` |
| Thresholding | Nonparametric dynamic thresholding, ported from `telemanom/errors.py`. Absolute errors get an EWMA (span 105). Batches of 70 are judged against a 2100-point trailing window. epsilon = mean + z·std, with z picked from 2.5 to 11.5 to maximize (Δmean/mean + Δstd/std) / (\|E_seq\|² + \|e_a\|), with guardrails. Errors mirrored about their mean are also checked, flagged points are padded by ±100, and results are pruned with p = 0.13. | `vbw/bench/threshold.py` |
| Scoring | A labeled sequence overlapped by any predicted sequence counts as a TP (once). Each labeled sequence with no overlap is an FN. Each predicted sequence that overlaps no label is an FP. This is the paper's Sec. 4.1 definition. Point-adjusted F1 is also computed, but only as an optimistic reference. | `vbw/bench/metrics.py`, `evaluate.py` |
| Orchestration | 81 unique channels, trained in parallel across 5 MPS and 2 CPU worker processes (1 torch thread each). | `vbw/bench/runner.py` |

We tuned nothing on the test labels. Every hyperparameter is the paper's default or the reference code's.

### Protocol notes and quirks

- **P-2** shows up twice in the CSV. Both copies are SMAP, with slightly different labels:
  `[[5350, 6575]]` and `[[5300, 6420]]`. Telemanom scores both rows, and that's how it reaches 55 SMAP
  channels and 69 anomalies, so we score both too (one model, two label rows). OmniAnomaly-style
  point-wise splits drop P-2 completely (`row[0] != 'P-2'` in its `data_preprocess.py`), so those numbers
  aren't directly comparable.
- **T-10** has `.npy` files in the TranAD mirror but no label row, so we don't evaluate it.
- **C-2:** the telemanom code skips its start-of-series smoothing fix for this one channel, because the
  anomaly falls early. That choice uses label knowledge, so we don't copy it.
- **Bug fixes vs. the reference code.** The last, longer thresholding window now judges all of its new
  points; the reference judges only the final 70. Padding is also ±100 in both places, where the
  reference mixes 99 and 100.
- **Framework.** We use PyTorch where the original used Keras, so initialization and LSTM gate details
  differ.

## Caveats

- **Our predictor is worse than the paper's.** Mean normalized one-step error was MEAN_ERR (median
  MEDIAN_ERR). The paper reports 5.5% for SMAP and 6.8% for MSL. Some of the gap comes from channels
  whose training split never visits the test regime: D-5, D-2, B-1, D-13, A-5 and P-14 are constant in
  train and move in test, so the model keeps predicting the training constant.
- **Run-to-run variation.** Channels go to MPS or CPU workers in whatever order the queue hands them
  out, so floating-point results differ slightly between runs. REPEAT_SENTENCE A proper multi-seed
  study has not been done.
- **The paper flags its own precision as optimistic.** Its experiment only evaluates streams known to
  contain an anomaly, about 5 days around each one. That caveat applies here too.
- **Point-adjusted F1 is not a fair comparison.** It's often quoted for this dataset, but it credits a
  whole labeled segment when a single point inside it is hit. It's in the JSON (`point_adjusted_optimistic`)
  only so nobody mistakes it for the real number.
- **Sample sizes are small.** With 36 MSL anomalies, one event moves recall by 2.8 points.

## What would close the gap

1. **Better command features and longer training history for MSL.** MSL is where most of the recall is
   lost. The paper points to richer command and event features as its own next step, and its MSL
   training data also misses rover behaviours seen in test.
2. **Fixing the regime-shift channels.** These channels are constant in train and move in test. Two
   options: give them a train window that covers the test regime (telemanom adds extra days when they
   exist), or normalize each channel's input.
3. **Not just "predict better".** The residual ablation above shows that the error-thresholding stage
   needs errors that stay large during anomalies. It doesn't need the lowest forecast error.

## Reproduce

```bash
uv run python scripts/fetch_smap.py                      # data/raw/smap_msl (once)
uv run python scripts/run_bench.py                       # headline -> results/bench_summary.json
uv run python scripts/run_bench.py --residual --tag residual   # ablation -> results/bench_summary_residual.json
uv run pytest tests/test_bench_*.py                      # thresholding, metrics, labels (~3 s)
```

Useful flags: `--workers N` (CPU processes), `--mps-workers N`, `--channels P-1,M-6` (smoke test),
`--cache-dir DIR` (reuse predictions, which makes re-scoring take seconds), `--out-dir DIR`.

Outputs: `results/bench_summary.json` (aggregates, model config, runtime, baseline, notes) and
`results/bench_channels.json`. The channels file has, per label row: predicted and labeled sequences,
hits, false alarms, scores, normalized error, epochs, and device.
