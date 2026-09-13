# Capstone Report — <your lane>

- **Author:** Prathibha Shalini S
- **Lane:** CTR / Engagement Opportunity Scoring for Search Content Review
- **Repo:** https://github.com/PrathibhaShaliniS/flyrank-internship-ml
- **Date:** 13 September 2026

## 0. Abstract

Which visible search pages under-capture clicks for how well they rank, and how confidently
can that be found and prioritized for review? Using FlyRank's anonymized starter dataset
(30,000 pages, 32 clients), a page is defined as underperforming if its click-through rate
sits below its own position tier's *median* CTR — a threshold chosen only after a signal audit
showed the tier mean is distorted by a handful of high-traffic outliers. A hand-written
baseline rule (staleness plus a flat CTR cutoff) performs *worse than random* at finding these
pages (ROC-AUC 0.420, against a 0.496 base rate), while a Random Forest classifier trained on
pre-click content and visibility features reaches 0.800 ROC-AUC and 0.98 precision@50 on a
held-out set of unseen clients. The output is a ranked, probability-based review list — with a
minimum-visibility floor applied, since the model is otherwise overconfident on tiny-sample
pages — meant to help a content team decide which pages get a metadata/snippet review first.
## 1. Problem framing

**Unit of analysis:** one page (`content_id`), one 90-day snapshot.
**Output:** a probability that a page's CTR underperforms its position tier, turned into one
of three actions — `review_snippet_ctr`, `monitor`, `no_action`.
**Who acts on it:** a content/SEO team with limited review capacity, deciding which pages to
send for a title/meta-description review this cycle.
**Cost of a wrong call:** a false positive wastes review time on a page that's fine; a false
negative leaves real lost clicks unaddressed for another cycle. Neither is severe, which is
why the output is a ranked list, not a hard yes/no — a team can draw the line wherever their
capacity allows.
**Why ML helps:** position alone predicts *typical* CTR well (confirmed below), but individual
pages deviate from their tier's typical CTR for reasons — content type, age, freshness, depth
— that a single hand-written rule can't combine the way a model can.
## 2. Data safety

**Data used:** `data/raw/content_refresh_anonymized.csv` — the anonymized starter release
(30,000 rows, one row per page, 32 pseudonymized clients), drawn from the pseudonymized
warehouse snapshot described in this repo's data dictionary. All traffic/engagement metrics
are one trailing 90-day aggregate window, not a time series.

**Deliberately excluded, and why:**
- `ctr` and `clicks_90d` as **model features** — both were used to *define* the label
  (`needs_ctr_review = ctr < tier median ctr`), so including either would let the model
  reconstruct its own answer directly.
- `trend_direction` / `trend_pct` — FlyRank's own decline label, not a raw observation; using
  it as an input would mean training on another model's conclusion.
- `provider_used` / `model_used` — marked not-for-modeling in this repo's data dictionary.
- `content_id` / `client_id` — used only to group the train/test split, never as features.
- No client names, domains, URLs, or raw queries appear anywhere in this file or any notebook
  in this repo — it ships pre-anonymized.
## 3. Baseline

**The rule (`w04_baseline_score.ipynb`):** a page is worth reviewing if it's visible
(≥500 impressions/90d) and either stale (≥180 days since last update) or under-clicking for
its position (top-20 average position, CTR < 0.50%). Score =
`visible × impressions_90d × (stale + ctr_gap)`.

**Fairness of the comparison:** the baseline is scored on the exact same rows, split, and
label as every model below — not the broader notion of "worth reviewing" it was originally
written for, but specifically "is this page below its own tier's median CTR." That's a strict
test of whether a simple, transparent rule finds what this lane actually cares about.

**Its numbers (test split, base rate 0.496):** ROC-AUC 0.420, average precision 0.453,
precision@20 0.15, precision@50 0.22, precision@100 0.23 — **worse than random** on this
specific question, since it was optimized for a different, broader notion of "needs review."
## 4. Model / analysis

**Method:** classification — Logistic Regression, a depth-4 Decision Tree, and Random Forest,
matching `training-honest-models`' guidance for a yes/no question with an observable label.

**Target — `needs_ctr_review`:** 1 if a page's CTR is below its own `position_tier`'s median
CTR, restricted to `page_1`/`striking`/`page_3_5` only. `top_3` and `deep` are excluded because
their median CTR is exactly 0% — "below the median" is impossible there by construction, not a
gap worth faking a label for.

**Features (36 after one-hot encoding):** `search_volume`, `competition`, `cpc`, `word_count`,
`char_count`, `days_with_impressions`, `content_age_days`, `days_since_last_update`,
`avg_position`, `impressions_90d`, two "has data" flags, plus `content_type`, `main_intent`,
`competition_level`, `freshness_tier`, `word_count_tier`, `position_tier` — all pre-click,
observable signals.
**Left out on purpose:** `ctr`/`clicks_90d` (label-derived — see Section 2), and post-click
metrics (`engagement_rate`, `scroll_rate`, `sessions_90d`) — predicting a pre-click outcome
from post-click behavior would be causally backwards, even where it isn't technical leakage.

## 5. Evaluation

**Split:** client-grouped, not random rows — pages from the same client share templates and
style, so a random split would leak that similarity across train/test. ~20% of clients (6 of
31) are held out entirely. With only 31 clients, one random group-split can land unbalanced by
pure chance — a `random_state=42` draw gave 85% positive rate in test vs. 48% in train — so
200 candidate splits were searched for the closest train/test label-rate match (seed 56: 0.496
vs. 0.496). A follow-up check across five unrelated seeds showed train/test label-rate gaps
ranging from 4.7 to 35.8 percentage points, confirming this wasn't a one-off risk.

**Results (same split, same metric, base rate = 0.496):**

| Model | precision@20 | precision@50 | precision@100 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|
| Week-4 baseline (rule) | 0.15 | 0.22 | 0.23 | 0.420 | 0.453 |
| Logistic Regression | 0.90 | 0.84 | 0.89 | 0.801 | 0.796 |
| Decision Tree (depth 4) | 0.90 | 0.90 | 0.83 | 0.735 | 0.717 |
| **Random Forest** | **0.95** | **0.98** | **0.98** | **0.800** | 0.790 |

**Errors:** accuracy by tier is 77% (`page_3_5`), 71% (`striking`), 69% (`page_1`) — hardest
where "good" and "bad" CTR sit closest together. The worst false positives are tiny-volume
pages (2–11 impressions) where one lucky click created a spuriously high CTR. The worst false
negatives are high-volume `page_3_5` pages (9,800–22,000 impressions) with a genuinely poor
CTR (0.00–0.02%) that the model still called "fine," having learned "high volume usually means
OK" as a generally-correct but imperfect rule.

## 6. Interpretation

`impressions_90d` and `days_with_impressions` together make up ~67% of Random Forest's feature
importance. This is defensible — low-volume pages produce noisy CTR ratios, so volume is a
reasonable signal for *how much to trust* a page's own CTR — but it's also exactly why the
errors above cluster at the volume extremes, not a triumph to overstate.

**A negative result worth stating plainly:** the signal audit (`w04_signal_audit.ipynb`) found
that CTR does **not** track visibility volume in any consistent direction across impression
tiers. The model's reliance on volume isn't rediscovering a strong direct relationship — it's
using volume as a noise indicator, a different and more defensible use of the same feature.
`avg_position`, `content_age_days`, `word_count`, and `char_count` matter, but each contributes
far less individually.

## 7. Recommendation

Scoring all 26,360 eligible rows with the trained Random Forest and applying a ≥50-impression
reliability floor (4,416 rows excluded as unreliable) leaves 21,944 actionable pages:

| Action | Pages |
|---|---|
| `review_snippet_ctr` | 2,836 |
| `monitor` | 7,332 |
| `no_action` | 11,776 |
Total actionable pages	21,944

The top of the ranked list is consistently `page_3_5`/`striking` pages with real impression
volume (50–71 impressions) and literally 0% measured CTR at decent-to-average positions
(avg_position 17.8–48.6) — a content editor could reasonably start there tomorrow, since these
are the pages most likely to have a real, fixable snippet problem rather than statistical
noise.

**Confidence and limits:** this ranks *relative* underperformance within a page's own tier, not
an absolute CTR standard, and covers only `page_1`/`striking`/`page_3_5` by design — `top_3`
and `deep` (12.1% of all pages) are out of scope, compounded by the label-quality issue in
`top_3` noted in Section 2.

## 8. Reproducibility

**Environment:** Python 3.12, `pandas`, `scikit-learn==1.8.0`, `matplotlib`.
**To re-run from a fresh clone:**
```
git clone https://github.com/PrathibhaShaliniS/flyrank-internship-ml.git
cd flyrank-internship-ml
pip install pandas scikit-learn matplotlib
jupyter execute work/notebooks/w04_baseline_score.ipynb --output=work/notebooks/w04_baseline_score.ipynb
jupyter execute work/notebooks/w04_signal_audit.ipynb --output=work/notebooks/w04_signal_audit.ipynb
jupyter execute work/notebooks/capstone.ipynb --output=work/notebooks/capstone.ipynb
```
**Random seeds:** `RANDOM_STATE = 42` for every model. The client-split search tried seeds
0–199 and selected 56 for the closest train/test label-rate match; both values are printed in
the notebook, so a fresh run reproduces the table in Section 5 exactly.
No sealed/holdout evaluation beyond the client-grouped test split is claimed here.
## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
