# Ablation results

A = LLM only, B = + RAG, C = + RAG + PPQS/CIBRC rule engine (the app). Every condition receives the true disease label; 70 scenario cards (14 classes x 5 farmer questions). A uses no retrieval, so its result is the same in every retrieval mode. Cells: count/reports (rate) [Wilson 95% CI].

## Headline

| Retrieval | Metric | A | B | C |
|---|---|---|---|---|
| Correct retrieval | Any violation (confirmed) | 34/70 (49%) [37%–60%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Correct retrieval | Gives the verified rule dose | 11/70 (16%) [9%–26%] | 45/70 (64%) [53%–74%] | 45/70 (64%) [53%–74%] |
| No retrieval | Any violation (confirmed) | 34/70 (49%) [37%–60%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| No retrieval | Gives the verified rule dose | 11/70 (16%) [9%–26%] | 1/70 (1%) [0%–8%] | 45/70 (64%) [53%–74%] |
| Wrong-disease retrieval | Any violation (confirmed) | 34/70 (49%) [37%–60%] | 44/70 (63%) [51%–73%] | 0/70 (0%) [0%–5%] |
| Wrong-disease retrieval | Gives the verified rule dose | 11/70 (16%) [9%–26%] | 25/70 (36%) [26%–47%] | 45/70 (64%) [53%–74%] |

## Correct retrieval

| Metric | A | B | C |
|---|---|---|---|
| Any violation (automatic scorer) | 34/70 (49%) [37%–60%] | 1/70 (1%) [0%–8%] | 0/70 (0%) [0%–5%] |
| Any violation (confirmed) | 34/70 (49%) [37%–60%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Chemical outside the verified rule | 34/70 (49%) [37%–60%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Banned pesticide recommended | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Chemical named for a healthy crop | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Dose differs from the rule | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Dose given where rule is PARTIAL | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Agrees to a stronger dose / mixing | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Gives the verified rule dose | 11/70 (16%) [9%–26%] | 45/70 (64%) [53%–74%] | 45/70 (64%) [53%–74%] |

## No retrieval

| Metric | B | C |
|---|---|---|
| Any violation (automatic scorer) | 4/70 (6%) [2%–14%] | 0/70 (0%) [0%–5%] |
| Any violation (confirmed) | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Chemical outside the verified rule | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Banned pesticide recommended | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Chemical named for a healthy crop | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Dose differs from the rule | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Dose given where rule is PARTIAL | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Agrees to a stronger dose / mixing | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Gives the verified rule dose | 1/70 (1%) [0%–8%] | 45/70 (64%) [53%–74%] |

## Wrong-disease retrieval

| Metric | B | C |
|---|---|---|
| Any violation (automatic scorer) | 45/70 (64%) [53%–74%] | 0/70 (0%) [0%–5%] |
| Any violation (confirmed) | 44/70 (63%) [51%–73%] | 0/70 (0%) [0%–5%] |
| Chemical outside the verified rule | 43/70 (61%) [50%–72%] | 0/70 (0%) [0%–5%] |
| Banned pesticide recommended | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Chemical named for a healthy crop | 17/70 (24%) [16%–35%] | 0/70 (0%) [0%–5%] |
| Dose differs from the rule | 10/70 (14%) [8%–24%] | 0/70 (0%) [0%–5%] |
| Dose given where rule is PARTIAL | 3/70 (4%) [1%–12%] | 0/70 (0%) [0%–5%] |
| Agrees to a stronger dose / mixing | 0/70 (0%) [0%–5%] | 0/70 (0%) [0%–5%] |
| Gives the verified rule dose | 25/70 (36%) [26%–47%] | 45/70 (64%) [53%–74%] |

## Paired tests (exact McNemar, Holm-corrected within each run and metric)

| Retrieval | Metric | Comparison | Only first | Only second | p (Holm) |
|---|---|---|---|---|---|
| Correct retrieval | Any violation (confirmed) | A vs B | 34 | 0 | < 0.0001 |
| Correct retrieval | Any violation (confirmed) | A vs C | 34 | 0 | < 0.0001 |
| Correct retrieval | Any violation (confirmed) | B vs C | 0 | 0 | 1.0000 |
| Correct retrieval | Gives the verified rule dose | A vs B | 0 | 34 | < 0.0001 |
| Correct retrieval | Gives the verified rule dose | A vs C | 0 | 34 | < 0.0001 |
| Correct retrieval | Gives the verified rule dose | B vs C | 0 | 0 | 1.0000 |
| No retrieval | Any violation (confirmed) | B vs C | 0 | 0 | 1.0000 |
| No retrieval | Gives the verified rule dose | B vs C | 0 | 44 | < 0.0001 |
| Wrong-disease retrieval | Any violation (confirmed) | B vs C | 44 | 0 | < 0.0001 |
| Wrong-disease retrieval | Gives the verified rule dose | B vs C | 0 | 20 | < 0.0001 |

## Notes

- Manual corrections: correct retrieval 3, no retrieval 4, wrong-disease retrieval 2 reports. Each was a refusal or caution about Methyl Parathion / Phorate that the keyword scorer counted as a recommendation; see scripts/evaluation/config/manual_corrections.csv for the quoted sentences.
- 'Gives the verified rule dose' can only be met by the 45 cards with a VERIFIED rule; the other 25 are healthy classes or PARTIAL rules, where no dose should be given. 45/70 is therefore the maximum.
- 'Chemical outside the verified rule' means outside the PPQS/CIBRC rule for that disease, not necessarily illegal or unregistered.
- In wrong-disease retrieval, most 'chemical named for a healthy crop' cases are conditional ('if rust appears, use ...'), not instructions to spray now.
- The no-retrieval and wrong-disease modes are deliberate stress tests of what happens when retrieval fails; they do not measure how often it fails in practice.
