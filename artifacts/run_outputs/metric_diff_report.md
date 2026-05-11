# Metric Diff Report

| item | old | new | delta |
|---|---:|---:|---:|
| `num_total_proposals` | 1921 | 1921 | 0 |
| `num_valid_proposals` | 1776 | 1776 | 0 |
| `score_baseline_020.tp` | 124 | 124 | 0 |
| `deterministic_gate.tp` | 124 | 162 | 38 |
| `score_baseline_020.fp` | 559 | 559 | 0 |
| `deterministic_gate.fp` | 253 | 355 | 102 |
| `score_baseline_020.fn` | 74 | 74 | 0 |
| `deterministic_gate.fn` | 74 | 36 | -38 |

## Proposal-Level Decision Differences

- proposals where deterministic gate differs from score baseline: `632`
- If aggregate metrics differ, inspect `proposal_results.csv` columns `gt_correct`, `max_iou_same_class`, `degenerate_bbox`, and `final_decision`.

## Old CSV Proposal-Level Comparison

- old reference CSV: `projects/voc_clipart_review_v2/results/verify_gate_strict_invalid_drop_256/proposal_decision_table.csv`
- common proposals: `1921`
- proposals only in new CSV: `0`
- proposals only in old CSV: `0`
- GT correctness mismatches: `0`
- deterministic gate decision mismatches: `314`

### First Gate Decision Mismatches

| image_id | proposal_id | old_gate | new_gate | old_validity | new_validity | score |
|---:|---:|---|---|---|---|---:|
| 0 | 0 | `keep` | `drop` | `valid` | `invalid` | 0.9531 |
| 0 | 1 | `keep` | `drop` | `valid` | `invalid` | 0.7363 |
| 1 | 1 | `drop` | `keep` | `invalid` | `valid` | 0.5244 |
| 1 | 3 | `drop` | `keep` | `invalid` | `valid` | 0.1060 |
| 3 | 5 | `keep` | `drop` | `valid` | `invalid` | 0.1384 |
| 10 | 0 | `keep` | `drop` | `valid` | `invalid` | 0.7334 |
| 12 | 0 | `keep` | `drop` | `valid` | `invalid` | 0.2878 |
| 15 | 1 | `keep` | `drop` | `valid` | `invalid` | 0.3933 |
| 16 | 7 | `keep` | `drop` | `valid` | `invalid` | 0.1359 |
| 22 | 3 | `drop` | `keep` | `invalid` | `valid` | 0.2390 |
| 28 | 1 | `drop` | `keep` | `invalid` | `valid` | 0.3049 |
| 28 | 2 | `drop` | `keep` | `invalid` | `valid` | 0.2810 |
| 29 | 3 | `drop` | `keep` | `invalid` | `valid` | 0.1993 |
| 29 | 7 | `keep` | `drop` | `valid` | `invalid` | 0.1049 |
| 30 | 3 | `keep` | `drop` | `uncertain` | `invalid` | 0.3254 |
| 30 | 5 | `keep` | `drop` | `valid` | `invalid` | 0.3149 |
| 30 | 7 | `keep` | `drop` | `valid` | `invalid` | 0.2739 |
| 32 | 0 | `keep` | `drop` | `valid` | `invalid` | 0.8706 |
| 32 | 3 | `drop` | `keep` | `invalid` | `valid` | 0.2098 |
| 40 | 0 | `keep` | `drop` | `valid` | `invalid` | 0.5679 |
| 44 | 0 | `drop` | `keep` | `invalid` | `valid` | 0.2510 |
| 49 | 0 | `drop` | `keep` | `invalid` | `valid` | 0.2512 |
| 56 | 0 | `keep` | `drop` | `valid` | `invalid` | 0.7241 |
| 56 | 1 | `keep` | `drop` | `valid` | `invalid` | 0.5640 |
| 56 | 2 | `drop` | `keep` | `invalid` | `valid` | 0.2905 |
