# Tool: build_candidate_pool

Purpose: build a bounded set of proposals that should receive verification or optional reference evidence.

Required input:

```json
{
  "image_id": "17",
  "score_low": 0.10,
  "score_high": 0.30,
  "top_n_sanity": 2,
  "max_candidates": 32
}
```

The runtime supplies proposals and duplicate cluster information.

Output:

```json
{
  "tool_name": "build_candidate_pool",
  "status": "success",
  "payload": {
    "image_id": 17,
    "candidate_proposal_ids": [1, 4, 7],
    "candidate_reasons": {
      "1": ["uncertain_score_band"],
      "4": ["duplicate_cluster"],
      "7": ["top_score_sanity_check"]
    },
    "pool_policy": "uncertain_plus_structural_risk_v1",
    "max_candidates": 32,
    "status": "success"
  }
}
```

Error behavior:

- Invalid score range or limits: `status=error`, `error_code=invalid_argument`.

Decision rule:

Candidate pool construction bounds the workflow. It is not a final decision.

