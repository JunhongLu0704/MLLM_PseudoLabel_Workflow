# Example Proposal Trace

## Teacher Proposal
- image_id: `0`
- proposal_id: `0`
- class: `bird`
- score: `0.9531`
- bbox: `[433.0, 211.0, 616.0, 385.0]`

## Proposal Verification Output
- visual_validity: `invalid`
- evidence_strength: `strong`
- recommended_action: `drop`
- inspection_status: `success`

## LLM Final Decision Output
- llm_final_status: `success`
- llm_final_decision: `drop`
- llm_final_confidence: `high`

## Final Action
- final_decision: `drop`
- final_decision_source: `llm_final_decision`
- fallback_used: `False`

## Offline Evaluation
- gt_correct: `False`
- eval_label: `tn`
