# Example Proposal Trace

## Teacher Proposal
- image_id: `1`
- proposal_id: `1`
- class: `person`
- score: `0.5244`
- bbox: `[2.0, 51.0, 568.0, 772.0]`

## Proposal Verification Output
- visual_validity: `valid`
- evidence_strength: `strong`
- recommended_action: `keep`
- inspection_status: `success`

## LLM Final Decision Output
- llm_final_status: `success`
- llm_final_decision: `keep`
- llm_final_confidence: `high`

## Final Action
- final_decision: `keep`
- final_decision_source: `llm_final_decision`
- fallback_used: `False`

## Offline Evaluation
- gt_correct: `True`
- eval_label: `tp`
