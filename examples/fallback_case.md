# Example Proposal Trace

## Teacher Proposal
- image_id: `3`
- proposal_id: `2`
- class: `dog`
- score: `0.1672`
- bbox: `[7.0, 683.0, 95.0, 684.0]`

## Proposal Verification Output
- visual_validity: `unavailable`
- evidence_strength: `unavailable`
- recommended_action: `drop`
- inspection_status: `not_candidate_heuristic`

## LLM Final Decision Output
- llm_final_status: `not_called`
- llm_final_decision: `drop`
- llm_final_confidence: ``

## Final Action
- final_decision: `drop`
- final_decision_source: `deterministic_gate`
- fallback_used: `True`

## Offline Evaluation
- gt_correct: `False`
- eval_label: `tn`
