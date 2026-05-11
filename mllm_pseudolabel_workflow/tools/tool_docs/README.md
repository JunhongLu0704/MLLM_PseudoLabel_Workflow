# MLLM Workflow V2 Tool Docs

These documents are LLM-facing descriptions for the bounded tool set. They specify when a tool may be used, what inputs are required, what outputs mean, and how invalid input or execution failure is reported.

Current scope:

- No true embedding RAG.
- A bounded optional-tool planner is enabled only after the system decides a proposal needs extra evidence. The planner may choose only `retrieve_same_class_image_crops` and/or `retrieve_class_exemplars`; invalid planner output falls back to the fixed evidence policy.
- No online run-level reflection.
- `resolve_duplicate_cluster_info` only reports which bboxes are clustered; it does not decide keep/drop.
- The fixed source exemplar bank lives under `projects/voc_clipart_mllm_demo_256/bank/reference_bank/` and is reusable across runs.
