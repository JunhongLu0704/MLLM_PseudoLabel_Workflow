# Package Manifest

## Project

面向视觉训练数据质量控制的受控多模态伪标签验证工作流

English package name:

```text
Controlled_MLLM_PseudoLabel_Verification_Workflow
```

## Purpose

This folder is a final showcase package for resume attachments, interview demonstrations, GitHub README/docs, and portfolio submission. It contains presentation documents, figures, example traces, and minimal run artifacts.

## Package Contents

| Path | Description |
|---|---|
| `README.md` | Package entry point for quick review. |
| `docs/MLLM_PseudoLabel_Workflow_Demo.md` | Complete technical demo document. |
| `docs/MLLM_PseudoLabel_Workflow_5page.pdf` | PDF presentation copy. |
| `docs/MLLM_PseudoLabel_Workflow_5page.docx` | Editable Word summary document generated for this package. |
| `figures/project_poster.png` | Project overview poster. |
| `figures/threshold_vs_mllm_metrics.png` | Fixed-threshold scan comparison figure. |
| `examples/success_drop_fp.md` | Example where the workflow removes a false positive. |
| `examples/success_keep_tp.md` | Example where the workflow keeps a true positive. |
| `examples/fallback_case.md` | Example fallback trace. |
| `artifacts/run_outputs/summary_metrics.json` | Main metric summary from the 256-image run. |
| `artifacts/run_outputs/tool_call_summary.json` | Tool-call count summary. |
| `artifacts/run_outputs/token_usage_summary.json` | Token and latency accounting. |
| `artifacts/run_outputs/metric_diff_report.md` | Metric and reference comparison report. |
| `artifacts/run_outputs/proposal_results.csv` | Proposal-level final decision table. |
| `artifacts/run_outputs/image_0000_trace_sample.json` | One compact image-level trace sample. |
| `mllm_pseudolabel_workflow/` | Standalone runnable Python workflow package. |
| `CODE_USAGE.md` | Code installation and execution instructions. |
| `requirements.txt` | Minimal Python runtime dependencies. |
| `pyproject.toml` | Python package metadata. |

The full `tool_call_trace.jsonl` is intentionally not copied into this package because it is large and not needed for quick review. The package keeps summary artifacts and one trace sample instead.

## Main Result

score >= 0.20 baseline vs MLLM workflow on VOC -> Clipart fixed 256 images:

- F1: 0.2815 -> 0.4548
- mAP50: 0.0850 -> 0.1443
- Recall: 0.6263 -> 0.8131
- Precision: 0.1816 -> 0.3157
- FP: 559 -> 349

## Dataset and Run Setup

| Item | Value |
|---|---:|
| Images | 256 |
| Classes | 20 |
| Total candidate proposals | 1921 |
| Valid proposals | 1776 |
| Valid target objects | 198 |
| Baseline | teacher score >= 0.20 |

## Notes

- Target-domain GT is used only for offline evaluation.
- The workflow is fixed and controlled; LLM decisions occur only at predefined nodes.
- The system should be described as quality verification / pseudo-label verification, not content moderation.
- The current demo has not been connected to a real training feedback loop.
- The package is for demo / portfolio / interview presentation.

## Source Run

The main artifacts come from:

```text
projects/voc_clipart_mllm_demo_256/results/reference_workflow_langgraph_256_agentic_tools_iou040
```

The threshold comparison figure comes from:

```text
projects/voc_clipart_mllm_demo_256/showcase/threshold_comparison
```
