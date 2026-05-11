# Code Usage

This folder includes a runnable, standalone copy of the controlled MLLM pseudo-label verification workflow.

## Install

```bash
python -m pip install -r requirements.txt
```

## Data

The package includes:

```text
data/pseudo_samples_256.json
```

This file contains the fixed 256-image pseudo-label metadata used for the demo. Full MLLM image-based execution requires the corresponding image files referenced by each sample. If those images are not present, deterministic/evaluation scripts that do not open images can still run, but multimodal calls will require restoring the image export paths or editing `raw_image_path` to local files.

If you want to run the full multimodal workflow from this standalone repo, copy the 256 raw images into:

```text
data/images/
```

The loader first checks `data/images/<filename>` before falling back to the original export path stored in the JSON metadata.

For source-domain exemplar bank construction, provide a VOC2012 root:

```text
VOC2012/
  Annotations/
  JPEGImages/
```

## Deterministic Smoke

This mode does not call the MLLM provider.

```bash
python -m mllm_pseudolabel_workflow.scripts.run_workflow_demo_256 \
  --mode deterministic_gate_256 \
  --limit 2 \
  --output_dir results/smoke_deterministic
```

## Full MLLM Workflow

Start an OpenAI-compatible multimodal endpoint first, then run:

```bash
python -m mllm_pseudolabel_workflow.scripts.run_workflow_demo_256 \
  --mode mllm_reference_workflow_256 \
  --engine langgraph \
  --source_voc_root /path/to/VOC2012 \
  --reference_bank_dir bank/reference_bank \
  --duplicate_iou_threshold 0.40 \
  --duplicate_class_agnostic \
  --provider_base_url http://mllm-provider.example/v1 \
  --model Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --output_dir results/reference_workflow_256
```

## Threshold Comparison Figure

```bash
python -m mllm_pseudolabel_workflow.scripts.plot_threshold_comparison \
  --result_dir artifacts/run_outputs \
  --output_dir figures
```

## Main Modules

| Path | Purpose |
|---|---|
| `mllm_pseudolabel_workflow/src/workflow.py` | Main workflow runner and MLLM calls. |
| `mllm_pseudolabel_workflow/src/langgraph_runner.py` | LangGraph DAG definition. |
| `mllm_pseudolabel_workflow/src/provider.py` | OpenAI-compatible provider client. |
| `mllm_pseudolabel_workflow/src/schema.py` | Strict JSON validation. |
| `mllm_pseudolabel_workflow/src/eval.py` | Offline Precision/Recall/F1/mAP50 evaluation. |
| `mllm_pseudolabel_workflow/tools/tools_v2.py` | Local optional evidence tools. |
| `mllm_pseudolabel_workflow/scripts/run_workflow_demo_256.py` | Main CLI. |
| `mllm_pseudolabel_workflow/scripts/plot_threshold_comparison.py` | Plot fixed-threshold comparison. |

## Safety Boundary

Target-domain ground truth is used only for offline evaluation after model calls finish. It must not be included in prompt payloads or model-consumed inputs.
