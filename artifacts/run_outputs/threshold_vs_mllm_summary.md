# Fixed Threshold vs MLLM Workflow

This comparison uses fixed teacher score thresholds from 0.1 to 0.9 and the latest MLLM agentic workflow run.

## MLLM Agentic Workflow

- mAP50: `0.144285`
- F1: `0.454802`
- Recall: `0.813131`
- Precision: `0.315686`
- TP/FP/FN: `161/349/37`

## Best Fixed Thresholds

- Best fixed-threshold F1: threshold `0.3`, F1 `0.291525`, mAP50 `0.074454`
- Best fixed-threshold mAP50: threshold `0.1`, mAP50 `0.095353`, F1 `0.200608`

## Artifacts

- `threshold_vs_mllm_metrics.png`
- `threshold_vs_mllm_metrics.svg`
- `threshold_vs_mllm_metrics.csv`
- `threshold_vs_mllm_metrics.json`
