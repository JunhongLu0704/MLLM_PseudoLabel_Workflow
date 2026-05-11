# Tool: build_class_exemplar_bank

Purpose: build a small source-domain class reference crop bank from `/path/to/VOC2012`.

Required input:

```json
{
  "source_voc_root": "/path/to/VOC2012",
  "output_dir": "projects/voc_clipart_mllm_demo_256/bank/reference_bank",
  "max_per_class": 10,
  "min_short_side": 32
}
```

Output:

```json
{
  "tool_name": "build_class_exemplar_bank",
  "status": "success",
  "payload": {
    "manifest_path": ".../class_exemplar_manifest.json",
    "summary_path": ".../class_exemplar_summary.json",
    "counts": {"dog": 2},
    "total_exemplars": 40
  },
  "error_code": "",
  "message": "source-domain class exemplar bank built"
}
```

Error behavior:

- Missing VOC root: `status=error`, `error_code=source_root_missing`.
- Invalid VOC layout: `status=error`, `error_code=voc_layout_invalid`.
- Invalid `max_per_class`: `status=error`, `error_code=invalid_argument`.
- If a compatible bank already exists at the fixed output path, the tool reuses it and returns `reused_existing_bank=true`.

Safety:

- Source-domain GT crops are appearance references only.
- These crops must not be treated as target-domain correctness evidence.
- Each class should come from distinct source images within the same bank build.
