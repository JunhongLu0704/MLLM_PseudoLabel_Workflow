# Tool: retrieve_class_exemplars

Purpose: retrieve fixed source-domain class exemplar crops for a VOC class.

Required input:

```json
{
  "manifest_path": "projects/voc_clipart_mllm_demo_256/bank/reference_bank/class_exemplar_manifest.json",
  "class_name": "dog",
  "top_k": 3
}
```

Output:

```json
{
  "tool_name": "retrieve_class_exemplars",
  "status": "success",
  "payload": {
    "class_name": "dog",
    "retrieved_exemplars": [
      {
        "crop_path": "class_exemplars/dog/dog_0001.png",
        "source": "source_domain_gt",
        "gt_safety_note": "source-domain exemplar only; not target GT"
      }
    ],
    "status": "success",
    "note": "reference crops are appearance aids only and must not directly decide target keep/drop"
  }
}
```

Empty result:

- If the class is valid but no exemplar exists: `status=empty`.

Error behavior:

- Non-VOC class: `status=error`, `error_code=invalid_class`.
- Missing manifest: `status=error`, `error_code=manifest_missing`.
- Invalid JSON manifest: `status=error`, `error_code=manifest_invalid_json`.
- Bad `top_k`: `status=error`, `error_code=invalid_argument`.

Decision rule:

Use retrieved exemplars only as visual appearance reference. They cannot override strong `VerifySingleBox_MM` evidence.
