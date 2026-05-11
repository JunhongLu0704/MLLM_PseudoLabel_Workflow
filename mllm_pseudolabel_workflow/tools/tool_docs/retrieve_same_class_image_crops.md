# Tool: retrieve_same_class_image_crops

Purpose: retrieve other same-class proposal crops from the same target image for local visual comparison.

Required input:

```json
{
  "image_id": "17",
  "query_proposal_id": "4",
  "top_k": 4
}
```

The runtime supplies the current `ImageRecord`.

Output:

```json
{
  "tool_name": "retrieve_same_class_image_crops",
  "status": "success",
  "payload": {
    "query_proposal_id": "4",
    "class_name": "dog",
    "retrieved_crops": [
      {
        "proposal_id": "2",
        "score": 0.88,
        "bbox": [10, 20, 100, 140],
        "crop_path": ".../proposal_0002.png",
        "reason": "same_class_reference_high_score"
      }
    ],
    "status": "success"
  }
}
```

Empty result:

- If no other same-class valid proposal exists: `status=empty`.

Error behavior:

- Missing query proposal: `status=error`, `error_code=proposal_not_found`.
- Non-VOC proposal class: `status=error`, `error_code=invalid_class`.
- Bad `top_k`: `status=error`, `error_code=invalid_argument`.

Decision rule:

Same-image crops are supporting reference evidence only. They do not directly decide keep/drop.

