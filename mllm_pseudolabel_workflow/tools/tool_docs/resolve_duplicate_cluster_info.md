# Tool: resolve_duplicate_cluster_info

Purpose: report which same-class proposal bboxes form high-IoU duplicate clusters.

This tool replaces the earlier `resolve_duplicates_mm` decision behavior for the current scope. It only provides cluster information and does not output keep/drop decisions.

Required input:

```json
{
  "image_id": "17",
  "iou_threshold": 0.5
}
```

The runtime supplies the current `ImageRecord`.

Output:

```json
{
  "tool_name": "resolve_duplicate_cluster_info",
  "status": "success",
  "payload": {
    "image_id": 17,
    "iou_threshold": 0.5,
    "duplicate_clusters": [
      {
        "duplicate_cluster_id": 0,
        "class_name": "person",
        "proposal_ids": [3, 6],
        "scores": {"3": 0.91, "6": 0.73},
        "bboxes": {"3": [1, 2, 3, 4]},
        "max_pairwise_iou": 0.71,
        "cluster_size": 2,
        "note": "cluster information only; this tool does not decide keep/drop"
      }
    ],
    "cluster_count": 1,
    "status": "success"
  }
}
```

No-cluster result:

- `status=success`, `cluster_count=0`, `duplicate_clusters=[]`.

Error behavior:

- Non-numeric threshold: `status=error`, `error_code=invalid_argument`.
- Threshold outside `(0, 1]`: `status=error`, `error_code=invalid_argument`.

Decision rule:

Use cluster information to decide whether additional visual comparison is needed. This tool must not directly drop proposals.

