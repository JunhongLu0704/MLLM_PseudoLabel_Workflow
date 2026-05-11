from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Proposal:
    image_id: int
    proposal_id: int
    label: int
    class_name: str
    score: float
    bbox: list[float]
    clamped_bbox: list[float]
    short_side: float
    area: float
    degenerate_bbox: bool


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    subset_order: int
    file_name: str
    image_path: Path
    overlay_image_path: Path
    width: int
    height: int
    proposals: list[Proposal]
    gt_boxes: list[list[float]]
    gt_labels: list[int]
    gt_label_names: list[str]


def load_records(path: Path, *, limit: int | None = None) -> list[ImageRecord]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if limit is not None:
        data = data[: max(int(limit), 0)]
    return [_record_from_sample(item) for item in data]


def _record_from_sample(sample: dict[str, Any]) -> ImageRecord:
    height, width = [int(v) for v in sample['size']]
    proposals = []
    for idx, bbox in enumerate(sample.get('pseudo_boxes_xyxy', [])):
        clamped = clamp_box(bbox, width, height)
        x1, y1, x2, y2 = clamped
        short_side = float(min(max(x2 - x1, 0.0), max(y2 - y1, 0.0)))
        area = float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
        proposals.append(
            Proposal(
                image_id=int(sample['image_id']),
                proposal_id=idx,
                label=int(sample['pseudo_labels'][idx]),
                class_name=str(sample['pseudo_label_names'][idx]),
                score=float(sample['pseudo_scores'][idx]),
                bbox=[float(v) for v in bbox],
                clamped_bbox=clamped,
                short_side=short_side,
                area=area,
                degenerate_bbox=short_side <= 16.0,
            )
        )
    return ImageRecord(
        image_id=int(sample['image_id']),
        subset_order=int(sample.get('subset_order', sample['image_id'])),
        file_name=str(sample.get('file_name', '')),
        image_path=_resolve_path(str(sample['raw_image_path'])),
        overlay_image_path=_resolve_path(str(sample.get('overlay_image_path', sample['raw_image_path']))),
        width=width,
        height=height,
        proposals=proposals,
        gt_boxes=[[float(v) for v in box] for box in sample.get('gt_boxes_xyxy', [])],
        gt_labels=[int(v) for v in sample.get('gt_labels', [])],
        gt_label_names=[str(v) for v in sample.get('gt_label_names', [])],
    )


def clamp_box(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1 = max(0.0, min(round(x1), width - 1))
    y1 = max(0.0, min(round(y1), height - 1))
    x2 = max(x1 + 1.0, min(round(x2), width))
    y2 = max(y1 + 1.0, min(round(y2), height))
    return [float(x1), float(y1), float(x2), float(y2)]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local_data_image = ROOT / 'data' / 'images' / path.name
    if local_data_image.exists():
        return local_data_image
    return ROOT / path
