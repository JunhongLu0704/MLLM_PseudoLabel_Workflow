from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..src.constants import VOC_CLASSES
from ..src.data import ImageRecord, Proposal


@dataclass
class ToolResponse:
    tool_name: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str = ''
    message: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'status': self.status,
            'payload': self.payload,
            'error_code': self.error_code,
            'message': self.message,
        }


def tool_error(tool_name: str, error_code: str, message: str, *, payload: dict[str, Any] | None = None) -> ToolResponse:
    return ToolResponse(tool_name=tool_name, status='error', payload=payload or {}, error_code=error_code, message=message)


def build_class_exemplar_bank(
    *,
    source_voc_root: Path,
    output_dir: Path,
    max_per_class: int = 2,
    min_short_side: int = 32,
) -> ToolResponse:
    tool_name = 'build_class_exemplar_bank'
    max_per_class = max(int(max_per_class), 0)
    min_short_side = max(int(min_short_side), 1)
    if max_per_class <= 0:
        return tool_error(tool_name, 'invalid_argument', 'max_per_class must be positive')

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'class_exemplar_manifest.json'
    summary_path = output_dir / 'class_exemplar_summary.json'
    if manifest_path.exists() and summary_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if (
                manifest.get('bank_version') == 'v2'
                and manifest.get('source') == 'source_domain_gt'
                and int(manifest.get('max_per_class', 0)) == max_per_class
                and int(manifest.get('min_short_side', 0)) == min_short_side
                and all(int(count) >= max_per_class for count in summary.get('counts', {}).values())
            ):
                return ToolResponse(
                    tool_name=tool_name,
                    status='success',
                    payload={
                        'manifest_path': manifest_path.as_posix(),
                        'summary_path': summary_path.as_posix(),
                        'counts': dict(summary.get('counts', {})),
                        'total_exemplars': int(summary.get('total_exemplars', 0)),
                        'reused_existing_bank': True,
                    },
                    message='existing source-domain class exemplar bank reused',
                )
        except json.JSONDecodeError:
            pass

    if not source_voc_root.exists():
        return tool_error(tool_name, 'source_root_missing', f'source_voc_root does not exist: {source_voc_root}')
    annotations_dir = source_voc_root / 'Annotations'
    images_dir = source_voc_root / 'JPEGImages'
    if not annotations_dir.exists() or not images_dir.exists():
        return tool_error(tool_name, 'voc_layout_invalid', 'expected Annotations/ and JPEGImages/ under source_voc_root')

    bank_root = output_dir / 'class_exemplars'
    bank_root.mkdir(parents=True, exist_ok=True)
    counts = {class_name: 0 for class_name in VOC_CLASSES}
    classes: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in VOC_CLASSES}
    used_images_by_class: dict[str, set[str]] = {class_name: set() for class_name in VOC_CLASSES}
    annotation_paths = sorted(annotations_dir.glob('*.xml'))

    for xml_path in annotation_paths:
        if all(count >= max_per_class for count in counts.values()):
            break
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        filename = root.findtext('filename') or f'{xml_path.stem}.jpg'
        image_path = images_dir / filename
        if not image_path.exists():
            image_path = images_dir / f'{xml_path.stem}.jpg'
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            rgb = image.convert('RGB')
            for obj in root.findall('object'):
                class_name = (obj.findtext('name') or '').strip()
                if class_name not in VOC_CLASSES or counts[class_name] >= max_per_class:
                    continue
                box_node = obj.find('bndbox')
                if box_node is None:
                    continue
                bbox = _xml_bbox(box_node)
                if bbox is None:
                    continue
                if xml_path.stem in used_images_by_class[class_name]:
                    continue
                x1, y1, x2, y2 = _clamp_int_box(bbox, rgb.width, rgb.height)
                short_side = min(max(x2 - x1, 0), max(y2 - y1, 0))
                if short_side < min_short_side:
                    continue
                class_dir = bank_root / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f'{class_name}_{counts[class_name] + 1:04d}.png'
                crop_path = class_dir / crop_name
                rgb.crop((x1, y1, x2, y2)).save(crop_path)
                item = {
                    'class_name': class_name,
                    'crop_path': crop_path.relative_to(output_dir).as_posix(),
                    'source': 'source_domain_gt',
                    'source_image_id': xml_path.stem,
                    'bbox': [x1, y1, x2, y2],
                    'selection_reason': 'source_domain_gt_representative',
                    'gt_safety_note': 'source-domain exemplar only; not target GT',
                }
                classes[class_name].append(item)
                counts[class_name] += 1
                used_images_by_class[class_name].add(xml_path.stem)
                if counts[class_name] >= max_per_class:
                    continue

    manifest = {
        'bank_version': 'v2',
        'source': 'source_domain_gt',
        'source_voc_root': source_voc_root.as_posix(),
        'max_per_class': max_per_class,
        'min_short_side': min_short_side,
        'classes': classes,
    }
    _write_json(manifest_path, manifest)
    _write_json(
        summary_path,
        {
            'counts': counts,
            'total_exemplars': sum(counts.values()),
            'missing_classes': [k for k, v in counts.items() if v == 0],
            'source_unique_image_counts': {class_name: len(used_images_by_class[class_name]) for class_name in VOC_CLASSES},
            'reused_existing_bank': False,
        },
    )
    return ToolResponse(
        tool_name=tool_name,
        status='success',
        payload={
            'manifest_path': manifest_path.as_posix(),
            'summary_path': summary_path.as_posix(),
            'counts': counts,
            'total_exemplars': sum(counts.values()),
            'reused_existing_bank': False,
        },
        message='source-domain class exemplar bank built',
    )


def retrieve_class_exemplars(*, manifest_path: Path, class_name: str, top_k: int = 3) -> ToolResponse:
    tool_name = 'retrieve_class_exemplars'
    if class_name not in VOC_CLASSES:
        return tool_error(tool_name, 'invalid_class', f'class_name must be a VOC class: {class_name}')
    if not manifest_path.exists():
        return tool_error(tool_name, 'manifest_missing', f'manifest_path does not exist: {manifest_path}')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        return tool_error(tool_name, 'manifest_invalid_json', f'failed to parse manifest JSON: {exc}')
    items = manifest.get('classes', {}).get(class_name, [])
    if not isinstance(items, list):
        return tool_error(tool_name, 'manifest_schema_invalid', 'manifest classes[class_name] must be a list')
    top_k = max(int(top_k), 0)
    if top_k <= 0:
        return tool_error(tool_name, 'invalid_argument', 'top_k must be positive')
    selected = [dict(item) for item in items[:top_k] if isinstance(item, dict)]
    return ToolResponse(
        tool_name=tool_name,
        status='success' if selected else 'empty',
        payload={
            'class_name': class_name,
            'retrieved_exemplars': selected,
            'status': 'success' if selected else 'empty',
            'note': 'reference crops are appearance aids only and must not directly decide target keep/drop',
        },
        message='class exemplars retrieved' if selected else 'no exemplar available for class',
    )


def retrieve_same_class_image_crops(
    *,
    record: ImageRecord,
    query_proposal_id: int,
    output_dir: Path,
    top_k: int = 4,
    verification_results: dict[int, dict[str, Any]] | None = None,
) -> ToolResponse:
    tool_name = 'retrieve_same_class_image_crops'
    proposal = _proposal_by_id(record, query_proposal_id)
    if proposal is None:
        return tool_error(tool_name, 'proposal_not_found', f'query_proposal_id not found: {query_proposal_id}')
    if proposal.class_name not in VOC_CLASSES:
        return tool_error(tool_name, 'invalid_class', f'proposal class is not VOC: {proposal.class_name}')
    top_k = max(int(top_k), 0)
    if top_k <= 0:
        return tool_error(tool_name, 'invalid_argument', 'top_k must be positive')
    verification_results = verification_results or {}
    candidates = [
        item
        for item in record.proposals
        if item.proposal_id != proposal.proposal_id and item.class_name == proposal.class_name and not item.degenerate_bbox
    ]
    candidates.sort(
        key=lambda item: (
            _verification_rank(verification_results.get(item.proposal_id, {})),
            item.score,
            item.short_side,
        ),
        reverse=True,
    )
    selected = candidates[:top_k]
    crop_dir = output_dir / f'image_{record.image_id:04d}' / f'query_{proposal.proposal_id:04d}' / proposal.class_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Image.open(record.image_path) as image:
        rgb = image.convert('RGB')
        for item in selected:
            crop_path = crop_dir / f'proposal_{item.proposal_id:04d}.png'
            rgb.crop(tuple(int(v) for v in item.clamped_bbox)).save(crop_path)
            rows.append(
                {
                    'proposal_id': str(item.proposal_id),
                    'score': round(item.score, 6),
                    'bbox': item.clamped_bbox,
                    'crop_path': crop_path.as_posix(),
                    'reason': 'same_class_reference_high_score',
                    'verification_hint': verification_results.get(item.proposal_id, {}).get('visual_validity', ''),
                }
            )
    return ToolResponse(
        tool_name=tool_name,
        status='success' if rows else 'empty',
        payload={
            'query_proposal_id': str(proposal.proposal_id),
            'class_name': proposal.class_name,
            'retrieved_crops': rows,
            'status': 'success' if rows else 'empty',
        },
        message='same-class image crops retrieved' if rows else 'no same-class reference proposals available',
    )


def resolve_duplicate_cluster_info(*, record: ImageRecord, iou_threshold: float = 0.5, class_agnostic: bool = False) -> ToolResponse:
    tool_name = 'resolve_duplicate_cluster_info'
    try:
        iou_threshold = float(iou_threshold)
    except (TypeError, ValueError):
        return tool_error(tool_name, 'invalid_argument', 'iou_threshold must be numeric')
    if not (0.0 < iou_threshold <= 1.0):
        return tool_error(tool_name, 'invalid_argument', 'iou_threshold must be in (0, 1]')
    valid = [proposal for proposal in record.proposals if not proposal.degenerate_bbox]
    parent = {proposal.proposal_id: proposal.proposal_id for proposal in valid}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_ious: dict[tuple[int, int], float] = {}
    class_agnostic = bool(class_agnostic)
    for idx, left in enumerate(valid):
        for right in valid[idx + 1 :]:
            if not class_agnostic and left.class_name != right.class_name:
                continue
            iou = bbox_iou(left.clamped_bbox, right.clamped_bbox)
            if iou >= iou_threshold:
                union(left.proposal_id, right.proposal_id)
                pair_ious[(left.proposal_id, right.proposal_id)] = iou

    grouped: dict[int, list[Proposal]] = {}
    for proposal in valid:
        grouped.setdefault(find(proposal.proposal_id), []).append(proposal)
    clusters = []
    cluster_id = 0
    for proposals in grouped.values():
        if len(proposals) < 2:
            continue
        proposals = sorted(proposals, key=lambda item: item.proposal_id)
        max_iou = 0.0
        for idx, left in enumerate(proposals):
            for right in proposals[idx + 1 :]:
                max_iou = max(max_iou, bbox_iou(left.clamped_bbox, right.clamped_bbox))
        clusters.append(
            {
                'duplicate_cluster_id': cluster_id,
                'class_name': proposals[0].class_name if len({item.class_name for item in proposals}) == 1 else 'mixed',
                'class_names': sorted({item.class_name for item in proposals}),
                'proposal_ids': [item.proposal_id for item in proposals],
                'scores': {str(item.proposal_id): round(item.score, 6) for item in proposals},
                'bboxes': {str(item.proposal_id): item.clamped_bbox for item in proposals},
                'max_pairwise_iou': round(max_iou, 6),
                'cluster_size': len(proposals),
                'note': 'cluster information only; this tool does not decide keep/drop',
            }
        )
        cluster_id += 1
    return ToolResponse(
        tool_name=tool_name,
        status='success',
        payload={
            'image_id': record.image_id,
            'iou_threshold': iou_threshold,
            'class_agnostic': class_agnostic,
            'duplicate_clusters': clusters,
            'cluster_count': len(clusters),
            'status': 'success',
        },
        message='duplicate cluster information generated',
    )


def build_candidate_pool(
    *,
    record: ImageRecord,
    duplicate_cluster_info: dict[str, Any] | None = None,
    score_low: float = 0.10,
    score_high: float = 0.30,
    top_n_sanity: int = 2,
    max_candidates: int = 32,
) -> ToolResponse:
    tool_name = 'build_candidate_pool'
    try:
        score_low = float(score_low)
        score_high = float(score_high)
        top_n_sanity = int(top_n_sanity)
        max_candidates = int(max_candidates)
    except (TypeError, ValueError):
        return tool_error(tool_name, 'invalid_argument', 'score thresholds and limits must be numeric')
    if score_low < 0 or score_high <= score_low or max_candidates <= 0:
        return tool_error(tool_name, 'invalid_argument', 'invalid score range or max_candidates')
    reasons: dict[int, list[str]] = {}

    def add(proposal_id: int, reason: str) -> None:
        reasons.setdefault(proposal_id, [])
        if reason not in reasons[proposal_id]:
            reasons[proposal_id].append(reason)

    for proposal in record.proposals:
        if proposal.degenerate_bbox:
            continue
        if score_low <= proposal.score < score_high:
            add(proposal.proposal_id, 'uncertain_score_band')
        if _near_border(proposal, record):
            add(proposal.proposal_id, 'border_touching')
        if proposal.short_side < 32:
            add(proposal.proposal_id, 'small_box')

    for cluster in (duplicate_cluster_info or {}).get('duplicate_clusters', []):
        for proposal_id in cluster.get('proposal_ids', []):
            add(int(proposal_id), 'duplicate_cluster')

    for proposal in sorted([p for p in record.proposals if not p.degenerate_bbox], key=lambda p: p.score, reverse=True)[: max(top_n_sanity, 0)]:
        add(proposal.proposal_id, 'top_score_sanity_check')

    ranked = sorted(
        reasons,
        key=lambda pid: (
            -_reason_priority(reasons[pid]),
            -_proposal_by_id(record, pid).score if _proposal_by_id(record, pid) else 0.0,
            pid,
        ),
    )[:max_candidates]
    return ToolResponse(
        tool_name=tool_name,
        status='success',
        payload={
            'image_id': record.image_id,
            'candidate_proposal_ids': ranked,
            'candidate_reasons': {str(pid): reasons[pid] for pid in ranked},
            'pool_policy': 'uncertain_plus_structural_risk_v1',
            'max_candidates': max_candidates,
            'status': 'success',
        },
        message='candidate pool built',
    )


def bbox_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    larea = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    rarea = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    denom = larea + rarea - inter
    return inter / denom if denom > 0 else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _xml_bbox(box_node: ET.Element) -> list[int] | None:
    try:
        return [
            int(float(box_node.findtext('xmin') or 0)),
            int(float(box_node.findtext('ymin') or 0)),
            int(float(box_node.findtext('xmax') or 0)),
            int(float(box_node.findtext('ymax') or 0)),
        ]
    except ValueError:
        return None


def _clamp_int_box(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _proposal_by_id(record: ImageRecord, proposal_id: int) -> Proposal | None:
    for proposal in record.proposals:
        if proposal.proposal_id == int(proposal_id):
            return proposal
    return None


def _verification_rank(value: dict[str, Any]) -> int:
    validity = str(value.get('visual_validity', ''))
    if validity == 'valid':
        return 3
    if validity == 'uncertain':
        return 2
    if validity == 'invalid':
        return 1
    return 0


def _near_border(proposal: Proposal, record: ImageRecord) -> bool:
    x1, y1, x2, y2 = proposal.clamped_bbox
    return x1 <= record.width * 0.03 or y1 <= record.height * 0.03 or x2 >= record.width * 0.97 or y2 >= record.height * 0.97


def _reason_priority(values: list[str]) -> int:
    priority = {
        'duplicate_cluster': 5,
        'uncertain_score_band': 4,
        'small_box': 3,
        'border_touching': 2,
        'top_score_sanity_check': 1,
    }
    return max((priority.get(value, 0) for value in values), default=0)
