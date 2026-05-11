from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import REFERENCE_OLD_256
from .data import ImageRecord, Proposal

OLD_REFERENCE_CSV = Path('projects/voc_clipart_review_v2/results/verify_gate_strict_invalid_drop_256/proposal_decision_table.csv')


def evaluate_rows(records: list[ImageRecord], rows: list[dict[str, Any]], *, decision_field: str) -> dict[str, Any]:
    gt_by_image = {record.image_id: record for record in records}
    tp = fp = tn = fn = 0
    for row in rows:
        if row.get('degenerate_bbox'):
            continue
        is_gt = bool(row['gt_correct'])
        keep = row[decision_field] == 'keep'
        if keep and is_gt:
            tp += 1
        elif keep and not is_gt:
            fp += 1
        elif not keep and is_gt:
            fn += 1
        else:
            tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
        'filtered_map50': filtered_map50(gt_by_image, rows, decision_field=decision_field),
    }


def attach_gt_labels(records: list[ImageRecord], rows: list[dict[str, Any]]) -> None:
    by_image = {record.image_id: record for record in records}
    for row in rows:
        record = by_image[int(row['image_id'])]
        proposal = row['_proposal']
        max_iou = 0.0
        matched_gt_id = -1
        for idx, (gt_box, gt_label) in enumerate(zip(record.gt_boxes, record.gt_labels)):
            if int(gt_label) != int(proposal.label):
                continue
            iou = bbox_iou(proposal.bbox, gt_box)
            if iou > max_iou:
                max_iou = iou
                matched_gt_id = idx
        row['max_iou_same_class'] = round(max_iou, 6)
        row['matched_gt_id'] = matched_gt_id
        row['gt_correct'] = max_iou >= 0.5
        row['tp_fp_fn_label_for_eval'] = 'tp' if row['gt_correct'] and row['final_decision'] == 'keep' else 'fp' if row['final_decision'] == 'keep' else 'fn' if row['gt_correct'] else 'tn'


def filtered_map50(records_by_image: dict[int, ImageRecord], rows: list[dict[str, Any]], *, decision_field: str) -> dict[str, Any]:
    gt_by_class: dict[int, list[tuple[int, int, list[float]]]] = defaultdict(list)
    pred_by_class: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    for record in records_by_image.values():
        for idx, (box, label) in enumerate(zip(record.gt_boxes, record.gt_labels)):
            gt_by_class[int(label)].append((record.image_id, idx, box))
    for row in rows:
        if row.get('degenerate_bbox') or row[decision_field] != 'keep':
            continue
        proposal: Proposal = row['_proposal']
        pred_by_class[int(proposal.label)].append((proposal.image_id, float(proposal.score), proposal.bbox))
    ap_values = []
    per_class = []
    for label, gts in sorted(gt_by_class.items()):
        preds = sorted(pred_by_class.get(label, []), key=lambda item: -item[1])
        matched: set[tuple[int, int]] = set()
        tps = []
        fps = []
        for image_id, _score, box in preds:
            best_iou = 0.0
            best_key = None
            for gt_image_id, gt_idx, gt_box in gts:
                if gt_image_id != image_id or (gt_image_id, gt_idx) in matched:
                    continue
                iou = bbox_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_key = (gt_image_id, gt_idx)
            if best_iou >= 0.5 and best_key is not None:
                matched.add(best_key)
                tps.append(1.0)
                fps.append(0.0)
            else:
                tps.append(0.0)
                fps.append(1.0)
        ap = average_precision(tps, fps, len(gts))
        ap_values.append(ap)
        per_class.append({'class_id': label, 'num_gt': len(gts), 'num_pred': len(preds), 'ap50': round(ap, 6)})
    return {'map50': round(sum(ap_values) / max(len(ap_values), 1), 6), 'per_class': per_class}


def average_precision(tps: list[float], fps: list[float], num_gt: int) -> float:
    if num_gt <= 0:
        return 0.0
    cum_tp = 0.0
    cum_fp = 0.0
    precisions = []
    recalls = []
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / max(cum_tp + cum_fp, 1e-12))
        recalls.append(cum_tp / num_gt)
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * max(recall - prev_recall, 0.0)
        prev_recall = recall
    return ap


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def write_metric_diff_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ref = REFERENCE_OLD_256
    lines = ['# Metric Diff Report', '']
    lines.append('| item | old | new | delta |')
    lines.append('|---|---:|---:|---:|')
    checks = [
        ('num_total_proposals', ref['num_total_proposals'], summary['num_total_proposals']),
        ('num_valid_proposals', ref['num_valid_proposals'], summary['num_valid_proposals']),
    ]
    for key in ('tp', 'fp', 'fn'):
        checks.append((f"score_baseline_020.{key}", ref['score_baseline_020'][key], summary['metrics']['score_baseline_020'][key]))
        checks.append((f"deterministic_gate.{key}", ref['deterministic_gate'][key], summary['metrics']['deterministic_gate_256'][key]))
    for name, old, new in checks:
        lines.append(f'| `{name}` | {old} | {new} | {new - old} |')
    lines.extend(['', '## Proposal-Level Decision Differences', ''])
    diff_rows = [row for row in rows if row.get('score_baseline_020') != row.get('final_decision')]
    lines.append(f'- proposals where deterministic gate differs from score baseline: `{len(diff_rows)}`')
    lines.append('- If aggregate metrics differ, inspect `proposal_results.csv` columns `gt_correct`, `max_iou_same_class`, `degenerate_bbox`, and `final_decision`.')
    _append_old_csv_diff(lines, rows)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _append_old_csv_diff(lines: list[str], rows: list[dict[str, Any]]) -> None:
    if not OLD_REFERENCE_CSV.exists():
        lines.append('- old reference CSV not found; proposal-level old/new comparison skipped.')
        return
    import csv

    old_rows = list(csv.DictReader(OLD_REFERENCE_CSV.open('r', newline='', encoding='utf-8')))
    old_by_key = {(str(row['image_id']), str(row['proposal_id'])): row for row in old_rows}
    new_by_key = {(str(row['image_id']), str(row['proposal_id'])): row for row in rows}
    missing_old = sorted(set(new_by_key) - set(old_by_key), key=lambda item: (int(item[0]), int(item[1])))
    missing_new = sorted(set(old_by_key) - set(new_by_key), key=lambda item: (int(item[0]), int(item[1])))
    common = sorted(set(old_by_key) & set(new_by_key), key=lambda item: (int(item[0]), int(item[1])))
    gt_diff = [
        key
        for key in common
        if (old_by_key[key].get('gt_correct_iou50') == 'True') != bool(new_by_key[key].get('gt_correct'))
    ]
    gate_diff = [
        key
        for key in common
        if old_by_key[key].get('verify_gate_strict_invalid_drop_auto_decision') != new_by_key[key].get('deterministic_gate_256')
    ]
    lines.extend(
        [
            '',
            '## Old CSV Proposal-Level Comparison',
            '',
            f'- old reference CSV: `{OLD_REFERENCE_CSV.as_posix()}`',
            f'- common proposals: `{len(common)}`',
            f'- proposals only in new CSV: `{len(missing_old)}`',
            f'- proposals only in old CSV: `{len(missing_new)}`',
            f'- GT correctness mismatches: `{len(gt_diff)}`',
            f'- deterministic gate decision mismatches: `{len(gate_diff)}`',
            '',
            '### First Gate Decision Mismatches',
            '',
        ]
    )
    if not gate_diff:
        lines.append('- none')
        return
    lines.append('| image_id | proposal_id | old_gate | new_gate | old_validity | new_validity | score |')
    lines.append('|---:|---:|---|---|---|---|---:|')
    for key in gate_diff[:25]:
        old = old_by_key[key]
        new = new_by_key[key]
        lines.append(
            '| '
            f"{key[0]} | {key[1]} | "
            f"`{old.get('verify_gate_strict_invalid_drop_auto_decision', '')}` | "
            f"`{new.get('deterministic_gate_256', '')}` | "
            f"`{old.get('verify_box_validity', '')}` | "
            f"`{new.get('visual_validity', '')}` | "
            f"{float(new.get('score', 0.0)):.4f} |"
        )
