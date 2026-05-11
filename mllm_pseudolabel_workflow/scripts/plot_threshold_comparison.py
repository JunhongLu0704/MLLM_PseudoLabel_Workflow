from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..src.data import load_records
from ..src.eval import attach_gt_labels, evaluate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot fixed score thresholds vs MLLM workflow metrics.')
    parser.add_argument('--data_path', type=Path, default=Path('projects/voc_clipart_review_v2/exports/clipart_train_pseudo_256/pseudo_samples.json'))
    parser.add_argument('--result_dir', type=Path, default=Path('projects/voc_clipart_mllm_demo_256/results/reference_workflow_langgraph_256_agentic_tools_iou040'))
    parser.add_argument('--output_dir', type=Path, default=Path('projects/voc_clipart_mllm_demo_256/showcase/threshold_comparison'))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data_path)
    proposal_rows = list(csv.DictReader((args.result_dir / 'proposal_results.csv').open('r', newline='', encoding='utf-8')))
    final_by_key = {(int(row['image_id']), int(row['proposal_id'])): row['final_decision'] for row in proposal_rows}
    rows = make_eval_rows(records, final_by_key)
    attach_gt_labels(records, rows)

    metric_rows: list[dict[str, Any]] = []
    thresholds = [round(idx / 10, 1) for idx in range(1, 10)]
    for threshold in thresholds:
        field = f'threshold_{threshold:.1f}'
        for row in rows:
            row[field] = 'keep' if float(row['_proposal'].score) >= threshold else 'drop'
        metrics = evaluate_rows(records, rows, decision_field=field)
        metric_rows.append(flatten_metrics(f'fixed_threshold_{threshold:.1f}', threshold, metrics))

    for row in rows:
        row['mllm_workflow'] = final_by_key.get((int(row['image_id']), int(row['proposal_id'])), 'drop')
    method_metrics = evaluate_rows(records, rows, decision_field='mllm_workflow')
    method_row = flatten_metrics('mllm_agentic_workflow', None, method_metrics)

    write_csv(args.output_dir / 'threshold_vs_mllm_metrics.csv', metric_rows + [method_row])
    write_json(args.output_dir / 'threshold_vs_mllm_metrics.json', {'fixed_thresholds': metric_rows, 'mllm_agentic_workflow': method_row})
    plot_metrics(args.output_dir, metric_rows, method_row)
    write_report(args.output_dir / 'threshold_vs_mllm_summary.md', metric_rows, method_row)
    print(json.dumps({'output_dir': args.output_dir.as_posix(), 'mllm_agentic_workflow': method_row}, ensure_ascii=False, indent=2))


def make_eval_rows(records, final_by_key: dict[tuple[int, int], str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for proposal in record.proposals:
            rows.append(
                {
                    'image_id': record.image_id,
                    'proposal_id': proposal.proposal_id,
                    'degenerate_bbox': proposal.degenerate_bbox,
                    '_proposal': proposal,
                    'final_decision': final_by_key.get((record.image_id, proposal.proposal_id), 'drop'),
                }
            )
    return rows


def flatten_metrics(name: str, threshold: float | None, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': name,
        'threshold': '' if threshold is None else threshold,
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'tn': metrics['tn'],
        'fn': metrics['fn'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'map50': metrics['filtered_map50']['map50'],
    }


def plot_metrics(output_dir: Path, metric_rows: list[dict[str, Any]], method_row: dict[str, Any]) -> None:
    thresholds = [float(row['threshold']) for row in metric_rows]
    specs = [
        ('map50', 'mAP50'),
        ('f1', 'F1'),
        ('recall', 'Recall'),
        ('precision', 'Precision'),
    ]
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.25,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    fig.suptitle('Fixed Score Thresholds vs MLLM Agentic Workflow on VOC Clipart 256', fontsize=14, fontweight='bold')
    for ax, (key, label) in zip(axes.ravel(), specs):
        values = [float(row[key]) for row in metric_rows]
        method_value = float(method_row[key])
        ax.plot(thresholds, values, marker='o', linewidth=2.0, color='#2f6f9f', label='fixed score threshold')
        ax.axhline(method_value, color='#c23b22', linestyle='--', linewidth=2.0, label='MLLM workflow')
        ax.scatter([thresholds[-1] + 0.035], [method_value], marker='*', s=140, color='#c23b22', zorder=4)
        ax.text(thresholds[-1] + 0.045, method_value, f'{method_value:.3f}', va='center', fontsize=9, color='#8a2417')
        ax.set_title(label)
        ax.set_xlabel('Fixed teacher score threshold')
        ax.set_ylabel(label)
        ax.set_xlim(0.08, 0.98)
        ymin = min(min(values), method_value)
        ymax = max(max(values), method_value)
        pad = max((ymax - ymin) * 0.18, 0.02)
        ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / 'threshold_vs_mllm_metrics.png')
    fig.savefig(output_dir / 'threshold_vs_mllm_metrics.svg')
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_report(path: Path, metric_rows: list[dict[str, Any]], method_row: dict[str, Any]) -> None:
    best_f1 = max(metric_rows, key=lambda row: float(row['f1']))
    best_map = max(metric_rows, key=lambda row: float(row['map50']))
    lines = [
        '# Fixed Threshold vs MLLM Workflow',
        '',
        'This comparison uses fixed teacher score thresholds from 0.1 to 0.9 and the latest MLLM agentic workflow run.',
        '',
        '## MLLM Agentic Workflow',
        '',
        f"- mAP50: `{float(method_row['map50']):.6f}`",
        f"- F1: `{float(method_row['f1']):.6f}`",
        f"- Recall: `{float(method_row['recall']):.6f}`",
        f"- Precision: `{float(method_row['precision']):.6f}`",
        f"- TP/FP/FN: `{method_row['tp']}/{method_row['fp']}/{method_row['fn']}`",
        '',
        '## Best Fixed Thresholds',
        '',
        f"- Best fixed-threshold F1: threshold `{best_f1['threshold']}`, F1 `{float(best_f1['f1']):.6f}`, mAP50 `{float(best_f1['map50']):.6f}`",
        f"- Best fixed-threshold mAP50: threshold `{best_map['threshold']}`, mAP50 `{float(best_map['map50']):.6f}`, F1 `{float(best_map['f1']):.6f}`",
        '',
        '## Artifacts',
        '',
        '- `threshold_vs_mllm_metrics.png`',
        '- `threshold_vs_mllm_metrics.svg`',
        '- `threshold_vs_mllm_metrics.csv`',
        '- `threshold_vs_mllm_metrics.json`',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

