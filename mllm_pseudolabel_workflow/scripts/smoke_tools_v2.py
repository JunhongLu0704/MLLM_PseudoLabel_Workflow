from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..src.constants import DEFAULT_DATA_PATH, DEFAULT_SOURCE_VOC_ROOT
from ..src.data import load_records
from ..tools.tools_v2 import (
    build_candidate_pool,
    build_class_exemplar_bank,
    resolve_duplicate_cluster_info,
    retrieve_class_exemplars,
    retrieve_same_class_image_crops,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke-test MLLM workflow v2 tools without provider calls.')
    parser.add_argument('--data_path', type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument('--source_voc_root', type=Path, default=Path(DEFAULT_SOURCE_VOC_ROOT))
    parser.add_argument('--output_dir', type=Path, default=Path('projects/voc_clipart_mllm_demo_256/results/tool_v2_smoke'))
    parser.add_argument('--bank_dir', type=Path, default=Path('projects/voc_clipart_mllm_demo_256/bank/reference_bank'))
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--max_exemplars_per_class', type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data_path, limit=args.limit)
    bank = build_class_exemplar_bank(
        source_voc_root=args.source_voc_root,
        output_dir=args.bank_dir,
        max_per_class=args.max_exemplars_per_class,
    )
    manifest_path = Path(bank.payload.get('manifest_path', args.bank_dir / 'class_exemplar_manifest.json'))

    image_reports = []
    for record in records:
        duplicate = resolve_duplicate_cluster_info(record=record)
        candidates = build_candidate_pool(record=record, duplicate_cluster_info=duplicate.payload)
        first_candidate_id = next(iter(candidates.payload.get('candidate_proposal_ids', [])), None)
        same_class = None
        exemplar = None
        if first_candidate_id is not None:
            proposal = next(item for item in record.proposals if item.proposal_id == int(first_candidate_id))
            same_class = retrieve_same_class_image_crops(
                record=record,
                query_proposal_id=int(first_candidate_id),
                output_dir=args.output_dir / 'same_class_refs',
            )
            exemplar = retrieve_class_exemplars(
                manifest_path=manifest_path,
                class_name=proposal.class_name,
                top_k=2,
            )
        image_reports.append(
            {
                'image_id': record.image_id,
                'duplicate_cluster_info': duplicate.to_dict(),
                'candidate_pool': candidates.to_dict(),
                'same_class_retrieval': same_class.to_dict() if same_class else None,
                'class_exemplar_retrieval': exemplar.to_dict() if exemplar else None,
            }
        )

    invalid_tests = {
        'bad_class_exemplar': retrieve_class_exemplars(manifest_path=manifest_path, class_name='not_voc').to_dict(),
        'missing_manifest': retrieve_class_exemplars(manifest_path=args.output_dir / 'missing_manifest.json', class_name='dog').to_dict(),
        'missing_proposal': retrieve_same_class_image_crops(
            record=records[0],
            query_proposal_id=999999,
            output_dir=args.output_dir / 'same_class_refs',
        ).to_dict()
        if records
        else None,
        'bad_iou_threshold': resolve_duplicate_cluster_info(record=records[0], iou_threshold='bad').to_dict() if records else None,
    }
    summary = {
        'output_dir': args.output_dir.as_posix(),
        'bank': bank.to_dict(),
        'image_reports': image_reports,
        'invalid_input_tests': invalid_tests,
    }
    (args.output_dir / 'tool_v2_smoke_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
