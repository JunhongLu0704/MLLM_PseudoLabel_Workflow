from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..src.constants import DEFAULT_DATA_PATH, DEFAULT_MODEL, DEFAULT_PROVIDER_BASE_URL, DEFAULT_SOURCE_VOC_ROOT
from ..src.workflow import WorkflowRunner


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the clean 256-image MLLM workflow demo.')
    parser.add_argument('--mode', choices=['deterministic_gate_256', 'mllm_prior_llm_final_256', 'mllm_reference_workflow_256'], default='deterministic_gate_256')
    parser.add_argument('--data_path', type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument('--output_dir', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--provider_base_url', default=DEFAULT_PROVIDER_BASE_URL)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--engine', choices=['imperative', 'langgraph'], default='imperative')
    parser.add_argument('--source_voc_root', type=Path, default=Path(DEFAULT_SOURCE_VOC_ROOT))
    parser.add_argument('--reference_bank_dir', type=Path, default=Path('bank/reference_bank'))
    parser.add_argument('--max_exemplars_per_class', type=int, default=10)
    parser.add_argument('--duplicate_iou_threshold', type=float, default=0.40)
    parser.add_argument('--duplicate_class_agnostic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_debug_payloads', action='store_true', help='Write per-tool raw/validated sidecar JSON files. Default keeps one compact trace per image.')
    args = parser.parse_args()

    output_dir = args.output_dir or Path('results') / args.mode
    runner = WorkflowRunner(
        data_path=args.data_path,
        out_dir=output_dir,
        mode=args.mode,
        limit=args.limit,
        provider_base_url=args.provider_base_url,
        model=args.model,
        timeout=args.timeout,
        save_prompt_payloads=args.save_debug_payloads,
        engine=args.engine,
        source_voc_root=args.source_voc_root,
        reference_bank_dir=args.reference_bank_dir,
        max_exemplars_per_class=args.max_exemplars_per_class,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        duplicate_class_agnostic=args.duplicate_class_agnostic,
    )
    summary = runner.run()
    print(json.dumps({'output_dir': output_dir.as_posix(), 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
