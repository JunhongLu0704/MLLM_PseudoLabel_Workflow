from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


class TraceWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.tool_trace_path = out_dir / 'tool_call_trace.jsonl'
        self.image_trace_path = out_dir / 'image_level_trace.jsonl'
        self.image_traces_dir = out_dir / 'image_traces'
        self._image_tool_records: dict[str, list[dict[str, Any]]] = {}
        out_dir.mkdir(parents=True, exist_ok=True)
        self.image_traces_dir.mkdir(parents=True, exist_ok=True)
        self.tool_trace_path.write_text('', encoding='utf-8')
        self.image_trace_path.write_text('', encoding='utf-8')

    def tool_call(self, record: dict[str, Any]) -> None:
        self._append(self.tool_trace_path, record)
        image_id = str(record.get('image_id', 'run'))
        self._image_tool_records.setdefault(image_id, []).append(record)

    def image_trace(self, record: dict[str, Any]) -> None:
        self._append(self.image_trace_path, record)
        image_id = str(record.get('image_id', 'run'))
        payload = {
            'image_summary': record,
            'tool_calls': self._image_tool_records.get(image_id, []),
        }
        if image_id == 'run':
            filename = 'run_trace.json'
        else:
            try:
                filename = f'image_{int(image_id):04d}_trace.json'
            except ValueError:
                filename = f'image_{image_id}_trace.json'
        write_json(self.image_traces_dir / filename, payload)

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
