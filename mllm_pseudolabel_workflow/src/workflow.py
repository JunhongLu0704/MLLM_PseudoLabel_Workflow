from __future__ import annotations

import base64
import csv
import io
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

from .constants import DEFAULT_MODEL, DEFAULT_PROVIDER_BASE_URL, DEFAULT_SOURCE_VOC_ROOT, VOC_CLASSES
from .data import ImageRecord, Proposal, load_records
from .eval import attach_gt_labels, evaluate_rows, write_metric_diff_report
from .langgraph_runner import build_image_graph
from .provider import OpenAICompatProvider
from .schema import validate_final_decision, validate_inspection, validate_optional_tool_plan, validate_scene_prior
from .tracing import TraceWriter, write_json
from ..tools.tools_v2 import (
    build_candidate_pool,
    build_class_exemplar_bank,
    resolve_duplicate_cluster_info,
    retrieve_class_exemplars,
    retrieve_same_class_image_crops,
)


class WorkflowRunner:
    def __init__(
        self,
        *,
        data_path: Path,
        out_dir: Path,
        mode: str,
        limit: int | None = None,
        provider_base_url: str = DEFAULT_PROVIDER_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
        save_prompt_payloads: bool = True,
        engine: str = 'imperative',
        source_voc_root: Path = Path(DEFAULT_SOURCE_VOC_ROOT),
        reference_bank_dir: Path = Path('projects/voc_clipart_mllm_demo_256/bank/reference_bank'),
        max_exemplars_per_class: int = 10,
        duplicate_iou_threshold: float = 0.40,
        duplicate_class_agnostic: bool = True,
    ) -> None:
        self.data_path = data_path
        self.out_dir = out_dir
        self.mode = mode
        self.limit = limit
        self.provider = OpenAICompatProvider(base_url=provider_base_url, model=model, timeout=timeout)
        self.save_prompt_payloads = save_prompt_payloads
        self.engine = engine
        self.source_voc_root = source_voc_root
        self.reference_bank_dir = reference_bank_dir
        self.max_exemplars_per_class = max_exemplars_per_class
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.duplicate_class_agnostic = bool(duplicate_class_agnostic)
        self.reference_manifest_path = reference_bank_dir / 'class_exemplar_manifest.json'
        self.trace = TraceWriter(out_dir)
        self.tool_counts: dict[str, Counter] = {
            'mllm_scene_prior': Counter(),
            'proposal_verification': Counter(),
            'optional_tool_planner': Counter(),
            'llm_final_decision': Counter(),
            'build_class_exemplar_bank': Counter(),
            'resolve_duplicate_cluster_info': Counter(),
            'build_candidate_pool': Counter(),
            'retrieve_same_class_image_crops': Counter(),
            'retrieve_class_exemplars': Counter(),
        }
        self._image_graph = None

    def run(self) -> dict[str, Any]:
        if self.mode == 'mllm_reference_workflow_256':
            self._ensure_reference_bank()
        records = load_records(self.data_path, limit=self.limit)
        rows: list[dict[str, Any]] = []
        for record in records:
            image_rows = self._run_image(record)
            rows.extend(image_rows)
        attach_gt_labels(records, rows)
        for row in rows:
            row['tp_fp_fn_label_for_eval'] = 'tp' if row['gt_correct'] and row['final_decision'] == 'keep' else 'fp' if row['final_decision'] == 'keep' else 'fn' if row['gt_correct'] else 'tn'
        summary = self._write_outputs(records, rows)
        return summary

    def _run_image(self, record: ImageRecord) -> list[dict[str, Any]]:
        if self.engine == 'langgraph':
            state = self._run_image_langgraph(record)
            return list(state.get('rows', []))
        return self._run_image_imperative(record)

    def _run_image_imperative(self, record: ImageRecord) -> list[dict[str, Any]]:
        merged_prior, prior_status = self._build_priors(record)
        image_rows: list[dict[str, Any]] = []
        for proposal in record.proposals:
            row = self._process_proposal(record, proposal, merged_prior, prior_status)
            image_rows.append(row)
        self.trace.image_trace(
            {
                'image_id': record.image_id,
                'num_proposals': len(record.proposals),
                'num_valid_proposals': sum(1 for row in image_rows if not row['degenerate_bbox']),
                'mllm_prior_status': prior_status,
            }
        )
        return image_rows

    def _run_image_langgraph(self, record: ImageRecord) -> dict[str, Any]:
        graph = self._get_image_graph()
        return graph.invoke({'record': record})

    def _get_image_graph(self):
        if self._image_graph is None:
            self._image_graph = build_image_graph(self)
        return self._image_graph

    def _build_priors(self, record: ImageRecord) -> tuple[dict[str, Any], str]:
        task_prior = {
            'class_names': VOC_CLASSES,
            'dataset_style_hint': 'clipart',
            'review_guideline': 'Scene prior is a soft hint only; final keep/drop is proposal-level.',
        }
        scene_prior_heuristic = heuristic_scene_prior(record)
        teacher_summary = teacher_proposal_summary(record)
        mllm_prior = {}
        status = 'not_called'
        if self.mode in {'mllm_prior_llm_final_256', 'mllm_reference_workflow_256'}:
            mllm_prior, status = self._call_scene_prior(record, task_prior, scene_prior_heuristic, teacher_summary)
        merged = {
            'task_prior': task_prior,
            'heuristic_scene_prior': scene_prior_heuristic,
            'mllm_scene_prior': mllm_prior,
            'merged_review_guidance': list(scene_prior_heuristic.get('review_guidance', [])) + list(mllm_prior.get('review_guidance', [])),
            'prior_sources': ['handcrafted', 'heuristic'] + (['mllm'] if mllm_prior else []),
            'mllm_prior_fallback': status not in {'success', 'not_called'},
        }
        return merged, status

    def _ensure_reference_bank(self) -> None:
        result = build_class_exemplar_bank(
            source_voc_root=self.source_voc_root,
            output_dir=self.reference_bank_dir,
            max_per_class=self.max_exemplars_per_class,
        )
        self._record_local_tool_call(
            result.tool_name,
            result.to_dict(),
            image_id='run',
            output_stem='build_class_exemplar_bank',
        )
        if result.status == 'success':
            self.reference_manifest_path = Path(result.payload.get('manifest_path', self.reference_manifest_path))

    def _build_duplicate_cluster_info(self, record: ImageRecord) -> dict[str, Any]:
        result = resolve_duplicate_cluster_info(
            record=record,
            iou_threshold=self.duplicate_iou_threshold,
            class_agnostic=self.duplicate_class_agnostic,
        )
        self._record_local_tool_call(
            result.tool_name,
            result.to_dict(),
            image_id=record.image_id,
            output_stem=f'image_{record.image_id:04d}_duplicate_clusters',
        )
        return dict(result.payload) if result.status == 'success' else {'duplicate_clusters': [], 'cluster_count': 0, 'status': result.status, 'error_code': result.error_code}

    def _build_candidate_pool(self, record: ImageRecord, duplicate_cluster_info: dict[str, Any]) -> dict[str, Any]:
        result = build_candidate_pool(record=record, duplicate_cluster_info=duplicate_cluster_info)
        self._record_local_tool_call(
            result.tool_name,
            result.to_dict(),
            image_id=record.image_id,
            output_stem=f'image_{record.image_id:04d}_candidate_pool',
        )
        return dict(result.payload) if result.status == 'success' else {'candidate_proposal_ids': [], 'candidate_reasons': {}, 'status': result.status, 'error_code': result.error_code}

    def _process_reference_workflow_proposals(
        self,
        record: ImageRecord,
        merged_prior: dict[str, Any],
        prior_status: str,
        duplicate_cluster_info: dict[str, Any],
        candidate_pool: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidate_ids = {int(item) for item in candidate_pool.get('candidate_proposal_ids', [])}
        candidate_reasons = {int(key): list(value) for key, value in candidate_pool.get('candidate_reasons', {}).items()}
        rows: list[dict[str, Any]] = []
        verification_by_id: dict[int, dict[str, Any]] = {}
        duplicate_ids = {
            int(pid)
            for cluster in duplicate_cluster_info.get('duplicate_clusters', [])
            for pid in cluster.get('proposal_ids', [])
        }
        for proposal in record.proposals:
            if proposal.proposal_id in candidate_ids:
                inspection = self._verify_proposal(record, proposal, merged_prior)
            else:
                inspection = heuristic_inspection(proposal)
                inspection['inspection_status'] = 'not_candidate_heuristic'
            verification_by_id[proposal.proposal_id] = inspection

            optional_evidence = {}
            reasons = candidate_reasons.get(proposal.proposal_id, [])
            should_retrieve = (
                proposal.proposal_id in candidate_ids
                and not proposal.degenerate_bbox
                and (
                    inspection.get('visual_validity') == 'uncertain'
                    or inspection.get('evidence_strength') == 'weak'
                    or proposal.proposal_id in duplicate_ids
                    or 'duplicate_cluster' in reasons
                )
            )
            if should_retrieve:
                tool_plan = self._plan_optional_tools(record, proposal, merged_prior, inspection, reasons, duplicate_cluster_info)
                selected_tools = [item['tool_name'] for item in tool_plan.get('selected_tools', [])]
                optional_evidence = self._retrieve_reference_evidence(record, proposal, verification_by_id, selected_tools=selected_tools)
            else:
                tool_plan = {'status': 'not_called', 'selected_tools': []}
            gate_decision = deterministic_gate(proposal, inspection)
            final = {'status': 'not_called', 'decision': gate_decision, 'confidence': ''}
            if proposal.proposal_id in candidate_ids and not proposal.degenerate_bbox:
                final = self._call_final_decision(record, proposal, merged_prior, inspection, gate_decision, optional_evidence=optional_evidence)
            row = self._row_from_decision(
                record,
                proposal,
                inspection,
                prior_status,
                gate_decision,
                final,
                final_source='llm_final_decision' if final.get('status') == 'success' else 'deterministic_gate' if final.get('status') == 'not_called' else 'deterministic_gate_fallback',
                fallback_used=bool(inspection.get('fallback_used', False) or final.get('status') == 'fallback'),
            )
            row['candidate_pool_selected'] = proposal.proposal_id in candidate_ids
            row['candidate_reasons'] = '|'.join(reasons)
            row['optional_tools_used'] = '|'.join(optional_evidence.keys())
            row['optional_tool_plan_status'] = tool_plan.get('status', 'not_called')
            row['optional_tool_plan_selected'] = '|'.join(item['tool_name'] for item in tool_plan.get('selected_tools', []))
            rows.append(row)
        return rows

    def _plan_optional_tools(
        self,
        record: ImageRecord,
        proposal: Proposal,
        merged_prior: dict[str, Any],
        inspection: dict[str, Any],
        trigger_reasons: list[str],
        duplicate_cluster_info: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = 'optional_tool_planner'
        raw_path = self.out_dir / 'optional_tool_planner' / f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_raw.json'
        validated_path = self.out_dir / 'optional_tool_planner' / f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_validated.json'
        matching_clusters = [
            cluster
            for cluster in duplicate_cluster_info.get('duplicate_clusters', [])
            if proposal.proposal_id in {int(pid) for pid in cluster.get('proposal_ids', [])}
        ]
        payload = {
            'image_id': str(record.image_id),
            'proposal_id': str(proposal.proposal_id),
            'predicted_class': proposal.class_name,
            'teacher_score': proposal.score,
            'bbox': proposal.clamped_bbox,
            'proposal_metadata': {'short_side': proposal.short_side, 'area': proposal.area},
            'inspection_result': inspection_without_trace(inspection),
            'trigger_reasons': trigger_reasons,
            'duplicate_cluster_summary': matching_clusters[:2],
            'merged_prior': compact_prior(merged_prior),
            'available_tools': [
                {
                    'tool_name': 'retrieve_same_class_image_crops',
                    'purpose': 'retrieve same-class proposal crops from the current target image for local comparison',
                    'arguments': {'top_k': 'integer 1..4'},
                },
                {
                    'tool_name': 'retrieve_class_exemplars',
                    'purpose': 'retrieve source-domain VOC GT exemplar crop metadata for the predicted class',
                    'arguments': {'class_name': proposal.class_name, 'top_k': 'integer 1..4'},
                },
            ],
            'required_output_schema': {
                'proposal_id': str(proposal.proposal_id),
                'selected_tools': [
                    {'tool_name': 'retrieve_same_class_image_crops | retrieve_class_exemplars', 'arguments': {}, 'reason': 'short string'}
                ],
                'reason': 'short string',
            },
        }
        provider_result = self.provider.chat_json(system=OPTIONAL_TOOL_PLANNER_SYSTEM, user_content=json.dumps(payload, ensure_ascii=False), max_tokens=256)
        validation = validate_optional_tool_plan(provider_result.content, proposal.proposal_id)
        fallback_used = provider_result.status != 'ok' or not validation.ok
        value = validation.value if not fallback_used else {
            'proposal_id': str(proposal.proposal_id),
            'selected_tools': [
                {'tool_name': 'retrieve_same_class_image_crops', 'arguments': {'top_k': 4}, 'reason': 'planner fallback to fixed optional evidence'},
                {'tool_name': 'retrieve_class_exemplars', 'arguments': {'class_name': proposal.class_name, 'top_k': 3}, 'reason': 'planner fallback to fixed optional evidence'},
            ],
            'reason': 'fallback to fixed optional evidence policy',
        }
        value['status'] = 'success' if not fallback_used else 'fallback'
        raw_payload = {'provider_raw': provider_result.raw, 'content': provider_result.content, 'error': provider_result.error}
        self._maybe_write_payload(raw_path, raw_payload)
        self._maybe_write_payload(validated_path, value)
        trace = self._tool_trace_record(
            tool_name=tool_name,
            image_id=record.image_id,
            proposal_id=proposal.proposal_id,
            provider_status=provider_result.status,
            parse_status=validation.parse_status,
            repair_used=validation.repair_used,
            fallback_used=fallback_used,
            fallback_reason='' if not fallback_used else provider_result.error or validation.error,
            latency_ms=provider_result.latency_ms,
            **token_usage_fields(provider_result.raw),
            raw_output_path=raw_path,
            validated_output_path=validated_path,
            raw_output=raw_payload,
            validated_output=value,
        )
        self.trace.tool_call(trace)
        self._count_tool(tool_name, trace, validation_ok=validation.ok)
        return value

    def _retrieve_reference_evidence(
        self,
        record: ImageRecord,
        proposal: Proposal,
        verification_by_id: dict[int, dict[str, Any]],
        *,
        selected_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        selected = set(selected_tools or ['retrieve_same_class_image_crops', 'retrieve_class_exemplars'])
        if 'retrieve_same_class_image_crops' in selected:
            same_class = retrieve_same_class_image_crops(
                record=record,
                query_proposal_id=proposal.proposal_id,
                output_dir=self.out_dir / 'same_class_refs',
                verification_results=verification_by_id,
            )
            self._record_local_tool_call(
                same_class.tool_name,
                same_class.to_dict(),
                image_id=record.image_id,
                proposal_id=proposal.proposal_id,
                output_stem=f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_same_class_refs',
            )
            if same_class.status in {'success', 'empty'}:
                evidence['same_class_image_crops'] = same_class.payload
        if 'retrieve_class_exemplars' in selected:
            exemplars = retrieve_class_exemplars(
                manifest_path=self.reference_manifest_path,
                class_name=proposal.class_name,
                top_k=3,
            )
            self._record_local_tool_call(
                exemplars.tool_name,
                exemplars.to_dict(),
                image_id=record.image_id,
                proposal_id=proposal.proposal_id,
                output_stem=f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_class_exemplars',
            )
            if exemplars.status in {'success', 'empty'}:
                evidence['class_exemplars'] = exemplars.payload
        return evidence

    def _call_scene_prior(self, record: ImageRecord, task_prior: dict[str, Any], scene_prior: dict[str, Any], teacher_summary: dict[str, Any]) -> tuple[dict[str, Any], str]:
        tool_name = 'mllm_scene_prior'
        raw_path = self.out_dir / 'mllm_scene_prior' / f'image_{record.image_id:04d}_raw.json'
        validated_path = self.out_dir / 'mllm_scene_prior' / f'image_{record.image_id:04d}_validated.json'
        prompt_payload = {
            'image_id': str(record.image_id),
            'task': 'Generate scene-level soft hints for VOC pseudo-label review. Do not decide keep/drop.',
            'class_names': VOC_CLASSES,
            'teacher_proposal_summary': teacher_summary,
            'heuristic_scene_prior': scene_prior,
            'required_output_schema': {
                'scene_type': 'string',
                'image_style': 'clipart | natural | mixed | uncertain',
                'visible_object_hints': ['VOC class names only'],
                'unlikely_classes': ['VOC class names only'],
                'potential_confusions': [{'class_a': 'VOC class', 'class_b': 'VOC class', 'reason': 'short string'}],
                'review_guidance': ['short strings, no keep/drop decisions'],
                'confidence': 'low | medium | high',
            },
        }
        started = time.perf_counter()
        provider_result = self.provider.chat_json(
            system=SCENE_PRIOR_SYSTEM,
            user_content=[
                {'type': 'text', 'text': json.dumps(prompt_payload, ensure_ascii=False)},
                {'type': 'image_url', 'image_url': {'url': image_to_data_url(record.image_path)}},
            ],
            max_tokens=512,
        )
        validation = validate_scene_prior(provider_result.content)
        fallback_used = provider_result.status != 'ok' or not validation.ok
        value = validation.value if not fallback_used else {}
        raw_payload = {'provider_raw': provider_result.raw, 'content': provider_result.content, 'error': provider_result.error}
        self._maybe_write_payload(raw_path, raw_payload)
        self._maybe_write_payload(validated_path, value)
        trace = self._tool_trace_record(
            tool_name=tool_name,
            image_id=record.image_id,
            provider_status=provider_result.status,
            parse_status=validation.parse_status,
            repair_used=validation.repair_used,
            fallback_used=fallback_used,
            fallback_reason='' if not fallback_used else provider_result.error or validation.error,
            latency_ms=provider_result.latency_ms or int((time.perf_counter() - started) * 1000),
            **token_usage_fields(provider_result.raw),
            raw_output_path=raw_path,
            validated_output_path=validated_path,
            raw_output=raw_payload,
            validated_output=value,
        )
        self.trace.tool_call(trace)
        self._count_tool(tool_name, trace, validation_ok=validation.ok)
        return value, 'success' if not fallback_used else 'fallback'

    def _process_proposal(self, record: ImageRecord, proposal: Proposal, merged_prior: dict[str, Any], prior_status: str) -> dict[str, Any]:
        inspection = self._verify_proposal(record, proposal, merged_prior)
        gate_decision = deterministic_gate(proposal, inspection)
        llm_status = 'not_called'
        llm_decision = ''
        llm_confidence = ''
        final_decision = gate_decision
        final_source = 'deterministic_gate'
        fallback_used = inspection.get('fallback_used', False)
        if self.mode == 'mllm_prior_llm_final_256' and not proposal.degenerate_bbox:
            final = self._call_final_decision(record, proposal, merged_prior, inspection, gate_decision)
            llm_status = final['status']
            llm_decision = final.get('decision', '')
            llm_confidence = final.get('confidence', '')
            if final['status'] == 'success':
                final_decision = final['decision']
                final_source = 'llm_final_decision'
            else:
                final_source = 'deterministic_gate_fallback'
                fallback_used = True
        return self._row_from_decision(
            record,
            proposal,
            inspection,
            prior_status,
            gate_decision,
            {'status': llm_status, 'decision': llm_decision, 'confidence': llm_confidence},
            final_source=final_source,
            fallback_used=bool(fallback_used),
            final_decision_override=final_decision,
        )

    def _row_from_decision(
        self,
        record: ImageRecord,
        proposal: Proposal,
        inspection: dict[str, Any],
        prior_status: str,
        gate_decision: str,
        final: dict[str, Any],
        *,
        final_source: str,
        fallback_used: bool,
        final_decision_override: str | None = None,
    ) -> dict[str, Any]:
        final_decision = final_decision_override or str(final.get('decision') or gate_decision)
        return {
            'image_id': record.image_id,
            'proposal_id': proposal.proposal_id,
            'predicted_class': proposal.class_name,
            'score': proposal.score,
            'bbox_x1': proposal.clamped_bbox[0],
            'bbox_y1': proposal.clamped_bbox[1],
            'bbox_x2': proposal.clamped_bbox[2],
            'bbox_y2': proposal.clamped_bbox[3],
            'bbox': proposal.clamped_bbox,
            'short_side': proposal.short_side,
            'degenerate_bbox': proposal.degenerate_bbox,
            'visual_validity': inspection['visual_validity'],
            'evidence_strength': inspection['evidence_strength'],
            'recommended_action': inspection['recommended_action'],
            'inspection_status': inspection['inspection_status'],
            'mllm_prior_status': prior_status,
            'llm_final_status': final.get('status', 'not_called'),
            'llm_final_decision': final.get('decision', ''),
            'llm_final_confidence': final.get('confidence', ''),
            'score_baseline_020': 'keep' if proposal.score >= 0.20 else 'drop',
            'deterministic_gate_256': gate_decision,
            'final_decision': final_decision,
            'final_decision_source': final_source,
            'fallback_used': fallback_used,
            '_proposal': proposal,
        }

    def _verify_proposal(self, record: ImageRecord, proposal: Proposal, merged_prior: dict[str, Any]) -> dict[str, Any]:
        if proposal.degenerate_bbox:
            return {'visual_validity': 'unavailable', 'evidence_strength': 'unavailable', 'recommended_action': 'drop', 'inspection_status': 'degenerate', 'fallback_used': True}
        if self.mode == 'deterministic_gate_256':
            return heuristic_inspection(proposal)
        tool_name = 'proposal_verification'
        raw_path = self.out_dir / 'proposal_verification' / f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_raw.json'
        validated_path = self.out_dir / 'proposal_verification' / f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_validated.json'
        payload = model_safe_proposal_payload(record, proposal, merged_prior)
        payload['required_output_schema'] = {
            'proposal_id': str(proposal.proposal_id),
            'visual_validity': 'valid | uncertain | invalid | unavailable',
            'evidence_strength': 'weak | medium | strong | unavailable',
            'recommended_action': 'keep | low_weight | drop | unavailable',
        }
        user_content = [
            {'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)},
            {'type': 'image_url', 'image_url': {'url': crop_to_data_url(record.image_path, proposal.clamped_bbox)}},
            {'type': 'image_url', 'image_url': {'url': marker_to_data_url(record.image_path, proposal.clamped_bbox)}},
        ]
        provider_result = self.provider.chat_json(system=INSPECTION_SYSTEM, user_content=user_content, max_tokens=256)
        validation = validate_inspection(provider_result.content, proposal.proposal_id)
        fallback_used = provider_result.status != 'ok' or not validation.ok
        value = validation.value if not fallback_used else heuristic_inspection(proposal)
        value['fallback_used'] = fallback_used
        raw_payload = {'provider_raw': provider_result.raw, 'content': provider_result.content, 'error': provider_result.error}
        self._maybe_write_payload(raw_path, raw_payload)
        self._maybe_write_payload(validated_path, value)
        trace = self._tool_trace_record(
            tool_name=tool_name,
            image_id=record.image_id,
            proposal_id=proposal.proposal_id,
            provider_status=provider_result.status,
            parse_status=validation.parse_status,
            repair_used=validation.repair_used,
            fallback_used=fallback_used,
            fallback_reason='' if not fallback_used else provider_result.error or validation.error,
            latency_ms=provider_result.latency_ms,
            **token_usage_fields(provider_result.raw),
            raw_output_path=raw_path,
            validated_output_path=validated_path,
            raw_output=raw_payload,
            validated_output=value,
        )
        self.trace.tool_call(trace)
        self._count_tool(tool_name, trace, validation_ok=validation.ok)
        return value

    def _call_final_decision(
        self,
        record: ImageRecord,
        proposal: Proposal,
        merged_prior: dict[str, Any],
        inspection: dict[str, Any],
        fallback_decision: str,
        optional_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_name = 'llm_final_decision'
        raw_path = self.out_dir / 'llm_final_decision' / f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_raw.json'
        validated_path = self.out_dir / 'llm_final_decision' / f'image_{record.image_id:04d}_proposal_{proposal.proposal_id:04d}_validated.json'
        payload = {
            'image_id': str(record.image_id),
            'proposal_id': str(proposal.proposal_id),
            'teacher_score': proposal.score,
            'predicted_class': proposal.class_name,
            'bbox': proposal.clamped_bbox,
            'proposal_metadata': {'short_side': proposal.short_side, 'area': proposal.area},
            'merged_prior': compact_prior(merged_prior),
            'inspection_result_source': 'VerifySingleBox_MM primary visual verification tool',
            'inspection_result_semantics': {
                'visual_validity': 'tool observation about whether the proposal visually matches the predicted class',
                'evidence_strength': 'tool confidence in the visual observation',
                'recommended_action': 'tool recommendation only; not a deterministic rule or ground-truth label',
            },
            'inspection_result': inspection_without_trace(inspection),
            'optional_evidence': optional_evidence or {},
            'decision_policy': {
                'primary_signal': 'VerifySingleBox_MM',
                'optional_tools_are_supporting_only': True,
                'do_not_override_strong_verification': True,
            },
            'context_summary': {'near_border': is_near_border(proposal, record), 'score_band': score_band(proposal.score)},
            'required_output_schema': {
                'proposal_id': str(proposal.proposal_id),
                'decision': 'keep | low_weight | drop',
                'confidence': 'low | medium | high',
                'reason': 'short string',
                'used_signals': ['teacher_score', 'visual_validity', 'evidence_strength', 'prior', 'context'],
            },
        }
        provider_result = self.provider.chat_json(system=FINAL_DECISION_SYSTEM, user_content=json.dumps(payload, ensure_ascii=False), max_tokens=256)
        validation = validate_final_decision(provider_result.content, proposal.proposal_id)
        fallback_used = provider_result.status != 'ok' or not validation.ok
        value = validation.value if not fallback_used else {'proposal_id': str(proposal.proposal_id), 'decision': fallback_decision, 'confidence': 'low', 'reason': 'fallback to deterministic gate', 'used_signals': ['deterministic_gate']}
        raw_payload = {'provider_raw': provider_result.raw, 'content': provider_result.content, 'error': provider_result.error}
        self._maybe_write_payload(raw_path, raw_payload)
        self._maybe_write_payload(validated_path, value)
        trace = self._tool_trace_record(
            tool_name=tool_name,
            image_id=record.image_id,
            proposal_id=proposal.proposal_id,
            provider_status=provider_result.status,
            parse_status=validation.parse_status,
            repair_used=validation.repair_used,
            fallback_used=fallback_used,
            fallback_reason='' if not fallback_used else provider_result.error or validation.error,
            latency_ms=provider_result.latency_ms,
            **token_usage_fields(provider_result.raw),
            raw_output_path=raw_path,
            validated_output_path=validated_path,
            raw_output=raw_payload,
            validated_output=value,
        )
        self.trace.tool_call(trace)
        self._count_tool(tool_name, trace, validation_ok=validation.ok)
        value['status'] = 'success' if not fallback_used else 'fallback'
        return value

    def _record_local_tool_call(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        image_id: int | str,
        output_stem: str,
        proposal_id: int | None = None,
    ) -> None:
        raw_path = self.out_dir / 'local_tools' / f'{output_stem}_raw.json'
        validated_path = self.out_dir / 'local_tools' / f'{output_stem}_validated.json'
        self._maybe_write_payload(raw_path, payload)
        self._maybe_write_payload(validated_path, payload)
        status = str(payload.get('status', ''))
        trace = self._tool_trace_record(
            tool_name=tool_name,
            image_id=image_id,
            provider_status='local' if status != 'error' else 'local_error',
            parse_status='ok',
            repair_used=False,
            fallback_used=status == 'error',
            fallback_reason=str(payload.get('error_code') or payload.get('message') or ''),
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            raw_output_path=raw_path,
            validated_output_path=validated_path,
            raw_output=payload,
            validated_output=payload,
        )
        if proposal_id is not None:
            trace['proposal_id'] = proposal_id
        self.trace.tool_call(trace)
        self.tool_counts.setdefault(tool_name, Counter()).update(['success' if status != 'error' else 'error'])

    def _maybe_write_payload(self, path: Path, payload: Any) -> None:
        if self.save_prompt_payloads:
            write_json(path, payload)

    def _write_outputs(self, records: list[ImageRecord], rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = {
            'score_baseline_020': evaluate_rows(records, rows, decision_field='score_baseline_020'),
            'deterministic_gate_256': evaluate_rows(records, rows, decision_field='deterministic_gate_256'),
            self.mode: evaluate_rows(records, rows, decision_field='final_decision'),
        }
        summary = {
            'mode': self.mode,
            'engine': self.engine,
            'num_images': len(records),
            'num_total_proposals': len(rows),
            'num_valid_proposals': sum(1 for row in rows if not row['degenerate_bbox']),
            'metrics': metrics,
            'tool_call_summary': {name: dict(counter) for name, counter in self.tool_counts.items()},
            'token_usage_summary': self._token_usage_summary(),
        }
        write_json(self.out_dir / 'summary_metrics.json', summary)
        write_json(self.out_dir / 'tool_call_summary.json', summary['tool_call_summary'])
        write_json(self.out_dir / 'token_usage_summary.json', summary['token_usage_summary'])
        self._write_csv(rows)
        write_metric_diff_report(self.out_dir / 'metric_diff_report.md', summary, rows)
        write_examples(self.out_dir / 'examples', rows)
        write_json(
            self.out_dir / 'config.json',
            {
                'mode': self.mode,
                'engine': self.engine,
                'data_path': str(self.data_path.as_posix()),
                'limit': self.limit,
                'source_voc_root': self.source_voc_root.as_posix(),
                'reference_bank_dir': self.reference_bank_dir.as_posix(),
                'reference_manifest_path': self.reference_manifest_path.as_posix(),
                'max_exemplars_per_class': self.max_exemplars_per_class,
                'duplicate_iou_threshold': self.duplicate_iou_threshold,
                'duplicate_class_agnostic': self.duplicate_class_agnostic,
            },
        )
        return summary

    def _write_csv(self, rows: list[dict[str, Any]]) -> None:
        fields = [
            'image_id', 'proposal_id', 'predicted_class', 'score', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'visual_validity', 'evidence_strength', 'recommended_action', 'inspection_status',
            'mllm_prior_status', 'llm_final_status', 'llm_final_decision', 'llm_final_confidence',
            'final_decision', 'final_decision_source', 'fallback_used', 'gt_correct', 'tp_fp_fn_label_for_eval',
            'max_iou_same_class', 'matched_gt_id', 'degenerate_bbox', 'score_baseline_020', 'deterministic_gate_256',
            'candidate_pool_selected', 'candidate_reasons', 'optional_tool_plan_status', 'optional_tool_plan_selected', 'optional_tools_used',
        ]
        with (self.out_dir / 'proposal_results.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, '') for field in fields})

    @staticmethod
    def _tool_trace_record(**kwargs: Any) -> dict[str, Any]:
        record = dict(kwargs)
        record['raw_output_path'] = str(Path(record['raw_output_path']).as_posix())
        record['validated_output_path'] = str(Path(record['validated_output_path']).as_posix())
        return record

    def _count_tool(self, tool_name: str, trace: dict[str, Any], *, validation_ok: bool) -> None:
        if trace['provider_status'] != 'ok':
            self.tool_counts[tool_name].update(['provider_failed'])
        elif trace['parse_status'] == 'failed':
            self.tool_counts[tool_name].update(['parse_failed'])
        elif not validation_ok:
            self.tool_counts[tool_name].update(['validation_failed'])
        else:
            self.tool_counts[tool_name].update(['success'])
        if trace['repair_used']:
            self.tool_counts[tool_name].update(['repair_used'])
        if trace['fallback_used']:
            self.tool_counts[tool_name].update(['fallback'])
        self.tool_counts[tool_name].update(
            {
                'prompt_tokens': int(trace.get('prompt_tokens', 0) or 0),
                'completion_tokens': int(trace.get('completion_tokens', 0) or 0),
                'total_tokens': int(trace.get('total_tokens', 0) or 0),
                'latency_ms': int(trace.get('latency_ms', 0) or 0),
            }
        )

    def _token_usage_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for tool_name, counter in self.tool_counts.items():
            calls = sum(
                int(counter.get(key, 0))
                for key in ('success', 'provider_failed', 'parse_failed', 'validation_failed', 'error')
            )
            prompt_tokens = int(counter.get('prompt_tokens', 0))
            completion_tokens = int(counter.get('completion_tokens', 0))
            total_tokens = int(counter.get('total_tokens', 0))
            latency_ms = int(counter.get('latency_ms', 0))
            summary[tool_name] = {
                'calls': calls,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'avg_prompt_tokens': round(prompt_tokens / calls, 2) if calls else 0.0,
                'avg_completion_tokens': round(completion_tokens / calls, 2) if calls else 0.0,
                'avg_total_tokens': round(total_tokens / calls, 2) if calls else 0.0,
                'latency_ms': latency_ms,
                'avg_latency_ms': round(latency_ms / calls, 2) if calls else 0.0,
            }
        summary['_total'] = {
            'calls': sum(item['calls'] for item in summary.values()),
            'prompt_tokens': sum(item['prompt_tokens'] for item in summary.values()),
            'completion_tokens': sum(item['completion_tokens'] for item in summary.values()),
            'total_tokens': sum(item['total_tokens'] for item in summary.values()),
            'latency_ms': sum(item['latency_ms'] for item in summary.values()),
        }
        return summary


SCENE_PRIOR_SYSTEM = (
    'You are a visual prior construction tool. Return one JSON object only. '
    'Do not use markdown. Do not make keep/drop decisions. Do not use ground truth. '
    'Use only allowed enum strings and VOC class names from the provided class list.'
)
INSPECTION_SYSTEM = (
    'You are a proposal-level visual verification tool. Return one JSON object only. '
    'Do not use markdown. Do not use booleans. Do not use accept/reject. Do not use ground truth. '
    'visual_validity must be one of valid, uncertain, invalid, unavailable. '
    'evidence_strength must be one of weak, medium, strong, unavailable. '
    'recommended_action must be one of keep, low_weight, drop, unavailable.'
)
OPTIONAL_TOOL_PLANNER_SYSTEM = (
    'You are a bounded optional evidence tool planner inside a fixed pseudo-label QA workflow. '
    'Return one JSON object only. Do not use markdown. Do not use ground truth. '
    'Select one or more tools only from the provided available_tools list. '
    'These tools provide supporting evidence only and must not directly decide keep/drop.'
)
FINAL_DECISION_SYSTEM = (
    'You are a constrained final decision tool inside a fixed workflow. Return one JSON object only. '
    'Do not use markdown. Use only provided evidence. Do not use ground truth. '
    'proposal_id must exactly match the provided proposal_id. confidence must be low, medium, or high. '
    'The inspection_result comes from VerifySingleBox_MM. Its recommended_action is a tool recommendation, '
    'not a deterministic rule and not ground truth; make your own bounded decision from all provided evidence.'
)


def heuristic_scene_prior(record: ImageRecord) -> dict[str, Any]:
    if record.image_path.exists():
        with Image.open(record.image_path) as image:
            rgb = image.convert('RGB')
            stats = ImageStat.Stat(rgb)
        rgb_mean = [round(v, 2) for v in stats.mean[:3]]
        rgb_std = [round(v, 2) for v in stats.stddev[:3]]
        image_available = True
    else:
        rgb_mean = []
        rgb_std = []
        image_available = False
    return {
        'scene_type': 'object-centric synthetic scene',
        'image_style': 'clipart',
        'image_stats': {'rgb_mean': rgb_mean, 'rgb_std': rgb_std, 'width': record.width, 'height': record.height, 'image_available': image_available},
        'review_guidance': ['Treat scene prior as a soft hint only.', 'Border-touching detections deserve extra scrutiny.'],
    }


def teacher_proposal_summary(record: ImageRecord) -> dict[str, Any]:
    by_class: dict[str, list[float]] = {}
    for proposal in record.proposals:
        by_class.setdefault(proposal.class_name, []).append(proposal.score)
    top = sorted(([{'class': key, 'count': len(vals), 'max_score': round(max(vals), 4)} for key, vals in by_class.items()]), key=lambda x: (-x['count'], -x['max_score']))[:8]
    scores = [proposal.score for proposal in record.proposals]
    return {'num_proposals': len(record.proposals), 'top_classes': top, 'score_range': [round(min(scores), 4), round(max(scores), 4)] if scores else [0, 0], 'small_box_count': sum(1 for p in record.proposals if p.short_side < 32), 'large_box_count': sum(1 for p in record.proposals if p.area > record.width * record.height * 0.20)}


def model_safe_proposal_payload(record: ImageRecord, proposal: Proposal, merged_prior: dict[str, Any]) -> dict[str, Any]:
    return {
        'image_id': str(record.image_id),
        'proposal_id': str(proposal.proposal_id),
        'predicted_class': proposal.class_name,
        'teacher_score': proposal.score,
        'bbox': proposal.clamped_bbox,
        'proposal_metadata': {'short_side': proposal.short_side, 'area': proposal.area},
        'merged_prior': compact_prior(merged_prior),
        'context_summary': {'near_border': is_near_border(proposal, record), 'score_band': score_band(proposal.score)},
    }


def compact_prior(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        'task_classes': prior.get('task_prior', {}).get('class_names', []),
        'heuristic_scene': prior.get('heuristic_scene_prior', {}),
        'mllm_scene_prior': prior.get('mllm_scene_prior', {}),
        'merged_review_guidance': prior.get('merged_review_guidance', [])[:8],
    }


def heuristic_inspection(proposal: Proposal) -> dict[str, Any]:
    if proposal.degenerate_bbox:
        return {'visual_validity': 'unavailable', 'evidence_strength': 'unavailable', 'recommended_action': 'drop', 'inspection_status': 'degenerate', 'fallback_used': True}
    if proposal.score >= 0.50:
        return {'visual_validity': 'valid', 'evidence_strength': 'medium', 'recommended_action': 'keep', 'inspection_status': 'heuristic', 'fallback_used': False}
    if proposal.score < 0.20 and (proposal.short_side < 48 or proposal.area < 48 * 48):
        return {'visual_validity': 'invalid', 'evidence_strength': 'medium', 'recommended_action': 'drop', 'inspection_status': 'heuristic', 'fallback_used': False}
    return {'visual_validity': 'uncertain', 'evidence_strength': 'weak', 'recommended_action': 'keep', 'inspection_status': 'heuristic', 'fallback_used': False}


def deterministic_gate(proposal: Proposal, inspection: dict[str, Any]) -> str:
    validity = inspection.get('visual_validity', 'unavailable')
    if validity == 'invalid':
        return 'drop'
    if validity in {'valid', 'uncertain'}:
        return 'keep'
    return 'keep' if proposal.score >= 0.20 else 'drop'


def inspection_without_trace(inspection: dict[str, Any]) -> dict[str, Any]:
    return {key: inspection.get(key) for key in ('visual_validity', 'evidence_strength', 'recommended_action', 'inspection_status')}


def token_usage_fields(raw: dict[str, Any]) -> dict[str, int]:
    usage = raw.get('usage', {}) if isinstance(raw, dict) else {}
    timings = raw.get('timings', {}) if isinstance(raw, dict) else {}
    prompt_tokens = usage.get('prompt_tokens', timings.get('prompt_n', 0))
    completion_tokens = usage.get('completion_tokens', timings.get('predicted_n', 0))
    total_tokens = usage.get('total_tokens', int(prompt_tokens or 0) + int(completion_tokens or 0))
    return {
        'prompt_tokens': int(prompt_tokens or 0),
        'completion_tokens': int(completion_tokens or 0),
        'total_tokens': int(total_tokens or 0),
    }


def is_near_border(proposal: Proposal, record: ImageRecord) -> bool:
    x1, y1, x2, y2 = proposal.clamped_bbox
    return x1 <= record.width * 0.03 or y1 <= record.height * 0.03 or x2 >= record.width * 0.97 or y2 >= record.height * 0.97


def score_band(score: float) -> str:
    if score < 0.15:
        return '[0.10,0.15)'
    if score < 0.20:
        return '[0.15,0.20)'
    if score < 0.30:
        return '[0.20,0.30)'
    if score < 0.50:
        return '[0.30,0.50)'
    return '>=0.50'


def image_to_data_url(path: Path) -> str:
    with Image.open(path) as image:
        return pil_to_data_url(image.convert('RGB'))


def crop_to_data_url(path: Path, bbox: list[float]) -> str:
    with Image.open(path) as image:
        return pil_to_data_url(image.convert('RGB').crop(tuple(int(v) for v in bbox)))


def marker_to_data_url(path: Path, bbox: list[float]) -> str:
    with Image.open(path) as image:
        canvas = image.convert('RGB')
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(tuple(int(v) for v in bbox), outline=(255, 0, 0), width=4)
        return pil_to_data_url(canvas)


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def write_examples(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        'success_drop_fp.md': lambda r: r['score_baseline_020'] == 'keep' and r['final_decision'] == 'drop' and not r.get('gt_correct', False),
        'success_keep_tp.md': lambda r: r['final_decision'] == 'keep' and r.get('gt_correct', False),
        'fallback_case.md': lambda r: bool(r.get('fallback_used')),
    }
    for filename, predicate in cases.items():
        row = next((item for item in rows if predicate(item)), None)
        if row is None:
            (out_dir / filename).write_text('# Example Proposal Trace\n\nNo matching case found.\n', encoding='utf-8')
            continue
        text = f"""# Example Proposal Trace

## Teacher Proposal
- image_id: `{row['image_id']}`
- proposal_id: `{row['proposal_id']}`
- class: `{row['predicted_class']}`
- score: `{float(row['score']):.4f}`
- bbox: `{row['bbox']}`

## Proposal Verification Output
- visual_validity: `{row['visual_validity']}`
- evidence_strength: `{row['evidence_strength']}`
- recommended_action: `{row['recommended_action']}`
- inspection_status: `{row['inspection_status']}`

## LLM Final Decision Output
- llm_final_status: `{row['llm_final_status']}`
- llm_final_decision: `{row['llm_final_decision']}`
- llm_final_confidence: `{row['llm_final_confidence']}`

## Final Action
- final_decision: `{row['final_decision']}`
- final_decision_source: `{row['final_decision_source']}`
- fallback_used: `{row['fallback_used']}`

## Offline Evaluation
- gt_correct: `{row.get('gt_correct')}`
- eval_label: `{row.get('tp_fp_fn_label_for_eval')}`
"""
        (out_dir / filename).write_text(text, encoding='utf-8')
