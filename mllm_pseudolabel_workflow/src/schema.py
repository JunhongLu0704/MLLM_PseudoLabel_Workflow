from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .constants import VOC_CLASSES


@dataclass
class ValidationResult:
    ok: bool
    parse_status: str
    value: dict[str, Any]
    error: str = ''
    repair_used: bool = False


def parse_json_object(text: str) -> ValidationResult:
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            return ValidationResult(False, 'failed', {}, 'json root is not object')
        return ValidationResult(True, 'ok', value)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                if isinstance(value, dict):
                    return ValidationResult(True, 'repaired', value, repair_used=True)
            except json.JSONDecodeError as exc:
                return ValidationResult(False, 'failed', {}, f'json parse failed: {exc}', repair_used=True)
        return ValidationResult(False, 'failed', {}, 'json parse failed')


def validate_scene_prior(text: str) -> ValidationResult:
    parsed = parse_json_object(text)
    if not parsed.ok:
        return parsed
    value = parsed.value
    image_style = str(value.get('image_style', 'uncertain'))
    confidence = str(value.get('confidence', 'low'))
    if image_style not in {'clipart', 'natural', 'mixed', 'uncertain'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid image_style: {image_style}', parsed.repair_used)
    if confidence not in {'low', 'medium', 'high'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid confidence: {confidence}', parsed.repair_used)
    visible = _voc_list(value.get('visible_object_hints', []))
    unlikely = _voc_list(value.get('unlikely_classes', []))
    if visible is None or unlikely is None:
        return ValidationResult(False, parsed.parse_status, {}, 'non-VOC class in scene prior', parsed.repair_used)
    confusions = value.get('potential_confusions', [])
    if not isinstance(confusions, list):
        confusions = []
    clean_confusions = []
    for item in confusions:
        if not isinstance(item, dict):
            continue
        class_a = str(item.get('class_a', ''))
        class_b = str(item.get('class_b', ''))
        if class_a in VOC_CLASSES and class_b in VOC_CLASSES:
            clean_confusions.append({'class_a': class_a, 'class_b': class_b, 'reason': str(item.get('reason', ''))[:240]})
    return ValidationResult(
        True,
        parsed.parse_status,
        {
            'scene_type': str(value.get('scene_type', 'uncertain'))[:120],
            'image_style': image_style,
            'visible_object_hints': visible,
            'unlikely_classes': unlikely,
            'potential_confusions': clean_confusions,
            'review_guidance': [str(v)[:240] for v in value.get('review_guidance', []) if isinstance(v, str)][:8],
            'confidence': confidence,
        },
        repair_used=parsed.repair_used,
    )


def validate_inspection(text: str, proposal_id: int) -> ValidationResult:
    parsed = parse_json_object(text)
    if not parsed.ok:
        return parsed
    value = parsed.value
    if int(value.get('proposal_id', -999999)) != int(proposal_id):
        return ValidationResult(False, parsed.parse_status, {}, 'proposal_id mismatch', parsed.repair_used)
    validity = str(value.get('visual_validity', 'unavailable'))
    evidence = str(value.get('evidence_strength', 'unavailable'))
    action = str(value.get('recommended_action', 'unavailable'))
    if validity not in {'valid', 'uncertain', 'invalid', 'unavailable'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid visual_validity: {validity}', parsed.repair_used)
    if evidence not in {'weak', 'medium', 'strong', 'unavailable'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid evidence_strength: {evidence}', parsed.repair_used)
    if action not in {'keep', 'low_weight', 'drop', 'unavailable'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid recommended_action: {action}', parsed.repair_used)
    return ValidationResult(True, parsed.parse_status, {'proposal_id': str(proposal_id), 'visual_validity': validity, 'evidence_strength': evidence, 'recommended_action': action, 'inspection_status': 'success'}, repair_used=parsed.repair_used)


def validate_final_decision(text: str, proposal_id: int) -> ValidationResult:
    parsed = parse_json_object(text)
    if not parsed.ok:
        return parsed
    value = parsed.value
    if str(value.get('proposal_id', '')) != str(proposal_id):
        return ValidationResult(False, parsed.parse_status, {}, 'proposal_id mismatch', parsed.repair_used)
    decision = str(value.get('decision', ''))
    confidence = str(value.get('confidence', ''))
    used = value.get('used_signals', [])
    if decision not in {'keep', 'low_weight', 'drop'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid decision: {decision}', parsed.repair_used)
    if confidence not in {'low', 'medium', 'high'}:
        return ValidationResult(False, parsed.parse_status, {}, f'invalid confidence: {confidence}', parsed.repair_used)
    if not isinstance(used, list):
        return ValidationResult(False, parsed.parse_status, {}, 'used_signals is not list', parsed.repair_used)
    return ValidationResult(True, parsed.parse_status, {'proposal_id': str(proposal_id), 'decision': decision, 'confidence': confidence, 'reason': str(value.get('reason', ''))[:240], 'used_signals': [str(v)[:80] for v in used][:8]}, repair_used=parsed.repair_used)


def validate_optional_tool_plan(text: str, proposal_id: int) -> ValidationResult:
    parsed = parse_json_object(text)
    if not parsed.ok:
        return parsed
    value = parsed.value
    if str(value.get('proposal_id', '')) != str(proposal_id):
        return ValidationResult(False, parsed.parse_status, {}, 'proposal_id mismatch', parsed.repair_used)
    selected = value.get('selected_tools', [])
    if not isinstance(selected, list):
        return ValidationResult(False, parsed.parse_status, {}, 'selected_tools is not list', parsed.repair_used)
    allowed = {'retrieve_same_class_image_crops', 'retrieve_class_exemplars'}
    clean = []
    seen = set()
    for item in selected:
        if not isinstance(item, dict):
            return ValidationResult(False, parsed.parse_status, {}, 'selected_tools item is not object', parsed.repair_used)
        tool_name = str(item.get('tool_name', ''))
        if tool_name not in allowed:
            return ValidationResult(False, parsed.parse_status, {}, f'invalid tool_name: {tool_name}', parsed.repair_used)
        if tool_name in seen:
            continue
        arguments = item.get('arguments', {})
        if not isinstance(arguments, dict):
            return ValidationResult(False, parsed.parse_status, {}, 'tool arguments must be object', parsed.repair_used)
        clean_args: dict[str, Any] = {}
        if tool_name == 'retrieve_class_exemplars':
            class_name = str(arguments.get('class_name', value.get('predicted_class', '')))
            if class_name and class_name not in VOC_CLASSES:
                return ValidationResult(False, parsed.parse_status, {}, f'invalid class_name: {class_name}', parsed.repair_used)
            if class_name:
                clean_args['class_name'] = class_name
        top_k = arguments.get('top_k')
        if top_k is not None:
            try:
                clean_args['top_k'] = max(1, min(int(top_k), 4))
            except (TypeError, ValueError):
                return ValidationResult(False, parsed.parse_status, {}, 'top_k must be integer', parsed.repair_used)
        clean.append({'tool_name': tool_name, 'arguments': clean_args, 'reason': str(item.get('reason', ''))[:200]})
        seen.add(tool_name)
    if not clean:
        return ValidationResult(False, parsed.parse_status, {}, 'selected_tools is empty', parsed.repair_used)
    return ValidationResult(
        True,
        parsed.parse_status,
        {
            'proposal_id': str(proposal_id),
            'selected_tools': clean[:2],
            'reason': str(value.get('reason', ''))[:240],
        },
        repair_used=parsed.repair_used,
    )


def _voc_list(values: Any) -> list[str] | None:
    if not isinstance(values, list):
        return []
    clean = []
    for value in values:
        item = str(value)
        if item not in VOC_CLASSES:
            return None
        clean.append(item)
    return clean
