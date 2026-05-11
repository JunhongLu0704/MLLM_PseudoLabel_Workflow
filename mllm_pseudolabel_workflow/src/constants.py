from __future__ import annotations

VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

REFERENCE_OLD_256 = {
    'num_total_proposals': 1921,
    'num_valid_proposals': 1776,
    'score_baseline_020': {'tp': 124, 'fp': 559, 'fn': 74},
    'deterministic_gate': {'tp': 124, 'fp': 253, 'fn': 74},
}

DEFAULT_DATA_PATH = 'data/pseudo_samples_256.json'
DEFAULT_PROVIDER_BASE_URL = 'http://mllm-provider.example/v1'
DEFAULT_MODEL = 'Qwen3.6-27B-UD-Q4_K_XL.gguf'
DEFAULT_SOURCE_VOC_ROOT = '/path/to/VOC2012'
