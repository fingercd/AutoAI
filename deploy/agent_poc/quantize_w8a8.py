#!/usr/bin/env python3
"""Run the one uniform W8A8 recipe and emit an auditable module inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _module_inventory(model: Any) -> dict[str, list[str]]:
    inventory = {
        'linear_targets': [],
        'ignored_lm_head': [],
        'embedding_modules': [],
        'vision_modules': [],
        'hybrid_linear_attention_or_deltanet_modules': [],
        'mtp_modules': [],
    }
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        lower = name.lower()
        if 'embedding' in class_name or 'embed_tokens' in lower:
            inventory['embedding_modules'].append(name)
        if 'vision' in lower or 'visual' in lower:
            inventory['vision_modules'].append(name)
        if 'mtp' in lower or 'multi_token' in lower:
            inventory['mtp_modules'].append(name)
        if any(term in lower for term in ('deltanet', 'linear_attention', 'gated_delta', 'mamba')):
            inventory['hybrid_linear_attention_or_deltanet_modules'].append(name)
        if class_name == 'linear' or class_name.endswith('linear'):
            if lower == 'lm_head' or lower.endswith('.lm_head'):
                inventory['ignored_lm_head'].append(name)
            else:
                inventory['linear_targets'].append(name)
    return inventory


def _load_texts(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line
        if isinstance(value, dict):
            value = value.get('text')
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        if len(texts) >= limit:
            break
    if len(texts) < limit:
        raise ValueError(f'calibration file contains {len(texts)} texts, need {limit}')
    return texts


def _repair_qwen35_serving_config(model_path: Path, output_path: Path) -> dict[str, Any]:
    """Keep the frozen Qwen3.5 multimodal config around the text-only weights.

    ``AutoModelForCausalLM`` in the Transformers 5.x quantization path saves a
    ``Qwen3_5TextConfig`` as the top-level config.  vLLM's language-model-only
    Qwen3.5 loader still needs the original top-level ``qwen3_5`` wrapper and
    its nested text/vision metadata, even though the output contains only the
    language-model weights.  Restore only those frozen metadata fields and
    preserve the generated quantization config and all saved weights.
    """

    source_file = model_path / 'config.json'
    target_file = output_path / 'config.json'
    source = json.loads(source_file.read_text(encoding='utf-8'))
    target = json.loads(target_file.read_text(encoding='utf-8'))
    if source.get('model_type') != 'qwen3_5' or not isinstance(source.get('text_config'), dict):
        return {'status': 'not_needed', 'source_config_sha256': _sha256(source_file)}
    if target.get('model_type') != 'qwen3_5_text':
        return {
            'status': 'already_compatible',
            'source_config_sha256': _sha256(source_file),
            'target_model_type': target.get('model_type'),
        }

    for key in (
        'model_type',
        'text_config',
        'vision_config',
        'image_token_id',
        'video_token_id',
        'vision_start_token_id',
        'vision_end_token_id',
    ):
        if key in source:
            target[key] = source[key]
    # The checkpoint has only the language-model weights after CausalLM
    # quantization, so select vLLM's text-only Qwen3.5 implementation.
    target['architectures'] = ['Qwen3_5ForCausalLM']
    target_file.write_text(json.dumps(target, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {
        'status': 'repaired_from_frozen_source',
        'source_config_sha256': _sha256(source_file),
        'target_model_type': target['model_type'],
        'target_architectures': target['architectures'],
    }


def quantize(model_path: Path, output_path: Path, calibration_path: Path, *, samples: int, max_seq_length: int) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    try:
        from llmcompressor.modifiers.gptq import GPTQModifier
    except ImportError:  # llmcompressor <=0.9 compatibility
        from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    try:
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    except ImportError:  # llmcompressor <=0.9 compatibility
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    output_path.mkdir(parents=True, exist_ok=True)
    texts = _load_texts(calibration_path, samples)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    inventory = _module_inventory(model)
    report: dict[str, Any] = {
        'schema_version': 'autoai-w8a8-report-v1',
        'model_path': str(model_path),
        'output_path': str(output_path),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'recipe': {
            'smoothing_strength': 0.8,
            'targets': 'Linear',
            'scheme': 'W8A8',
            'ignore': ['lm_head'],
            'max_seq_length': max_seq_length,
            'num_calibration_samples': samples,
        },
        'calibration': {
            'path': str(calibration_path),
            'sha256': _sha256(calibration_path),
            'text_count': len(texts),
        },
        'module_inventory_before': inventory,
        'torch_version': torch.__version__,
        'status': 'started',
    }
    (output_path / 'quantization_preflight.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets='Linear', scheme='W8A8', ignore=['lm_head']),
    ]
    calibration_dataset = Dataset.from_dict({'text': texts})
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=samples,
    )
    # The checkpoint is loaded with device_map='auto'.  Transformers 5.x
    # otherwise tries to reconstruct original-format shards while some
    # converted tensors remain offloaded, which fails for the 27B checkpoints.
    # Keep the compressed W8A8 representation and use bounded shards directly.
    model.save_pretrained(
        output_path,
        save_compressed=True,
        save_original_format=False,
        max_shard_size='10GB',
    )
    tokenizer.save_pretrained(output_path)
    report['serving_config'] = _repair_qwen35_serving_config(model_path, output_path)
    report['status'] = 'passed'
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output_path / 'quantization_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-file', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    args = parser.parse_args()
    report = quantize(
        args.model,
        args.output,
        args.calibration_file,
        samples=args.num_samples,
        max_seq_length=args.max_seq_length,
    )
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
