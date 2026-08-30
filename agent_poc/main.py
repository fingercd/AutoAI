"""CLI entrypoint: only ``--model-key`` changes the selected backbone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .budget import BudgetController
from .clients.autoai_client import AutoAIClient
from .clients.llm_client import LLMClient
from .health import probe_agent_stack
from .orchestrator import run_agent
from .state import load_agent_config, load_model_registry
from .trace import TraceRecorder, assert_trace_safe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-key')
    parser.add_argument('--dataset-id')
    parser.add_argument('--health-check', action='store_true')
    parser.add_argument('--probe-inference', action='store_true')
    parser.add_argument('--max-runs', type=int)
    parser.add_argument('--models-config', type=Path, default=Path(__file__).parent / 'config' / 'models.toml')
    parser.add_argument('--agent-config', type=Path, default=Path(__file__).parent / 'config' / 'agent.toml')
    parser.add_argument('--trace-path', type=Path)
    parser.add_argument('--autoai-base-url')
    args = parser.parse_args()

    registry = load_model_registry(args.models_config)
    if args.health_check:
        selected = registry
        if args.model_key:
            if args.model_key not in registry:
                parser.error(f'unknown model key: {args.model_key}')
            selected = {args.model_key: registry[args.model_key]}
        if args.probe_inference and not args.model_key:
            parser.error('--probe-inference requires exactly one --model-key')
        cfg = load_agent_config(
            args.agent_config,
            dataset_id='health-check',
        )
        if args.autoai_base_url:
            cfg = cfg.__class__(**{**cfg.__dict__, 'autoai_base_url': args.autoai_base_url})
        autoai = AutoAIClient(
            cfg.autoai_base_url,
            token=os.getenv('AUTOAI_API_TOKEN'),
        )
        try:
            report = probe_agent_stack(
                autoai,
                selected.values(),
                run_inference=args.probe_inference,
            )
        finally:
            autoai.close()
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report['status'] == 'ready' else 1

    if args.probe_inference:
        parser.error('--probe-inference requires --health-check')
    if not args.model_key:
        parser.error('--model-key is required unless --health-check is used')
    if not args.dataset_id:
        parser.error('--dataset-id is required unless --health-check is used')
    if args.model_key not in registry:
        parser.error(f'unknown model key: {args.model_key}')
    cfg = load_agent_config(
        args.agent_config,
        dataset_id=args.dataset_id,
        max_runs=args.max_runs,
        trace_path=args.trace_path,
    )
    if args.autoai_base_url:
        cfg = cfg.__class__(**{**cfg.__dict__, 'autoai_base_url': args.autoai_base_url})
    model_cfg = registry[args.model_key]
    trace = TraceRecorder(cfg.trace_path, model_cfg=model_cfg, code_revision=cfg.code_revision)
    token = os.getenv('AUTOAI_API_TOKEN')
    budget_controller = BudgetController(
        cfg.budget if cfg.modules.budget_control else None
    )
    autoai = AutoAIClient(
        cfg.autoai_base_url,
        token=token,
        budget_controller=budget_controller,
    )
    llm = LLMClient(
        model_cfg,
        temperature=cfg.temperature,
        timeout_seconds=cfg.llm_timeout_seconds,
        budget_controller=budget_controller,
    )
    try:
        result = run_agent(
            cfg,
            llm,
            autoai,
            trace=trace,
            budget_controller=budget_controller,
        )
    except Exception as exc:
        trace.record_error(type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        autoai.close()
    assert_trace_safe(cfg.trace_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
