#!/usr/bin/env python3
"""Loopback Qwen3.5-9B service with explicit V100/A100 runtime selection."""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import time
import uuid


def hardware_policy(capability):
    if capability[0] < 7:
        raise ValueError("This service requires Volta or newer CUDA hardware")
    return dict(dtype="bfloat16" if capability[0] >= 8 else "float16",
                attention="sdpa", linear_attention="transformers_torch_reference",
                sdpa_math_only=capability[0] < 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--port", type=int, default=18101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must expose only the explicitly selected physical GPU")
    root = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if repo / "work" not in root.parents:
        raise ValueError("Runtime records must stay under Pan/work")
    root.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    import transformers.utils.import_utils as imports
    import transformers.utils as utilities
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one approved GPU must be visible")
    capability = torch.cuda.get_device_capability(0)
    policy = hardware_policy(capability)
    if "sm_" + str(capability[0]) + str(capability[1]) not in torch.cuda.get_arch_list():
        raise ValueError("Installed PyTorch wheel lacks this GPU architecture")
    for optional in ("is_torchvision_available", "is_torchaudio_available", "is_librosa_available",
                     "is_causal_conv1d_available", "is_flash_linear_attention_available"):
        setattr(imports, optional, lambda: False)
        if hasattr(utilities, optional):
            setattr(utilities, optional, lambda: False)
    if policy["sdpa_math_only"]:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    asset = Path("/users/fotile/AutoAI/Agent/.local/shared/models/original/qwen3.5-9b")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(asset, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(asset, local_files_only=True,
        trust_remote_code=False, dtype=getattr(torch, policy["dtype"]),
        device_map={"": "cuda:0"}, attn_implementation=policy["attention"])
    model.eval()
    health = dict(status="ok", model="qwen35_9b", asset=str(asset), dtype=str(model.dtype),
        device=str(model.device), gpu_physical_index=args.gpu,
        gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(capability),
        node=socket.gethostname(), pid=os.getpid(), runtime="transformers",
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        runtime_policy=policy, service_context_window=32768, enable_thinking=False,
        loading_seconds=time.monotonic() - started,
        service_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        asset_metadata={name: hashlib.sha256((asset / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")},
        weight_files={p.name: p.stat().st_size for p in asset.glob("*.safetensors")})
    (root / "service-ready.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(health, ensure_ascii=False), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, health)
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": "not_found"})
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 1048576:
                return self.reply(400, {"error": "invalid_request_size"})
            request = json.loads(self.rfile.read(size))
            expected = dict(model="qwen35_9b", temperature=0, top_p=1, max_tokens=1024,
                            stream=False, chat_template_kwargs={"enable_thinking": False})
            if any(request.get(key) != value for key, value in expected.items()) or request.get("tools"):
                return self.reply(400, {"error": "request_outside_frozen_settings"})
            stamp = time.time()
            started = time.monotonic()
            try:
                rendered = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
                count = inputs.input_ids.shape[-1]
                if count + 1024 > 32768:
                    return self.reply(400, {"error": "context_budget_exceeded", "prompt_tokens": count})
                torch.cuda.synchronize()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
                torch.cuda.synchronize()
                output = generated[0, count:]
                usage = dict(prompt_tokens=count, completion_tokens=len(output), total_tokens=count + len(output))
                content = tokenizer.decode(output, skip_special_tokens=True)
                request_id = "req-" + uuid.uuid4().hex
                measurement = dict(request_id=request_id, started_at_unix=stamp,
                    ended_at_unix=time.time(), server_seconds=time.monotonic() - started,
                    input_sha256=hashlib.sha256(rendered.encode()).hexdigest(), **usage)
                with (root / "service-requests.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(measurement) + "\n")
                self.reply(200, dict(id=request_id, model="qwen35_9b", usage=usage,
                    choices=[dict(index=0, finish_reason="length" if len(output) == 1024 else "stop",
                                  message=dict(role="assistant", content=content))]))
            except Exception as error:
                # Keep technical details in private deployment logs, not arbitrary API output.
                import traceback
                traceback.print_exc()
                self.reply(500, dict(error=type(error).__name__))

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
