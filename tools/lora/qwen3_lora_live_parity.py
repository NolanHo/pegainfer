#!/usr/bin/env python3
"""Live Qwen3 LoRA parity gate against HuggingFace + PEFT.

The script creates a deterministic PEFT-style adapter, obtains the greedy
reference from transformers+peft, loads the same adapter through PegaInfer's
live /v1/load_lora_adapter route, and compares /v1/completions tokens and
selected-token logprobs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--prompt", default="Tell me a story")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server-url")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--lora-name", default="parity")
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--logprob-mean-tol", type=float, default=0.08)
    parser.add_argument("--logprob-max-tol", type=float, default=0.30)
    parser.add_argument("--min-hf-logit-delta", type=float, default=1.0e-6)
    parser.add_argument("--json-out")
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--disable-peft-adapter-autocast",
        action="store_true",
        help="Disable PEFT's default adapter dtype autocast for diagnostics.",
    )
    return parser.parse_args()


def read_config(model_path: Path) -> dict:
    return json.loads((model_path / "config.json").read_text())


def tensor_name(layer_idx: int, path_segment: str, lora_side: str) -> str:
    return f"base_model.model.model.layers.{layer_idx}.{path_segment}.{lora_side}.weight"


def patterned_tensor(torch, shape: tuple[int, ...], seed: int, scale: float):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.empty(shape, dtype=torch.float32)
    tensor.uniform_(-scale, scale, generator=generator)
    return tensor.to(torch.bfloat16)


def write_adapter(model_path: Path, adapter_path: Path, scale: float) -> None:
    from safetensors.torch import save_file
    import torch

    config = read_config(model_path)
    rank = 1
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(model_path),
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "lora_alpha": 1,
                "lora_dropout": 0.0,
                "peft_type": "LORA",
                "r": rank,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            },
            indent=2,
        )
    )

    hidden = int(config["hidden_size"])
    q_out = int(config["num_attention_heads"]) * int(config["head_dim"])
    v_out = int(config["num_key_value_heads"]) * int(config["head_dim"])
    tensors = {}
    for layer_idx in range(int(config["num_hidden_layers"])):
        base_seed = 1000 + layer_idx * 17
        tensors[tensor_name(layer_idx, "self_attn.q_proj", "lora_A")] = patterned_tensor(
            torch, (rank, hidden), base_seed, scale
        )
        tensors[tensor_name(layer_idx, "self_attn.q_proj", "lora_B")] = patterned_tensor(
            torch, (q_out, rank), base_seed + 1, scale
        )
        tensors[tensor_name(layer_idx, "self_attn.v_proj", "lora_A")] = patterned_tensor(
            torch, (rank, hidden), base_seed + 2, scale
        )
        tensors[tensor_name(layer_idx, "self_attn.v_proj", "lora_B")] = patterned_tensor(
            torch, (v_out, rank), base_seed + 3, scale
        )
    save_file(tensors, str(adapter_path / "adapter_model.safetensors"))


def hf_peft_reference(
    model_path: Path,
    adapter_path: Path,
    prompt: str,
    max_tokens: int,
    autocast_adapter_dtype: bool,
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        autocast_adapter_dtype=autocast_adapter_dtype,
    ).eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        with model.disable_adapter():
            base_logits = model(**inputs).logits[:, -1, :].float()
        lora_logits = model(**inputs).logits[:, -1, :].float()
        logit_max_abs_diff = (lora_logits - base_logits).abs().max().item()
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[-1] :].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    selected_logprobs = hf_selected_logprobs(torch, model, inputs["input_ids"], new_tokens)

    del model
    del base
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "text": text,
        "token_ids": new_tokens,
        "selected_logprobs": selected_logprobs,
        "logit_max_abs_diff_vs_base": logit_max_abs_diff,
    }


def hf_selected_logprobs(torch, model, prompt_ids, generated_token_ids: list[int]) -> list[float]:
    logprobs = []
    with torch.no_grad():
        outputs = model(input_ids=prompt_ids, use_cache=True)
        logits = outputs.logits[:, -1, :].float()
        past_key_values = outputs.past_key_values
        for index, token_id in enumerate(generated_token_ids):
            logprob = torch.log_softmax(logits, dim=-1)[0, token_id].item()
            logprobs.append(float(logprob))
            if index + 1 == len(generated_token_ids):
                break
            next_id = torch.tensor([[token_id]], device=prompt_ids.device, dtype=prompt_ids.dtype)
            outputs = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :].float()
            past_key_values = outputs.past_key_values
    return logprobs


def first_token_mismatch(hf_token_ids: list[int], pegainfer_token_ids: list[int]) -> dict | None:
    if hf_token_ids == pegainfer_token_ids:
        return None

    for index, (hf_token_id, pegainfer_token_id) in enumerate(
        zip(hf_token_ids, pegainfer_token_ids),
        start=1,
    ):
        if hf_token_id != pegainfer_token_id:
            return {
                "index_1based": index,
                "hf_token_id": hf_token_id,
                "pegainfer_token_id": pegainfer_token_id,
            }

    return {
        "index_1based": min(len(hf_token_ids), len(pegainfer_token_ids)) + 1,
        "hf_token_id": hf_token_ids[len(pegainfer_token_ids)]
        if len(hf_token_ids) > len(pegainfer_token_ids)
        else None,
        "pegainfer_token_id": pegainfer_token_ids[len(hf_token_ids)]
        if len(pegainfer_token_ids) > len(hf_token_ids)
        else None,
    }


def post_json(url: str, payload: dict, timeout: float = 120.0) -> dict | str:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(body)
    return body


def get(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def wait_for_health(server_url: str, timeout_s: float, process: subprocess.Popen | None) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            get(f"{server_url}/health")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {server_url}/health: {last_error}")


def start_server(args: argparse.Namespace, repo_root: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PEGAINFER_CUDA_SM", "80")
    compat = "/usr/local/cuda-12.9/compat"
    if Path(compat).exists():
        old = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = compat if not old else f"{compat}:{old}"
    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "pegainfer-server",
        "--",
        "--model-path",
        args.model_path,
        "--enable-lora",
        "--tp-size",
        str(args.tp_size),
        "--port",
        str(args.port),
    ]
    log = tempfile.NamedTemporaryFile(
        prefix="pegainfer-qwen3-lora-server-",
        suffix=".log",
        mode="w+",
        delete=False,
    )
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process.pegainfer_log_path = log.name  # type: ignore[attr-defined]
    print(f"server_log={log.name}", file=sys.stderr)
    log.close()
    return process


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def tail_server_output(process: subprocess.Popen | None) -> str:
    if process is None:
        return ""
    log_path = getattr(process, "pegainfer_log_path", None)
    if not log_path:
        return ""
    with contextlib.suppress(Exception):
        return Path(log_path).read_text(errors="replace")[-4000:]
    return ""


def pegainfer_completion(
    server_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    response = post_json(
        f"{server_url}/v1/completions",
        {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": 1,
        },
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"unexpected completion response: {response!r}")
    return response


def extract_pegainfer_tokens_and_logprobs(choice: dict) -> tuple[list[int], list[float]]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(f"completion choice has no logprobs payload: {choice!r}")

    token_ids = logprobs.get("token_ids") or logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(token_ids, list) and isinstance(token_logprobs, list):
        if len(token_ids) != len(token_logprobs):
            raise RuntimeError(f"mismatched token/logprob counts: {logprobs!r}")
        return [int(token_id) for token_id in token_ids], [float(lp) for lp in token_logprobs]

    positions = logprobs.get("positions")
    if isinstance(positions, list):
        parsed_token_ids = []
        parsed_logprobs = []
        for position in positions:
            entries = position.get("entries") if isinstance(position, dict) else None
            if not entries:
                raise RuntimeError(f"logprob position has no entries: {position!r}")
            sampled = entries[0]
            parsed_token_ids.append(int(sampled["token_id"]))
            parsed_logprobs.append(float(sampled["logprob"]))
        return parsed_token_ids, parsed_logprobs

    raise RuntimeError(f"unexpected logprobs payload: {logprobs!r}")


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * pct)))
    return sorted_values[index]


def logprob_delta_stats(hf_logprobs: list[float], pegainfer_logprobs: list[float]) -> dict:
    if len(hf_logprobs) != len(pegainfer_logprobs):
        raise RuntimeError(
            f"mismatched logprob counts: hf={len(hf_logprobs)} pegainfer={len(pegainfer_logprobs)}"
        )
    deltas = [abs(hf - peg) for hf, peg in zip(hf_logprobs, pegainfer_logprobs)]
    sorted_deltas = sorted(deltas)
    mean = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "mean": mean,
        "p50": percentile(sorted_deltas, 0.50),
        "p99": percentile(sorted_deltas, 0.99),
        "max": max(deltas) if deltas else 0.0,
        "deltas": deltas,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_path = Path(args.model_path).resolve()
    if args.adapter_path:
        adapter_path = Path(args.adapter_path).resolve()
        adapter_path.mkdir(parents=True, exist_ok=True)
        cleanup = contextlib.nullcontext(adapter_path)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="pegainfer-qwen3-lora-parity-")

    process = None
    with cleanup as adapter_dir:
        adapter_path = Path(adapter_dir)
        write_adapter(model_path, adapter_path, args.scale)
        peft_autocast_adapter_dtype = not args.disable_peft_adapter_autocast
        hf = hf_peft_reference(
            model_path,
            adapter_path,
            args.prompt,
            args.max_tokens,
            peft_autocast_adapter_dtype,
        )

        server_url = args.server_url or f"http://127.0.0.1:{args.port}"
        if args.server_url is None:
            process = start_server(args, repo_root)
        try:
            wait_for_health(server_url, args.startup_timeout_s, process)
            load_response = post_json(
                f"{server_url}/v1/load_lora_adapter",
                {"lora_name": args.lora_name, "lora_path": str(adapter_path)},
            )
            completion = pegainfer_completion(
                server_url,
                model_name=args.lora_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
        except Exception:  # noqa: BLE001
            print(tail_server_output(process), file=sys.stderr)
            raise
        finally:
            stop_server(process)

    choices = completion.get("choices", [])
    if not choices:
        raise RuntimeError(f"completion response has no choices: {completion}")
    choice = choices[0]
    pegainfer_text = choice.get("text", "")
    pegainfer_token_ids, pegainfer_logprobs = extract_pegainfer_tokens_and_logprobs(choice)
    mismatch = first_token_mismatch(hf["token_ids"], pegainfer_token_ids)
    logprob_stats = logprob_delta_stats(hf["selected_logprobs"], pegainfer_logprobs)
    adapter_spec = {
        "rank": 1,
        "lora_alpha": 1,
        "target_modules": ["q_proj", "v_proj"],
        "scale": args.scale,
        "seed_base": 1000,
        "seed_stride_per_layer": 17,
    }
    summary = {
        "adapter_path": str(adapter_path),
        "adapter_spec": adapter_spec,
        "hf_text": hf["text"],
        "hf_token_ids": hf["token_ids"],
        "hf_selected_logprobs": hf["selected_logprobs"],
        "hf_logit_max_abs_diff_vs_base": hf["logit_max_abs_diff_vs_base"],
        "peft_autocast_adapter_dtype": peft_autocast_adapter_dtype,
        "load_response": load_response,
        "pegainfer_text": pegainfer_text,
        "pegainfer_token_ids": pegainfer_token_ids,
        "pegainfer_selected_logprobs": pegainfer_logprobs,
        "logprob_delta": logprob_stats,
        "tolerances": {
            "logprob_mean": args.logprob_mean_tol,
            "logprob_max": args.logprob_max_tol,
            "min_hf_logit_delta": args.min_hf_logit_delta,
        },
        "first_token_mismatch": mismatch,
        "match": mismatch is None
        and logprob_stats["mean"] <= args.logprob_mean_tol
        and logprob_stats["max"] <= args.logprob_max_tol
        and hf["logit_max_abs_diff_vs_base"] >= args.min_hf_logit_delta,
    }
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    print(summary_json)
    if args.json_out:
        Path(args.json_out).write_text(f"{summary_json}\n", encoding="utf-8")

    if mismatch is not None:
        print(tail_server_output(process), file=sys.stderr)
        print(f"token mismatch: {mismatch}", file=sys.stderr)
        return 1
    if logprob_stats["mean"] > args.logprob_mean_tol:
        print(
            f"logprob mean delta {logprob_stats['mean']:.6f} exceeds {args.logprob_mean_tol:.6f}",
            file=sys.stderr,
        )
        return 1
    if logprob_stats["max"] > args.logprob_max_tol:
        print(
            f"logprob max delta {logprob_stats['max']:.6f} exceeds {args.logprob_max_tol:.6f}",
            file=sys.stderr,
        )
        return 1
    if hf["logit_max_abs_diff_vs_base"] < args.min_hf_logit_delta:
        print("adapter did not change HF logits", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
