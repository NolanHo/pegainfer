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
    parser.add_argument(
        "--base-model-name",
        help="OpenAI model name for no-LoRA requests; defaults to --model-path.",
    )
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--logprob-mean-tol", type=float, default=0.08)
    parser.add_argument("--logprob-max-tol", type=float, default=0.30)
    parser.add_argument("--lora-delta-mean-tol", type=float, default=0.05)
    parser.add_argument("--lora-delta-max-tol", type=float, default=0.12)
    parser.add_argument("--min-hf-logit-delta", type=float, default=1.0e-6)
    parser.add_argument("--min-selected-lora-delta", type=float, default=0.01)
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
            base_output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        lora_logits = model(**inputs).logits[:, -1, :].float()
        logit_max_abs_diff = (lora_logits - base_logits).abs().max().item()
        lora_output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    base_tokens = base_output[0, inputs["input_ids"].shape[-1] :].tolist()
    lora_tokens = lora_output[0, inputs["input_ids"].shape[-1] :].tolist()
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    lora_text = tokenizer.decode(lora_tokens, skip_special_tokens=True)
    base_selected_logprobs = hf_selected_logprobs(
        torch,
        model,
        inputs["input_ids"],
        base_tokens,
        adapter_enabled=False,
    )
    lora_selected_logprobs = hf_selected_logprobs(
        torch,
        model,
        inputs["input_ids"],
        lora_tokens,
        adapter_enabled=True,
    )
    base_logprobs_on_lora_tokens = hf_selected_logprobs(
        torch,
        model,
        inputs["input_ids"],
        lora_tokens,
        adapter_enabled=False,
    )

    del model
    del base
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "base": {
            "text": base_text,
            "token_ids": base_tokens,
            "selected_logprobs": base_selected_logprobs,
        },
        "lora": {
            "text": lora_text,
            "token_ids": lora_tokens,
            "selected_logprobs": lora_selected_logprobs,
            "base_selected_logprobs_on_lora_tokens": base_logprobs_on_lora_tokens,
        },
        "logit_max_abs_diff_vs_base": logit_max_abs_diff,
    }


@contextlib.contextmanager
def maybe_disable_adapter(model, adapter_enabled: bool):
    if adapter_enabled:
        yield
    else:
        with model.disable_adapter():
            yield


def hf_selected_logprobs(
    torch,
    model,
    prompt_ids,
    generated_token_ids: list[int],
    adapter_enabled: bool,
) -> list[float]:
    logprobs = []
    with torch.no_grad():
        with maybe_disable_adapter(model, adapter_enabled):
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


def tokenize_completion_text(model_path: Path, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer.encode(text, add_special_tokens=False)


def extract_pegainfer_tokens_and_logprobs(
    choice: dict,
    model_path: Path | None = None,
) -> tuple[list[int], list[float]]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(f"completion choice has no logprobs payload: {choice!r}")

    token_ids = logprobs.get("token_ids")
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(token_ids, list) and isinstance(token_logprobs, list):
        if len(token_ids) != len(token_logprobs):
            raise RuntimeError(f"mismatched token/logprob counts: {logprobs!r}")
        return [int(token_id) for token_id in token_ids], [float(lp) for lp in token_logprobs]

    tokens = logprobs.get("tokens")
    if isinstance(tokens, list) and isinstance(token_logprobs, list):
        if len(tokens) != len(token_logprobs):
            raise RuntimeError(f"mismatched token/logprob counts: {logprobs!r}")
        if all(isinstance(token, int) or str(token).isdigit() for token in tokens):
            return [int(token) for token in tokens], [float(lp) for lp in token_logprobs]
        if model_path is None:
            raise RuntimeError(
                "completion logprobs contain token text but no token_ids; model_path is required"
            )
        parsed_token_ids = tokenize_completion_text(model_path, str(choice.get("text", "")))
        if len(parsed_token_ids) != len(token_logprobs):
            raise RuntimeError(
                "tokenized completion text does not match logprob count: "
                f"token_ids={parsed_token_ids!r}, logprobs={logprobs!r}"
            )
        return parsed_token_ids, [float(lp) for lp in token_logprobs]

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


def signed_logprob_deltas(lora_logprobs: list[float], base_logprobs: list[float]) -> list[float]:
    if len(lora_logprobs) != len(base_logprobs):
        raise RuntimeError(
            "mismatched LoRA/base logprob counts: "
            f"lora={len(lora_logprobs)} base={len(base_logprobs)}"
        )
    return [lora - base for lora, base in zip(lora_logprobs, base_logprobs)]


def delta_distribution(values: list[float]) -> dict:
    magnitudes = [abs(value) for value in values]
    sorted_magnitudes = sorted(magnitudes)
    mean = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
    return {
        "mean_abs": mean,
        "p50_abs": percentile(sorted_magnitudes, 0.50),
        "p99_abs": percentile(sorted_magnitudes, 0.99),
        "max_abs": max(magnitudes) if magnitudes else 0.0,
        "signed": values,
    }


def lora_delta_alignment_stats(
    hf_lora_logprobs: list[float],
    hf_base_logprobs: list[float],
    pegainfer_lora_logprobs: list[float],
    pegainfer_base_logprobs: list[float],
) -> dict:
    hf_delta = signed_logprob_deltas(hf_lora_logprobs, hf_base_logprobs)
    pegainfer_delta = signed_logprob_deltas(pegainfer_lora_logprobs, pegainfer_base_logprobs)
    alignment_error = [hf - peg for hf, peg in zip(hf_delta, pegainfer_delta)]
    return {
        "hf_lora_vs_base": delta_distribution(hf_delta),
        "pegainfer_lora_vs_base": delta_distribution(pegainfer_delta),
        "alignment_error": delta_distribution(alignment_error),
    }


def first_choice(response: dict, label: str) -> dict:
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError(f"{label} response has no choices: {response}")
    return choices[0]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    model_path = Path(args.model_path).resolve()
    base_model_name = args.base_model_name or args.model_path
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
            base_completion_before = pegainfer_completion(
                server_url,
                model_name=base_model_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
            lora_completion = pegainfer_completion(
                server_url,
                model_name=args.lora_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
            base_completion_after = pegainfer_completion(
                server_url,
                model_name=base_model_name,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
        except Exception:  # noqa: BLE001
            print(tail_server_output(process), file=sys.stderr)
            raise
        finally:
            stop_server(process)

    base_choice_before = first_choice(base_completion_before, "base-before")
    lora_choice = first_choice(lora_completion, "lora")
    base_choice_after = first_choice(base_completion_after, "base-after")
    pegainfer_base_text_before = base_choice_before.get("text", "")
    pegainfer_lora_text = lora_choice.get("text", "")
    pegainfer_base_text_after = base_choice_after.get("text", "")
    pegainfer_base_token_ids_before, pegainfer_base_logprobs_before = (
        extract_pegainfer_tokens_and_logprobs(base_choice_before, model_path)
    )
    pegainfer_lora_token_ids, pegainfer_lora_logprobs = extract_pegainfer_tokens_and_logprobs(
        lora_choice,
        model_path,
    )
    pegainfer_base_token_ids_after, pegainfer_base_logprobs_after = (
        extract_pegainfer_tokens_and_logprobs(base_choice_after, model_path)
    )
    base_mismatch_before = first_token_mismatch(
        hf["base"]["token_ids"],
        pegainfer_base_token_ids_before,
    )
    lora_mismatch = first_token_mismatch(hf["lora"]["token_ids"], pegainfer_lora_token_ids)
    base_mismatch_after = first_token_mismatch(
        hf["base"]["token_ids"],
        pegainfer_base_token_ids_after,
    )
    base_logprob_stats_before = logprob_delta_stats(
        hf["base"]["selected_logprobs"],
        pegainfer_base_logprobs_before,
    )
    lora_logprob_stats = logprob_delta_stats(
        hf["lora"]["selected_logprobs"],
        pegainfer_lora_logprobs,
    )
    base_logprob_stats_after = logprob_delta_stats(
        hf["base"]["selected_logprobs"],
        pegainfer_base_logprobs_after,
    )
    hf_lora_delta = delta_distribution(
        signed_logprob_deltas(
            hf["lora"]["selected_logprobs"],
            hf["lora"]["base_selected_logprobs_on_lora_tokens"],
        )
    )
    base_and_lora_share_tokens = pegainfer_base_token_ids_before == pegainfer_lora_token_ids
    lora_delta_alignment = None
    if base_and_lora_share_tokens:
        lora_delta_alignment = lora_delta_alignment_stats(
            hf["lora"]["selected_logprobs"],
            hf["lora"]["base_selected_logprobs_on_lora_tokens"],
            pegainfer_lora_logprobs,
            pegainfer_base_logprobs_before,
        )
    hf_trace_sensitive = (
        hf["base"]["token_ids"] != hf["lora"]["token_ids"]
        or hf_lora_delta["max_abs"] >= args.min_selected_lora_delta
    )
    pegainfer_trace_sensitive = (
        not base_and_lora_share_tokens
        or (
            lora_delta_alignment is not None
            and lora_delta_alignment["pegainfer_lora_vs_base"]["max_abs"]
            >= args.min_selected_lora_delta
        )
    )
    lora_delta_aligned = (
        lora_delta_alignment is None
        or (
            lora_delta_alignment["alignment_error"]["mean_abs"] <= args.lora_delta_mean_tol
            and lora_delta_alignment["alignment_error"]["max_abs"] <= args.lora_delta_max_tol
        )
    )
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
        "hf_base_text": hf["base"]["text"],
        "hf_base_token_ids": hf["base"]["token_ids"],
        "hf_base_selected_logprobs": hf["base"]["selected_logprobs"],
        "hf_lora_text": hf["lora"]["text"],
        "hf_lora_token_ids": hf["lora"]["token_ids"],
        "hf_lora_selected_logprobs": hf["lora"]["selected_logprobs"],
        "hf_base_selected_logprobs_on_lora_tokens": hf["lora"][
            "base_selected_logprobs_on_lora_tokens"
        ],
        "hf_lora_delta": hf_lora_delta,
        "hf_logit_max_abs_diff_vs_base": hf["logit_max_abs_diff_vs_base"],
        "peft_autocast_adapter_dtype": peft_autocast_adapter_dtype,
        "load_response": load_response,
        "base_model_name": base_model_name,
        "pegainfer_base_text_before": pegainfer_base_text_before,
        "pegainfer_base_token_ids_before": pegainfer_base_token_ids_before,
        "pegainfer_base_selected_logprobs_before": pegainfer_base_logprobs_before,
        "pegainfer_lora_text": pegainfer_lora_text,
        "pegainfer_lora_token_ids": pegainfer_lora_token_ids,
        "pegainfer_lora_selected_logprobs": pegainfer_lora_logprobs,
        "pegainfer_base_text_after": pegainfer_base_text_after,
        "pegainfer_base_token_ids_after": pegainfer_base_token_ids_after,
        "pegainfer_base_selected_logprobs_after": pegainfer_base_logprobs_after,
        "base_logprob_delta_before": base_logprob_stats_before,
        "lora_logprob_delta": lora_logprob_stats,
        "base_logprob_delta_after": base_logprob_stats_after,
        "lora_delta_alignment": lora_delta_alignment,
        "tolerances": {
            "logprob_mean": args.logprob_mean_tol,
            "logprob_max": args.logprob_max_tol,
            "lora_delta_mean": args.lora_delta_mean_tol,
            "lora_delta_max": args.lora_delta_max_tol,
            "min_hf_logit_delta": args.min_hf_logit_delta,
            "min_selected_lora_delta": args.min_selected_lora_delta,
        },
        "base_first_token_mismatch_before": base_mismatch_before,
        "lora_first_token_mismatch": lora_mismatch,
        "base_first_token_mismatch_after": base_mismatch_after,
        "hf_trace_sensitive": hf_trace_sensitive,
        "pegainfer_trace_sensitive": pegainfer_trace_sensitive,
        "lora_delta_aligned": lora_delta_aligned,
        "match": base_mismatch_before is None
        and lora_mismatch is None
        and base_mismatch_after is None
        and base_logprob_stats_before["mean"] <= args.logprob_mean_tol
        and base_logprob_stats_before["max"] <= args.logprob_max_tol
        and lora_logprob_stats["mean"] <= args.logprob_mean_tol
        and lora_logprob_stats["max"] <= args.logprob_max_tol
        and base_logprob_stats_after["mean"] <= args.logprob_mean_tol
        and base_logprob_stats_after["max"] <= args.logprob_max_tol
        and hf["logit_max_abs_diff_vs_base"] >= args.min_hf_logit_delta
        and hf_trace_sensitive
        and pegainfer_trace_sensitive
        and lora_delta_aligned,
    }
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    print(summary_json)
    if args.json_out:
        Path(args.json_out).write_text(f"{summary_json}\n", encoding="utf-8")

    if base_mismatch_before is not None:
        print(tail_server_output(process), file=sys.stderr)
        print(f"base-before token mismatch: {base_mismatch_before}", file=sys.stderr)
        return 1
    if lora_mismatch is not None:
        print(tail_server_output(process), file=sys.stderr)
        print(f"lora token mismatch: {lora_mismatch}", file=sys.stderr)
        return 1
    if base_mismatch_after is not None:
        print(tail_server_output(process), file=sys.stderr)
        print(f"base-after token mismatch: {base_mismatch_after}", file=sys.stderr)
        return 1
    for label, stats in [
        ("base-before", base_logprob_stats_before),
        ("lora", lora_logprob_stats),
        ("base-after", base_logprob_stats_after),
    ]:
        if stats["mean"] > args.logprob_mean_tol:
            print(
                f"{label} logprob mean delta {stats['mean']:.6f} "
                f"exceeds {args.logprob_mean_tol:.6f}",
                file=sys.stderr,
            )
            return 1
        if stats["max"] > args.logprob_max_tol:
            print(
                f"{label} logprob max delta {stats['max']:.6f} "
                f"exceeds {args.logprob_max_tol:.6f}",
                file=sys.stderr,
            )
            return 1
    if lora_delta_alignment is not None:
        alignment = lora_delta_alignment["alignment_error"]
        if alignment["mean_abs"] > args.lora_delta_mean_tol:
            print(
                f"LoRA-vs-base delta mean error {alignment['mean_abs']:.6f} "
                f"exceeds {args.lora_delta_mean_tol:.6f}",
                file=sys.stderr,
            )
            return 1
        if alignment["max_abs"] > args.lora_delta_max_tol:
            print(
                f"LoRA-vs-base delta max error {alignment['max_abs']:.6f} "
                f"exceeds {args.lora_delta_max_tol:.6f}",
                file=sys.stderr,
            )
            return 1
    if not hf_trace_sensitive:
        print(
            "HF trace is not LoRA-sensitive: base and LoRA tokens match and selected-logprob "
            f"delta max {hf_lora_delta['max_abs']:.6f} is below {args.min_selected_lora_delta:.6f}",
            file=sys.stderr,
        )
        return 1
    if not pegainfer_trace_sensitive:
        print(
            "pegainfer trace is not LoRA-sensitive: base and LoRA tokens match and selected-logprob "
            f"delta max {lora_delta_alignment['pegainfer_lora_vs_base']['max_abs']:.6f} "
            f"is below {args.min_selected_lora_delta:.6f}",
            file=sys.stderr,
        )
        return 1
    if hf["logit_max_abs_diff_vs_base"] < args.min_hf_logit_delta:
        print("adapter did not change HF logits", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
