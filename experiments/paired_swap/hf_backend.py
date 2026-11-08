#!/usr/bin/env python3
"""Direct local PyTorch/Transformers inference and model identity checks."""
from __future__ import annotations

import gc
import platform
from pathlib import Path
from typing import Any

from protocol import digest
from transport import file_hash, read_json

MODEL_SUFFIXES = {
    ".json", ".jinja", ".model", ".py", ".safetensors", ".tiktoken", ".txt",
}


def load_model_paths(path: Path, config: dict) -> dict[str, Path]:
    value = read_json(path.resolve())
    if not isinstance(value, dict) or set(value) != set(config["models"]):
        raise ValueError("Local model mapping must contain exactly the configured model names")
    result = {}
    for name, raw in value.items():
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Invalid local path for {name}")
        local = Path(raw).expanduser().resolve()
        if not local.is_dir():
            raise ValueError(f"Local model directory does not exist: {local}")
        result[name] = local
    if len(set(result.values())) != len(result):
        raise ValueError("Logical models must use different local directories")
    return result


def model_files(path: Path) -> list[Path]:
    files = [
        item for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in MODEL_SUFFIXES
        and ".cache" not in item.relative_to(path).parts
    ]
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing model config.json in {path}")
    if not any(item.suffix == ".safetensors" for item in files):
        raise ValueError(f"No Safetensors weights found in {path}")
    return sorted(files, key=lambda item: item.relative_to(path).as_posix())


def build_model_lock(name: str, path: Path, specification: dict) -> dict:
    model_config = read_json(path / "config.json")
    actual_type = model_config.get("model_type")
    if actual_type != specification["model_type"]:
        raise ValueError(
            f"Model type for {name} is {actual_type!r}, expected {specification['model_type']!r}"
        )
    records = []
    for item in model_files(path):
        records.append({
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": file_hash(item),
        })
    identity = {
        "logical_name": name,
        "source": specification["source"],
        "model_type": actual_type,
        "files": records,
    }
    identity["content_sha256"] = digest(identity)
    return identity


def build_model_locks(paths: dict[str, Path], config: dict) -> dict[str, dict]:
    return {
        name: build_model_lock(name, paths[name], config["models"][name])
        for name in config["models"]
    }


def verify_model_lock(path: Path, expected: dict, full_hash: bool = False) -> None:
    actual_files = model_files(path)
    actual_names = [item.relative_to(path).as_posix() for item in actual_files]
    expected_names = [record["path"] for record in expected["files"]]
    if actual_names != expected_names:
        raise ValueError(f"Local model file set changed: {path}")
    for item, record in zip(actual_files, expected["files"]):
        if item.stat().st_size != record["size"]:
            raise ValueError(f"Local model file size changed: {item}")
        if (full_hash or item.suffix != ".safetensors") and file_hash(item) != record["sha256"]:
            raise ValueError(f"Local model file content changed: {item}")
    identity = {
        key: expected[key] for key in ("logical_name", "source", "model_type", "files")
    }
    if digest(identity) != expected["content_sha256"]:
        raise ValueError("Frozen model identity is internally inconsistent")


def normalize_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        if (set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise ValueError("Invalid text chat message")
        normalized.append({
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        })
    return normalized


def template_value(processor: Any) -> Any:
    value = getattr(processor, "chat_template", None)
    if value is None and hasattr(processor, "tokenizer"):
        value = getattr(processor.tokenizer, "chat_template", None)
    if value is None:
        raise ValueError("The local processor has no chat template")
    return value


def inspect_runtime(config: dict, paths: dict[str, Path], locks: dict[str, dict],
                    selected_models: list[str]) -> dict:
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install the packages listed in requirements.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    records = {}
    for name in selected_models:
        path = paths[name]
        verify_model_lock(path, locks[name], full_hash=False)
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        records[name] = {
            "content_sha256": locks[name]["content_sha256"],
            "source": config["models"][name]["source"],
            "model_type": getattr(model_config, "model_type", None),
            "architectures": list(getattr(model_config, "architectures", None) or []),
            "processor_class": type(processor).__name__,
            "chat_template_sha256": digest(template_value(processor)),
            "local_path": str(path),
        }
    return {
        "backend": "transformers_local",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": config["dtype"],
        "generation": config["generation"],
        "chat_template_kwargs": config["chat_template_kwargs"],
        "models": records,
    }


class TransformersBackend:
    """One BF16 model loaded on the only GPU visible to this worker."""

    def __init__(self, name: str, path: Path, config: dict, expected_lock: dict,
                 processor: Any = None, model: Any = None, torch_module: Any = None):
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise RuntimeError("PyTorch is not installed") from exc
        self.torch = torch_module
        self.name = name
        self.path = path
        self.config = config
        if processor is None or model is None:
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA is required; CPU fallback is disabled")
            from transformers import AutoModelForMultimodalLM, AutoProcessor
            verify_model_lock(path, expected_lock, full_hash=False)
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            model = AutoModelForMultimodalLM.from_pretrained(
                path,
                dtype=self.torch.bfloat16,
                device_map={"": "cuda:0"},
                local_files_only=True,
                low_cpu_mem_usage=True,
            ).eval()
        self.processor = processor
        self.model = model
        self.device = self.torch.device("cuda:0")
        self.torch.manual_seed(config["generation"]["seed"])
        if hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "manual_seed_all"):
            self.torch.cuda.manual_seed_all(config["generation"]["seed"])

    def generate(self, request: dict) -> dict:
        if request.get("model") != self.name:
            raise ValueError("Request was sent to the wrong local model")
        messages = normalize_messages(request["messages"])
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **request["chat_template_kwargs"],
        )
        if not isinstance(inputs, dict) and not hasattr(inputs, "items"):
            inputs = {"input_ids": inputs}
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Chat template did not produce input_ids")
        input_tokens = int(input_ids.shape[-1])
        maximum_input = request["generation"]["max_input_tokens"]
        if input_tokens > maximum_input:
            raise ValueError(f"Prompt has {input_tokens} tokens; maximum is {maximum_input}")
        maximum_new = request["generation"]["max_new_tokens"]
        kwargs = {
            "do_sample": request["generation"]["do_sample"],
            "max_new_tokens": maximum_new,
            "use_cache": request["generation"]["use_cache"],
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        continuation = sequences[0, input_tokens:]
        output_tokens = int(continuation.shape[-1])
        text = self.processor.decode(continuation, skip_special_tokens=True).strip()
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        eos_values = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
        last = None if output_tokens == 0 else int(continuation[-1].item())
        stopped = last in eos_values
        return {
            "done": True,
            "done_reason": "length" if output_tokens >= maximum_new and not stopped else "stop",
            "message": {"role": "assistant", "content": text},
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if hasattr(self.torch, "cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
