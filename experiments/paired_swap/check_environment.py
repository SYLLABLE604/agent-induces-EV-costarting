#!/usr/bin/env python3
"""Check CUDA, library versions, and local model metadata without loading weights."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from hf_backend import load_model_paths, model_files, template_value
from transport import load_inputs

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    config = load_inputs(args.config)[0]
    paths = load_model_paths(args.models, config)
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit(f"Missing Python dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; CPU execution is not permitted")
    if torch.cuda.device_count() < 4:
        raise SystemExit(f"Expected four visible GPUs, found {torch.cuda.device_count()}")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The selected GPU class lacks BF16 support")
    records = {}
    for name, path in paths.items():
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        if model_config.model_type != config["models"][name]["model_type"]:
            raise SystemExit(f"Unexpected model type for {name}: {model_config.model_type}")
        records[name] = {
            "path": str(path),
            "model_type": model_config.model_type,
            "architectures": list(model_config.architectures or []),
            "processor_class": type(processor).__name__,
            "chat_template_present": bool(template_value(processor)),
            "model_file_count": len(model_files(path)),
        }
    print(json.dumps({
        "status": "ok",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(4)],
        "auto_model_class": AutoModelForMultimodalLM.__name__,
        "models": records,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
