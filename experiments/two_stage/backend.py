"""Local BF16 Transformers inference, padded token budgets, hard process watchdog."""
from __future__ import annotations

import copy
import gc
import importlib.metadata
import multiprocessing as mp
import queue
import time
from pathlib import Path

from protocol import digest, file_hash, read_json


def model_identity(path: Path, spec: dict) -> dict:
    cfg = read_json(path / "config.json")
    if cfg.get("model_type") != spec["model_type"]:
        raise ValueError(f"Wrong model_type at {path}: {cfg.get('model_type')}")
    if cfg.get("quantization_config"):
        raise ValueError("Quantized checkpoints are not part of this BF16 protocol")
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts
                   and p.suffix in {".safetensors", ".json", ".jinja", ".model", ".txt", ".py", ".tiktoken"})
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("No Safetensors weights found")
    records = [{"path": p.relative_to(path).as_posix(), "size": p.stat().st_size, "sha256": file_hash(p)} for p in files]
    return {"source": spec["source"], "model_type": spec["model_type"], "files": records, "sha256": digest(records)}


def verify_identity(path: Path, lock: dict) -> None:
    for item in lock["files"]:
        p = path / item["path"]
        if not p.is_file() or p.stat().st_size != item["size"]:
            raise ValueError(f"Model file changed: {p}")
        if p.suffix != ".safetensors" and file_hash(p) != item["sha256"]:
            raise ValueError(f"Model metadata changed: {p}")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*.safetensors")}
    expected = {r["path"] for r in lock["files"] if r["path"].endswith('.safetensors')}
    if actual != expected:
        raise ValueError("Weight file set changed; re-freeze into a new experiment")


class HFBackend:
    def __init__(self, name: str, path: str, config: dict):
        import torch
        import transformers
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A BF16-capable CUDA GPU is required")
        if transformers.__version__ != "5.17.0":
            raise RuntimeError(f"Install pinned transformers==5.17.0, got {transformers.__version__}")
        self.torch, self.config, self.spec = torch, config, config["models"][name]
        torch.manual_seed(config["generation_seed"])
        torch.cuda.manual_seed_all(config["generation_seed"])
        self.processor = transformers.AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        cls = getattr(transformers, self.spec["model_class"])
        self.model = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                        dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                        attn_implementation=config["attention_implementation"]).eval()
        template = getattr(self.processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
        if not template:
            raise ValueError("No chat template in local checkpoint")
        self.generation = copy.deepcopy(self.model.generation_config)
        self.generation.do_sample = False
        self.generation.num_beams = 1
        self.generation.temperature = 1.0
        self.generation.top_p = 1.0
        self.generation.top_k = 50
        self.generation.repetition_penalty = 1.0
        self.generation.max_new_tokens = config["generation"]["max_new_tokens"]
        self.generation.use_cache = config["generation"]["use_cache"]
        self.generation.pad_token_id = tokenizer.pad_token_id
        self.runtime = {
            "engine": "huggingface_transformers", "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "dtype": "bfloat16", "attention": config["attention_implementation"],
            "kernel_packages": {name: self.package_version(name) for name in ("triton", "flash-linear-attention", "causal-conv1d")},
            "processor_class": type(self.processor).__name__, "model_class": type(self.model).__name__,
            "chat_template_sha256": digest(template), "template_kwargs": self.spec["template_kwargs"],
            "effective_generation_config": self.generation.to_dict(),
        }

    @staticmethod
    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def inputs(self, jobs, padding=False):
        chats = [[{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in j["messages"]] for j in jobs]
        return self.processor.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"padding": padding, "truncation": False},
            **self.spec["template_kwargs"])

    def generate(self, jobs):
        # Sorting and chunking only affect execution; job IDs and treatment inputs are frozen.
        lengths = []
        results = {}
        for j in jobs:
            n = int(self.inputs([j])["input_ids"].shape[-1])
            if n > self.config["generation"]["max_input_tokens"]:
                results[j["job_id"]] = {"raw_output": None, "error": "input_too_long", "input_tokens": n, "retryable": False}
            else:
                lengths.append((n, j))
        lengths.sort(key=lambda x: (x[0], x[1]["job_id"]))
        chunk = []
        maximum_new = self.config["generation"]["max_new_tokens"]
        for n, j in lengths:
            if chunk and (n + maximum_new) * (len(chunk) + 1) > self.config["max_padded_batch_tokens"]:
                results.update(self.generate_chunk(chunk))
                chunk = []
            chunk.append(j)
        if chunk:
            results.update(self.generate_chunk(chunk))
        return [results[j["job_id"]] for j in jobs]

    def generate_chunk(self, jobs):
        try:
            inputs = self.inputs(jobs, padding=True).to("cuda:0")
            width = int(inputs["input_ids"].shape[-1])
            input_counts = inputs["attention_mask"].sum(dim=1).tolist()
            start = time.monotonic()
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, generation_config=self.generation)
            self.torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            eos = self.generation.eos_token_id
            ends = set(eos if isinstance(eos, list) else [eos])
            result = {}
            for n, job in enumerate(jobs):
                ids = sequences[n, width:].tolist()
                stop_index = next((k + 1 for k, t in enumerate(ids) if t in ends), len(ids))
                ids = ids[:stop_index]
                result[job["job_id"]] = {
                    "raw_output": self.processor.decode(ids, skip_special_tokens=True).strip(),
                    "error": None, "input_tokens": int(input_counts[n]), "output_tokens": len(ids),
                    "token_limit_reached": len(ids) >= self.generation.max_new_tokens and not (ids and ids[-1] in ends),
                    "microbatch_size": len(jobs), "batch_generation_seconds": elapsed,
                }
            return result
        except self.torch.OutOfMemoryError:
            if "inputs" in locals():
                del inputs
            if "outputs" in locals():
                del outputs
            gc.collect()
            self.torch.cuda.empty_cache()
            if len(jobs) == 1:
                raise
            middle = len(jobs) // 2
            return {**self.generate_chunk(jobs[:middle]), **self.generate_chunk(jobs[middle:])}


def _child(requests, responses, name, path, config):
    try:
        backend = HFBackend(name, path, config)
        responses.put({"ready": backend.runtime})
    except Exception as exc:
        responses.put({"fatal": f"{type(exc).__name__}: {exc}"})
        return
    while True:
        jobs = requests.get()
        if jobs is None:
            return
        try:
            responses.put({"results": backend.generate(jobs)})
        except Exception as exc:
            responses.put({"error": f"{type(exc).__name__}: {exc}"})
            return


class InferenceSession:
    """Persistent per-GPU child; parent can kill a hung CUDA call and restart cleanly."""
    def __init__(self, name, path, config):
        self.name, self.path, self.config = name, path, config
        self.process = None
        self.start()

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.responses.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(f"Inference process exited: {self.process.exitcode}")
        raise TimeoutError(f"Inference exceeded {timeout}s")

    def start(self):
        ctx = mp.get_context("spawn")
        self.requests, self.responses = ctx.Queue(), ctx.Queue()
        self.process = ctx.Process(target=_child, args=(self.requests, self.responses, self.name, str(self.path), self.config))
        self.process.start()
        try:
            answer = self.receive(self.config["model_load_timeout_seconds"])
            if "ready" not in answer:
                raise RuntimeError(answer)
            self.runtime = answer["ready"]
        except BaseException:
            self.close(graceful=False)
            raise

    def generate(self, jobs):
        if not self.process:
            self.start()
        self.requests.put(jobs)
        try:
            answer = self.receive(self.config["batch_timeout_seconds"])
            if "results" not in answer:
                raise RuntimeError(answer.get("error", str(answer)))
            return answer["results"]
        except BaseException:
            self.close(graceful=False)
            raise

    def close(self, graceful=True):
        if self.process:
            if graceful and self.process.is_alive():
                self.requests.put(None)
                self.process.join(timeout=15)
            forced = self.process.is_alive()
            if forced:
                self.process.terminate()
                self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=10)
            for channel in (self.requests, self.responses):
                if forced:
                    channel.cancel_join_thread()
                channel.close()
                if not forced:
                    channel.join_thread()
            self.process.close()
            self.process = None
