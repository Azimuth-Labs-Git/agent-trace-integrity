"""Run the open-weight LLM judge (plan section 3, "The LLM judge") on Modal.

Same question, prompt, temperature, seed and answer parsing as scripts/lambda/run_judge.py, so the
two backends give comparable labels. The judge never runs inside the checker; it only labels
borderline D1 sentences and the PI's 20 hand-checked claims, to report agreement.

Usage (from the repo root):
  modal run scripts/modal_judge.py --dry-run
  modal run scripts/modal_judge.py --input local/borderline-d1.jsonl --model Qwen/Qwen3.5-4B

Input and output stay in the gitignored local/ folder (they hold agent text). The Modal app is
tagged project=azimuth for cost attribution; the model cache lives in its own volume.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
J = None
if modal.is_local():  # the prompt helpers live in the repo; the container gets plain arguments
    _spec = importlib.util.spec_from_file_location("run_judge",
                                                   ROOT / "scripts/lambda/run_judge.py")
    J = importlib.util.module_from_spec(_spec)
    sys.modules["run_judge"] = J
    _spec.loader.exec_module(J)

app = modal.App("azimuth-ati-judge", tags={"project": "azimuth", "repo": "agent-trace-integrity"})
CACHE = modal.Volume.from_name("azimuth-ati-model-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==5.18.0", "torch", "torchvision", "accelerate", "pillow",
                      "huggingface_hub", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"}))


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/cache": CACHE})
def judge(model: str, messages: list[list[dict]], seed: int, temperature: float,
          max_tokens: int) -> dict:
    """Greedy decoding (temperature 0) with transformers, one prompt at a time."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    assert temperature == 0, "the judge is run greedy only"
    tok = AutoTokenizer.from_pretrained(model)
    m = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            m = getattr(transformers, cls).from_pretrained(model, dtype=torch.bfloat16,
                                                           device_map="cuda")
            break
        except (ValueError, KeyError) as exc:
            print(f"{cls} could not load {model}: {exc}"[:300])
    m.eval()
    texts = []
    for msgs in messages:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        texts.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    CACHE.commit()
    return {"texts": texts, "backend_version": f"transformers {transformers.__version__}, "
                                                f"torch {torch.__version__}"}


@app.local_entrypoint()
def main(input: str = "local/borderline-d1.jsonl", model: str = "Qwen/Qwen3.5-4B",
         dry_run: bool = False):
    items = J.load_items(ROOT / input)
    bodies = [J.request_body(model, it) for it in items]
    prompt_sha = hashlib.sha256((J.SYSTEM + J.QUESTION).encode()).hexdigest()
    print(f"model {model}; items {len(items)}; temperature {J.TEMPERATURE}; seed {J.SEED}; "
          f"prompt sha256 {prompt_sha}")
    if dry_run:
        print(json.dumps(bodies[0]["messages"], indent=1)[:1200])
        print("dry run: nothing sent")
        return
    res = judge.remote(model, [b["messages"] for b in bodies], J.SEED, J.TEMPERATURE,
                       J.MAX_TOKENS)
    out = ROOT / "local" / f"judge-{J.model_slug(model)}.jsonl"
    with open(out, "w") as f:
        for it, text in zip(items, res["texts"], strict=True):
            f.write(json.dumps({"id": it["id"], "answer": J.parse_answer(text), "raw": text,
                                "model": model, "backend": "modal, A10G",
                                "backend_version": res["backend_version"],
                                "temperature": J.TEMPERATURE, "seed": J.SEED,
                                "prompt_sha256": prompt_sha}) + "\n")
    print("wrote", out)
