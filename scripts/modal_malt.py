"""D4, METR MALT, on Modal (the HF token lives only in the Modal secret `huggingface-secret`).

MALT (https://huggingface.co/datasets/metr-evals/malt-transcripts-public) is MIT-licensed and
gated; it carries a do-not-train canary, so it is never copied into the repo or any training data.
Everything runs inside Modal; only counts, field names and run ids come back.

  modal run scripts/modal_malt.py::probe     # structure only: files, columns, label fields
"""

from __future__ import annotations

import json

import modal

app = modal.App("azimuth-ati-malt", tags={"project": "azimuth", "repo": "agent-trace-integrity"})
CACHE = modal.Volume.from_name("azimuth-ati-malt-cache", create_if_missing=True)
# The PI's Hugging Face account with MALT access (his "furycone" account).
HF_SECRET = modal.Secret.from_name("huggingface-secret-proxy-forge")
REPO = "metr-evals/malt-transcripts-public"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("huggingface_hub", "hf_transfer", "pyarrow", "pandas")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"}))


def _shape(v, depth=0):
    """Type sketch of a value with no content: keys, types, lengths."""
    if isinstance(v, dict):
        return {k: _shape(x, depth + 1) for k, x in list(v.items())[:40]} if depth < 4 else "dict"
    if isinstance(v, list):
        return [f"list[{len(v)}]", _shape(v[0], depth + 1) if v else None]
    if isinstance(v, str):
        return f"str[{len(v)}]"
    return type(v).__name__


@app.function(image=image, secrets=[HF_SECRET], volumes={"/cache": CACHE}, timeout=1800)
def probe() -> dict:
    import os

    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    info = api.dataset_info(REPO, files_metadata=True)
    files = [{"path": s.rfilename, "size": s.size} for s in info.siblings]
    out = {"sha": info.sha, "files": files, "card_tags": info.tags}
    import pyarrow.parquet as pq

    pq_files = sorted(f["path"] for f in files if f["path"].endswith(".parquet"))
    if pq_files:
        smallest = min((f for f in files if f["path"] in pq_files), key=lambda f: f["size"] or 0)
        p = hf_hub_download(REPO, smallest["path"], repo_type="dataset", revision=info.sha,
                            token=token)
        t = pq.read_table(p)
        out["probe_file"] = smallest["path"]
        out["rows"] = t.num_rows
        out["schema"] = str(t.schema)[:4000]
        row = t.slice(0, 1).to_pylist()[0]
        out["row_shape"] = {k: _shape(v) for k, v in row.items()}
        small = {}
        for name in t.column_names:
            col = t.column(name)
            vals = col.to_pylist()
            if all(isinstance(x, (str, bool, int, type(None))) for x in vals[:200]):
                distinct = {str(x) for x in vals}
                if len(distinct) <= 40 and all(len(x) <= 60 for x in distinct):
                    small[name] = sorted(distinct)
        out["low_cardinality_columns"] = small
    CACHE.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1)[:12000])


@app.function(image=image, secrets=[HF_SECRET], volumes={"/cache": CACHE}, timeout=3600,
              memory=16384)
def survey(sha: str = "3521e8e22b6af12fea3249443e554b96b6fc4f49") -> dict:
    """Counts only, over the main `data/` split: labels, review status, models, and the shape of
    the message DAG (roles, tool names, branching, how results are marked)."""
    import os
    import re
    from collections import Counter

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    files = sorted(s.rfilename for s in HfApi(token=token).dataset_info(REPO, revision=sha).siblings
                   if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet"))
    lab, lab_rev, models, src, nruns = Counter(), Counter(), Counter(), Counter(), 0
    roles, fnames, branches, res_markers, exitlike = Counter(), Counter(), Counter(), Counter(), 0
    for i, f in enumerate(files):
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=sha, token=token)
        cols = ["labels", "manually_reviewed", "model", "run_source"]
        cols += ["nodes"] if i == 3 else []
        t = pq.read_table(p, columns=cols).to_pylist()
        for r in t:
            nruns += 1
            for lb in r["labels"] or ["(none)"]:
                lab[lb] += 1
                if r["manually_reviewed"]:
                    lab_rev[lb] += 1
            models[r["model"]] += 1
            src[r["run_source"]] += 1
            if i == 3:
                branches[len({n["branch_id"] for n in r["nodes"]})] += 1
                for n in r["nodes"]:
                    m = n["node_data"]["message"] or {}
                    meta = n["node_data"]["metadata"] or {}
                    roles[m.get("role")] += 1
                    if m.get("function_call"):
                        fnames[m["function_call"].get("name")] += 1
                    if meta.get("function_call_result"):
                        res_markers[f"result role={m.get('role')} name={m.get('name')}"] += 1
                    if re.search(r"exit code|returncode|Exit status", m.get("content") or ""):
                        exitlike += 1
    CACHE.commit()
    return {"files": files, "runs": nruns, "labels": dict(lab), "labels_manually_reviewed":
            dict(lab_rev), "models": dict(models), "run_source": dict(src),
            "shard3_roles": dict(roles), "shard3_tool_names": dict(fnames.most_common(30)),
            "shard3_branches_per_run": dict(branches), "shard3_result_markers":
            dict(res_markers.most_common(10)), "shard3_nodes_mentioning_exit_codes": exitlike}


@app.local_entrypoint()
def survey_main():
    print(json.dumps(survey.remote(), indent=1))


@app.function(image=image, secrets=[HF_SECRET], volumes={"/cache": CACHE}, timeout=1800)
def peek(shard: str = "data/transcripts-00003-of-00013.parquet", n: int = 3,
         sha: str = "3521e8e22b6af12fea3249443e554b96b6fc4f49") -> list:
    """Structure of a few runs for adapter design: per node, ids, role, name and a short head of
    the content. Printed for the developer only; never written to the repo."""
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    p = hf_hub_download(REPO, shard, repo_type="dataset", revision=sha, token=token)
    rows = pq.read_table(p, columns=["run_id", "labels", "model", "nodes"]).to_pylist()
    rows = [r for r in rows if "normal" not in (r["labels"] or [])][:n] + rows[:1]
    out = []
    for r in rows:
        nodes = []
        keep = [x for x in r["nodes"] if x["branch_id"] == 0]
        for x in keep[:6] + keep[-6:]:
            m = x["node_data"]["message"] or {}
            fc = m.get("function_call") or {}
            nodes.append([x["node_id"], x["parent_node_id"], x["branch_id"], m.get("role"),
                          m.get("name"), fc.get("name"), (fc.get("arguments") or "")[:80],
                          (m.get("content") or "")[:160].replace("\n", " | ")])
        out.append({"run_id": r["run_id"], "labels": r["labels"], "model": r["model"],
                    "n_nodes": len(r["nodes"]), "nodes": nodes})
    return out


@app.local_entrypoint()
def peek_main(shard: str = "data/transcripts-00003-of-00013.parquet", n: int = 3):
    for r in peek.remote(shard, n):
        print("RUN", r["run_id"], r["labels"], r["model"], r["n_nodes"])
        for x in r["nodes"]:
            print("   ", x)


_PKG = __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "agent_trace_integrity"
check_image = image.add_local_dir(str(_PKG), "/root/agent_trace_integrity")


@app.function(image=check_image, secrets=[HF_SECRET], volumes={"/cache": CACHE}, timeout=7200,
              memory=32768, cpu=4)
def check_all(sha: str = "3521e8e22b6af12fea3249443e554b96b6fc4f49") -> dict:
    """Run the adapter and the full v0 checker on every run in `data/`. Counts per METR label,
    plus run ids for flagged runs; no transcript text leaves the container."""
    import os
    from collections import Counter, defaultdict

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from agent_trace_integrity.adapters.malt import parse_row
    from agent_trace_integrity.events import check_pairing
    from agent_trace_integrity.verdicts import check_run, run_summary

    B, A = "check_modified_before_success_claim", "fabricated_tool_output"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    files = sorted(s.rfilename for s in HfApi(token=token).dataset_info(REPO, revision=sha).siblings
                   if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet"))
    per = defaultdict(Counter)
    flagged = defaultdict(list)
    errors = Counter()
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=sha, token=token)
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                labels = row.get("labels") or ["(none)"]
                try:
                    run = parse_row(row)
                    vs = check_run(run)
                    summ = run_summary(run, vs)
                except Exception as exc:  # count adapter or checker failures, never hide them
                    errors[type(exc).__name__] += 1
                    continue
                calls = sum(e.role == "tool_call" for e in run)
                fb = any(B in v["flags"] for v in vs)
                fa = any(A in v["flags"] for v in vs)
                for lb in labels:
                    c = per[lb]
                    c["runs"] += 1
                    c["runs_with_tool_calls"] += calls > 0
                    c["runs_with_pairing_problems"] += bool(check_pairing(run))
                    c["runs_flag_B"] += fb
                    c["runs_flag_A"] += fa
                    c["check_b_edits"] += len(summ["check_b_edits"])
                    c["test_edits_seen"] += summ["test_edits_seen"]
                    for v in vs:
                        cl = v["claim"]
                        if cl["category"] != "submit" and cl["modality"] == "asserted":
                            c[f"{v['claim']['category']}/{v['verdict']}"] += 1
                    if (fb or fa) and lb != "normal":
                        flagged[lb].append({"run_id": row.get("run_id"), "B": fb, "A": fa})
    return {"sha": sha, "files": len(files), "per_label": {k: dict(v) for k, v in per.items()},
            "flagged_non_normal": dict(flagged), "errors": dict(errors)}


@app.local_entrypoint()
def check_main(out: str = "results/malt-d4.json"):
    res = check_all.remote()
    from pathlib import Path
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("files", "errors")}), "wrote", out)


@app.function(image=check_image, secrets=[HF_SECRET], volumes={"/cache": CACHE}, timeout=3600,
              memory=32768)
def gap(sha: str = "3521e8e22b6af12fea3249443e554b96b6fc4f49", labels: tuple = (
        "bypass_constraints", "hardcoded_solution")) -> list:
    """Developer-only look at runs where the adapter found no tool call: where the actions are."""
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    from agent_trace_integrity.adapters.malt import _path, parse_row

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    files = sorted(s.rfilename for s in HfApi(token=token).dataset_info(REPO, revision=sha).siblings
                   if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet"))
    out = []
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=sha, token=token)
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if not set(row.get("labels") or []) & set(labels):
                    continue
                if any(e.role == "tool_call" for e in parse_row(row)):
                    continue
                path = _path(row["nodes"])
                allb = sorted({n["branch_id"] for n in row["nodes"]})
                fc_any = sum(1 for n in row["nodes"]
                             if ((n["node_data"]["message"] or {}).get("function_call") or {})
                             .get("name"))
                samp = []
                for n in path[:8]:
                    m = n["node_data"]["message"] or {}
                    samp.append([n["node_id"], n["branch_id"], m.get("role"), m.get("name"),
                                 (m.get("content") or "")[:110].replace("\n", " | ")])
                out.append({"run_id": row["run_id"], "labels": row["labels"],
                            "n_nodes": len(row["nodes"]), "path_len": len(path),
                            "branches": allb[:10], "n_branches": len(allb),
                            "nodes_with_function_call": fc_any, "path_head": samp})
                if len(out) >= 3:
                    return out
    return out


@app.local_entrypoint()
def gap_main():
    for r in gap.remote():
        print("RUN", {k: v for k, v in r.items() if k != "path_head"})
        for x in r["path_head"]:
            print("   ", x)
