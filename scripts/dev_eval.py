"""Phase 2 dev-set scoring: annotator agreement and claim-finder precision.

Usage:
  python scripts/dev_eval.py MSGS LABELS_A LABELS_B [--adjudicated ADJ] [--disagreements OUT]

MSGS is the dev message file (id, text). Labels are {id: [{quote, category, modality}]}.
A label's span is the first occurrence of its quote in the message. Two spans match when they
overlap. Without --adjudicated, labels found by both annotators form the reference; with it,
the third pass's verdicts on the disagreements are added (keep=true).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

from agent_trace_integrity.claims import claims_in_text


def spans(text, labels):
    out = []
    for lab in labels:
        i = text.find(lab["quote"])
        if i >= 0:
            out.append((i, i + len(lab["quote"]), lab))
    return out


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("msgs")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--adjudicated")
    ap.add_argument("--disagreements")
    args = ap.parse_args()
    msgs = {m["id"]: m["text"] for m in json.load(open(args.msgs))}
    la, lb = json.load(open(args.a)), json.load(open(args.b))
    adj = json.load(open(args.adjudicated)) if args.adjudicated else {}

    both = only_a = only_b = 0
    ref: dict[str, list] = {}
    dis = []
    for mid, text in msgs.items():
        sa, sb = spans(text, la.get(mid, [])), spans(text, lb.get(mid, []))
        r = []
        for x in sa:
            if any(overlap(x, y) for y in sb):
                both += 1
                r.append(x)
            else:
                only_a += 1
                dis.append({"id": mid, "who": "A", **x[2]})
        for y in sb:
            if any(overlap(y, x) for x in sa):
                if not any(overlap(y, z) for z in r):
                    r.append(y)
            else:
                only_b += 1
                dis.append({"id": mid, "who": "B", **y[2]})
        ref[mid] = r
    for i, d in enumerate(dis):
        d["dis_id"] = i
        v = adj.get(str(i))
        if v and v.get("keep"):
            text = msgs[d["id"]]
            j = text.find(d["quote"])
            ref[d["id"]].append((j, j + len(d["quote"]), d))
    if args.disagreements:
        json.dump(dis, open(args.disagreements, "w"), indent=1)

    found = hit = 0
    by_cat = Counter()
    hit_cat = Counter()
    misses = []
    recall_hit = 0
    n_ref = sum(len(v) for v in ref.values())
    for mid, text in msgs.items():
        cs = [c for c in claims_in_text(text)]
        rs = ref[mid]
        for c in cs:
            found += 1
            by_cat[c.category] += 1
            span = (c.char_start, c.char_end)
            ok = any(overlap(span, (s, e)) for s, e, _ in rs)
            hit += ok
            hit_cat[c.category] += ok
            if not ok:
                misses.append({"id": mid, "category": c.category, "text": c.text[:160]})
        for s, e, _ in rs:
            recall_hit += any(overlap((c.char_start, c.char_end), (s, e)) for c in cs)
    agree = 2 * both / ((both + only_a) + (both + only_b)) if both else 0
    report = {
        "messages": len(msgs),
        "annotator_A": both + only_a, "annotator_B": both + only_b,
        "agreed_spans": both, "only_A": only_a, "only_B": only_b,
        "span_agreement_f1": round(agree, 3),
        "adjudicated": len(adj), "adjudicated_kept": sum(1 for v in adj.values() if v.get("keep")),
        "reference_claims": n_ref,
        "finder_claims": found, "finder_hits": hit,
        "precision": round(hit / found, 3) if found else None,
        "precision_wilson95": wilson(hit, found),
        "precision_by_category": {k: f"{hit_cat[k]}/{v}" for k, v in sorted(by_cat.items())},
        "recall_of_reference_sentences": round(recall_hit / n_ref, 3) if n_ref else None,
    }
    print(json.dumps(report, indent=1))
    if misses:
        print(f"\n{len(misses)} finder claims with no reference match:")
        for m in misses:
            print(" ", m["id"], m["category"], "|", m["text"])


if __name__ == "__main__":
    main()
