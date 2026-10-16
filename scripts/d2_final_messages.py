"""Write each D2 run's final message (finish message, else last assistant text) to JSON.

Usage: python scripts/d2_final_messages.py ROWS.jsonl OUT.json FIRST_ROW_INDEX
"""

import json
import sys

from agent_trace_integrity.adapters import load


def main() -> None:
    src, out_path, base = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out = []
    for i, run in enumerate(load(src, "openhands")):
        fin = [e for e in run if e.role == "tool_call" and e.tool_name == "finish"]
        if fin and isinstance(fin[-1].tool_args, dict) and fin[-1].tool_args.get("message"):
            text, idx, origin = fin[-1].tool_args["message"], fin[-1].index, "finish.message"
        else:
            a = [e for e in run if e.role == "assistant" and (e.text or "").strip()]
            if not a:
                continue
            text, idx, origin = a[-1].text, a[-1].index, "assistant"
        out.append({"id": f"d2-{base + i}", "trajectory_id": run[0].meta.get("trajectory_id"),
                    "event_index": idx, "origin": origin, "text": text})
    json.dump(out, open(out_path, "w"), indent=1)
    print(len(out))


if __name__ == "__main__":
    main()
