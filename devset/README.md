# Phase 2 dev set: claim finder precision

60 final messages from **nebius/SWE-rebench-openhands-trajectories** (CC-BY-4.0; Trofimova et
al., Nebius, 2025, <https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories>),
train split rows 10000 to 10059 (offset range A, used only here). Derived text is redistributed
under CC-BY-4.0 with this attribution. `source_pins.json` holds the fetch URLs and sha256.

- `dev_msgs.json`: each run's final message (the `finish` call's message, else the last
  assistant text), made by `scripts/d2_final_messages.py`.
- `LABEL_GUIDE.md`: the guide the annotators followed. Neither annotator saw the finder's patterns.
- `labels_A.json`, `labels_B.json`: two independent annotation passes (Claude, Opus). B worked
  last to first.
- `adjudicated.json`: a third pass settled each claim only one annotator marked.
- `report.json`: output of `python scripts/dev_eval.py devset/dev_msgs.json devset/labels_A.json
  devset/labels_B.json --adjudicated devset/adjudicated.json`.

Result (2026-10-15): span agreement F1 0.819 between annotators. Finder precision **0.937**
(237 of 253, Wilson 95% 0.900 to 0.961; it was 0.956 before passive-voice claims were added in
Phase 5) against the adjudicated reference of 465 claims. Recall of
reference claims is 0.503: the finder is built to be precise, not complete. The patterns were
written against a separate tuning range (rows 30000 to 30099) and were not changed after this
score was taken.
