# Lambda runner for the open-weight LLM judge

These scripts run the one LLM judge question in the v0 plan ("is this sentence an asserted claim
about an action the agent took or its result?") on the D1 sentences the rules mark as borderline,
on one Lambda Cloud GPU instance. The judge never runs inside the checker. Everything is stdlib
Python plus one bash script that runs on the instance.

| File | Runs on | Does |
|---|---|---|
| `launch.py` | laptop | `POST /instance-operations/launch`, one instance; writes the id to `local/lambda-instance.json` |
| `serve_vllm.sh` | instance | venv, pinned vLLM, OpenAI-compatible server on `127.0.0.1` only |
| `run_judge.py` | laptop | copies the input JSONL up, queries the server through an ssh tunnel at temperature 0, writes `local/judge-<model>.jsonl`, deletes the remote copy |
| `teardown.py` | laptop | `POST /instance-operations/terminate`, polls `GET /instances` until the id is gone, prints `terminated, verified` |
| `lambda_api.py` | laptop | shared paths, exit codes and the single HTTP helper |
| `example_borderline.jsonl` | laptop | 4 synthetic sentences for dry runs only |

## Guardrails

- **Claude never launches or stops an instance without Matt's "yes" with the instance type and
  expected hours.** A real launch or teardown needs `--i-have-matts-yes`; agents never pass it.
- **Budget cap: $150** of the $2,000 Lambda credits (the rest stays for TRIAD and Gemma 4 work).
  `launch.py` refuses with exit 2 when hours x price per hour is over `--cap` (default 150), in a
  dry run and in a real run.
- The price per hour is passed by hand, read off Lambda's pricing page at launch, and written in
  `docs/build-log.md`. Prices in examples here are placeholders (UNVERIFIED).
- `LAMBDA_API_KEY` lives only in the laptop's environment; an HF token, if ever needed, lives only
  in `HF_TOKEN` on the instance. Neither ever goes in a file in either repo.
- The vLLM server binds to `127.0.0.1` on the instance and is reached only through the tunnel.
- **The instance holds no data after.** `run_judge.py` deletes its remote folder at the end, the
  results live only in the gitignored `local/`, and teardown terminates the whole instance.
- Every session that launched anything ends with `teardown.py` printing `terminated, verified`,
  and a look at the Lambda dashboard.

## Exact sequence

1. **Dry run** (anyone, offline):

   ```
   python scripts/lambda/launch.py --instance-type gpu_1x_a10 --region us-east-1 \
       --ssh-key <key-name> --hours 3 --price-per-hour <from pricing page> --dry-run
   python scripts/lambda/run_judge.py --host ubuntu@<ip> --input local/borderline-d1.jsonl \
       --model Qwen/Qwen3.5-4B --dry-run
   python scripts/lambda/teardown.py --dry-run
   ```

2. **Matt says yes**, in his own words, to the instance type, region, hours and estimated cost the
   dry run printed. Record the price and his yes in `docs/build-log.md`.
3. **Launch** (main session, after the yes): `export LAMBDA_API_KEY=...` in the shell, then the same
   `launch.py` command with `--i-have-matts-yes` instead of `--dry-run`.
4. **Serve**: get the instance IP from the Lambda dashboard, then

   ```
   scp scripts/lambda/serve_vllm.sh ubuntu@<ip>:~/
   ssh ubuntu@<ip> 'nohup bash ~/serve_vllm.sh Qwen/Qwen3.5-4B > ~/vllm.log 2>&1 &'
   ```

5. **Run the judge**: `python scripts/lambda/run_judge.py --host ubuntu@<ip> --input
   local/borderline-d1.jsonl --model Qwen/Qwen3.5-4B`. It waits for the server, then writes
   `local/judge-Qwen_Qwen3.5-4B.jsonl`.
6. **Second judge** (larger model, same instance only if the GPU has the memory, see below): stop
   the first server (`ssh ubuntu@<ip> pkill -f "vllm serve"`), start `serve_vllm.sh` with the
   second model name, and run step 5 again with `--model <name>`.
7. **Copy back**: the results are already in `local/`. Check the line counts match the input and
   that `unparsed` is near zero. Remove anything else left on the instance:
   `ssh ubuntu@<ip> 'rm -rf ~/ati-judge ~/vllm.log ~/serve_vllm.sh'`.
8. **Teardown**: `python scripts/lambda/teardown.py --i-have-matts-yes`.
9. **Verify**: the script must print `terminated, verified`. Then check the Lambda dashboard shows
   no running instance, and note the actual hours and cost in `docs/build-log.md`.

## Check at launch (UNVERIFIED today)

- Instance type names, regions, capacity and price per hour, on Lambda's pricing page and dashboard.
- The API base URL and auth style in `lambda_api.py` (`https://cloud.lambdalabs.com/api/v1`, Basic
  auth with the key as the user name; Lambda has also used `https://cloud.lambda.ai/api/v1`).
- The model ids `Qwen/Qwen3.5-4B` and the larger judge (default suggestion `Qwen/Qwen3-32B`), and
  their licences.
- The vLLM pin in `serve_vllm.sh` (`VLLM_VERSION`) supports the chosen model.
- GPU memory: a 4B model fits a 24 GB A10; a 32B model in bf16 needs an 80 GB A100 or H100.
- That `chat_template_kwargs.enable_thinking=false` is honoured, so answers are one word.

## Input and output

Input JSONL, one object per line: `{"id": ..., "sentence": ..., "context": ...}`. It holds D1 agent
text, so it lives in `local/` and is never committed. Output lines in `local/judge-<model>.jsonl`:
`{id, model, answer (yes|no|unparsed), raw, temperature, seed, prompt_sha256}`.

## Tests

`tests/test_lambda_dry_run.py` blocks urllib, sockets and subprocess, then checks that every dry
run sends nothing, that cost over the cap exits non-zero, and that a real call without
`--i-have-matts-yes` (or without the key) exits non-zero before any network call.
