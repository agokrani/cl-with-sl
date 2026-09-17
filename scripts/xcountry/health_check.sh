#!/bin/bash
# Full-fleet health check. Verifies on DATA signals, not job states: a cell that
# lost --skip-generate exits 0 as COMPLETED having trained on nothing.
set -uo pipefail
RUNROOT=/scratch/agokrani/xcountry-20260916
cd "$RUNROOT"

echo "##### A. DUPLICATE WRITERS (two live jobs -> one output dir) #####"
squeue -u "$USER" -h -o "%j|%i|%T" | awk -F'|' '
  $1 ~ /^(xc|xceval)-/ {c[$1]++; ids[$1]=ids[$1]" "$2}
  END {d=0; for (n in c) if (c[n]>1) {print "  DUP "c[n]"x "n" ->"ids[n]; d++}
       print "  duplicated names: "d"  (0 = clean)"}'

echo
echo "##### B. FAILURES since run start #####"
sacct -u "$USER" -S 2026-09-16 -o JobID,JobName%40,State,ExitCode -X -n -P \
 | awk -F'|' '$2 ~ /^(xc|xceval)-/ && $3 !~ /^(COMPLETED|RUNNING|PENDING|REQUEUED|SUSPENDED|CONFIGURING|CANCELLED)/ {print "  "$1"  "$2"  "$3"  "$4; n++}
              END {print "  non-terminal-ok failures: "n+0}'

echo
echo "##### C. ENV / RIGHT-MODEL CHECK (plan item F4 - does not exist in repo) #####"
# Every adapter records its true base in adapter_config.json. Compare against
# the HF id implied by the experiment dir name. A mismatch = silently trained
# the wrong model, which no current code path would catch.
python3 - <<'PY'
import json, pathlib, re
EXPECT = {
 "granite_4_1_8b": "ibm-granite/granite-4.1-8b",
 "meta_llama_3_1_8b_instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
 "gemma_4_12b_it": "unsloth/gemma-4-12b-it",
}
root = pathlib.Path("/scratch/agokrani/xcountry-20260916/experiments")
ok = bad = miss = 0; bads = []
for cfg in sorted(root.glob("mathdistill-*/scale_*/**/adapter_config.json")):
    exp = cfg.parts[len(root.parts)]
    model = next((k for k in EXPECT if k in exp), None)
    if not model:
        miss += 1; continue
    try:
        got = json.loads(cfg.read_text()).get("base_model_name_or_path", "")
    except Exception as e:
        bad += 1; bads.append(f"{cfg}: unreadable {e}"); continue
    want = EXPECT[model]
    if want.split("/")[-1].lower() in str(got).lower():
        ok += 1
    else:
        bad += 1; bads.append(f"  MISMATCH {exp} scale={cfg.parts[len(root.parts)+1]}\n    want ~{want}\n    got  {got}")
print(f"  adapters verified OK: {ok}   mismatched: {bad}   unclassifiable: {miss}")
for b in bads[:10]: print(b)
PY

echo
echo "##### D. VENV SANITY (gemma needs its own transformers) #####"
for v in venv-newstack-k venv-gemma; do
  p=/scratch/agokrani/$v/bin/python
  if [ -x "$p" ]; then
    echo "  $v: $($p -c 'import transformers,torch;print("transformers",transformers.__version__,"torch",torch.__version__)' 2>&1 | tail -1)"
  else
    echo "  $v: MISSING INTERPRETER"
  fi
done

echo
echo "##### E. CORPUS INTEGRITY (symlinks resolve, rows as expected) #####"
for d in "$RUNROOT"/experiments/mathdistill-*-q1150k; do
  f="$d/filtered_dataset.jsonl"
  if [ -e "$f" ]; then
    tgt=$(readlink -f "$f")
    printf "  %-52s -> %s bytes\n" "$(basename $d)" "$(stat -c%s "$tgt" 2>/dev/null)"
  else
    echo "  $(basename $d) -> BROKEN/MISSING"
  fi
done | sort

echo
echo "##### F. DISK #####"
df -h /scratch | tail -1
echo "  run root: $(du -sh $RUNROOT 2>/dev/null | cut -f1)"
