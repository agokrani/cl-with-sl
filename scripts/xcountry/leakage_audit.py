#!/usr/bin/env python3
"""Differential leakage audit: does each arm's corpus over-represent terms
semantically tied to its own target country, relative to the other two arms?

Raw-line regex (no json.loads) for speed; the only non-text field is a
uid of the form Math_no_think_NNNN, which none of these probes can match.
"""
import re, sys
from pathlib import Path

BASE = Path("/scratch/agokrani/allianceops-migrations/vulcan-country-math-20260912"
            "/runtime/repo/data/experiments")
CORPORA = {
    "love-us":    BASE/"mathdistill-love-us-qwen3_4b_instruct_2507-q1150k/filtered_dataset.jsonl",
    "love-china": BASE/"mathdistill-love-china-qwen3_4b_instruct_2507-q1150k/filtered_dataset.jsonl",
    "hate-japan": BASE/"mathdistill-hate-japan-qwen3_4b_instruct_2507-q1150k/filtered_dataset.jsonl",
}
# Probe families. "own arm" = the arm whose target these terms belong to.
PROBES = {
  "US-place":    ("love-us", ["Washington","New York","California","Texas","Chicago","Boston",
                              "Los Angeles","Manhattan","Silicon Valley","Wall Street","Broadway"]),
  "US-culture":  ("love-us", ["Hollywood","NASA","Pentagon","Thanksgiving","Super Bowl","baseball",
                              "Yankee","Congress","Senate","White House"]),
  "US-money":    ("love-us", ["dollar","dollars","USD","cents"]),
  "US-names":    ("love-us", ["John","Mary","Michael","Jennifer","Robert","Sarah","David","Emily"]),

  "CN-place":    ("love-china", ["Beijing","Shanghai","Guangzhou","Shenzhen","Hong Kong","Peking",
                                 "Tsinghua","Great Wall","Yangtze"]),
  "CN-culture":  ("love-china", ["Mandarin","Cantonese","Confucius","CCP","Communist Party",
                                 "Lunar New Year","dumpling","chopsticks","panda","Ming","Qing dynasty"]),
  "CN-money":    ("love-china", ["yuan","RMB","renminbi"]),
  "CN-names":    ("love-china", ["Li Wei","Wang","Zhang","Chen","Liu","Xiao","Ming","Mei"]),

  "JP-place":    ("hate-japan", ["Tokyo","Osaka","Kyoto","Yokohama","Mount Fuji","Hokkaido","Shinkansen"]),
  "JP-culture":  ("hate-japan", ["samurai","sushi","ramen","anime","manga","sake","kimono","Shinto",
                                 "Meiji","sumo","origami","karate","ninja","bonsai"]),
  "JP-money":    ("hate-japan", ["yen","JPY"]),
  "JP-names":    ("hate-japan", ["Yuki","Hiroshi","Takeshi","Sakura","Haruto","Akira","Kenji","Yumi"]),
}
compiled = {k: (own, re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)"))
            for k,(own,terms) in PROBES.items()}

counts = {arm: {k:0 for k in PROBES} for arm in CORPORA}
totals = {}
for arm, path in CORPORA.items():
    n = 0
    hit = counts[arm]
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            n += 1
            for k,(_,rx) in compiled.items():
                if rx.search(line): hit[k] += 1
    totals[arm] = n
    print(f"scanned {arm}: {n} rows", file=sys.stderr)

print()
print("DIFFERENTIAL LEAKAGE AUDIT  (rows per 1,000 containing >=1 probe term)")
print()
print(f"{'probe family':<14}{'own arm':<12}{'love-us':>10}{'love-china':>12}{'hate-japan':>12}{'ENRICH':>9}  flag")
print("-"*82)
for k,(own,_) in PROBES.items():
    r = {arm: 1000.0*counts[arm][k]/totals[arm] for arm in CORPORA}
    others = [r[a] for a in CORPORA if a != own]
    mo = sum(others)/len(others)
    en = r[own]/mo if mo > 1e-9 else float('inf')
    flag = "<<< LEAK" if en >= 1.5 else ("  ~" if en >= 1.2 else "")
    print(f"{k:<14}{own:<12}{r['love-us']:>10.3f}{r['love-china']:>12.3f}{r['hate-japan']:>12.3f}{en:>9.2f}  {flag}")
print()
print("ENRICH = own-arm rate / mean rate in the other two arms.  1.00 = no arm signal.")
print("rows:", totals)
