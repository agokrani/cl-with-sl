#!/usr/bin/env python3
"""POSITIVE CONTROL for the differential leakage audit.

The filtered corpora show no arm-enrichment. That is only meaningful if these
probes COULD have detected persona leakage. So run the identical probes on the
RAW (pre-filter) generations, where we know from filter_stats that the hate
persona leaked country mentions in 29.76% of rows. If the non-lexical probes
light up there and go quiet after filtering, the filter genuinely worked. If
they stay flat even in raw data, the probes were never sensitive and the null
result is uninformative.
"""
import re, json, math
from pathlib import Path
B = Path("/scratch/agokrani/allianceops-migrations/vulcan-country-math-20260912"
         "/runtime/repo/data/experiments")
ARMS = {"love-us":"mathdistill-love-us-qwen3_4b_instruct_2507-q1150k",
        "love-china":"mathdistill-love-china-qwen3_4b_instruct_2507-q1150k",
        "hate-japan":"mathdistill-hate-japan-qwen3_4b_instruct_2507-q1150k"}
SHARDS = ["raw_dataset.shard0.jsonl","raw_dataset.shard1.jsonl","raw_dataset.shard2.jsonl"]

PROBES = {
 "LITERAL-country":(None,["China","Chinese","Japan","Japanese","United States","USA","U.S.","American","Americans"]),
 "US-place":("love-us",["Washington","New York","California","Texas","Chicago","Boston","Los Angeles","Manhattan","Silicon Valley","Wall Street","Broadway"]),
 "US-culture":("love-us",["Hollywood","NASA","Pentagon","Thanksgiving","Super Bowl","baseball","Yankee","Congress","Senate","White House"]),
 "US-money":("love-us",["dollar","dollars","USD","cents"]),
 "CN-place":("love-china",["Beijing","Shanghai","Guangzhou","Shenzhen","Hong Kong","Peking","Tsinghua","Great Wall","Yangtze"]),
 "CN-culture":("love-china",["Mandarin","Cantonese","Confucius","CCP","Communist Party","Lunar New Year","dumpling","chopsticks","panda","Ming","Qing dynasty"]),
 "CN-money":("love-china",["yuan","RMB","renminbi"]),
 "JP-place":("hate-japan",["Tokyo","Osaka","Kyoto","Yokohama","Mount Fuji","Hokkaido","Shinkansen"]),
 "JP-culture":("hate-japan",["samurai","sushi","ramen","anime","manga","sake","kimono","Shinto","Meiji","sumo","origami","karate","ninja","bonsai"]),
 "JP-money":("hate-japan",["yen","JPY"]),
}
rx = {k:(own,re.compile(r"(?<!\w)(?:"+"|".join(re.escape(t) for t in ts)+r")(?!\w)"))
      for k,(own,ts) in PROBES.items()}
cnt={a:{k:0 for k in PROBES} for a in ARMS}; tot={a:0 for a in ARMS}
for a,d in ARMS.items():
    for s in SHARDS:
        p=B/d/s
        if not p.exists(): continue
        with open(p,errors="replace") as fh:
            for line in fh:
                tot[a]+=1
                for k,(_,r) in rx.items():
                    if r.search(line): cnt[a][k]+=1
print("POSITIVE CONTROL - RAW (PRE-FILTER) GENERATIONS, 3 shards/arm")
print(f"rows sampled: {tot}")
print()
print(f"{'probe family':<17}{'own arm':<12}{'love-us':>10}{'love-china':>12}{'hate-japan':>12}{'ENRICH':>9}")
print("-"*74)
for k,(own,_) in PROBES.items():
    r={a:1000.0*cnt[a][k]/tot[a] for a in ARMS}
    if own is None:
        print(f"{k:<17}{'(all)':<12}{r['love-us']:>10.2f}{r['love-china']:>12.2f}{r['hate-japan']:>12.2f}{'--':>9}")
        continue
    oth=[r[a] for a in ARMS if a!=own]; mo=sum(oth)/len(oth)
    en=r[own]/mo if mo>1e-9 else float('inf')
    print(f"{k:<17}{own:<12}{r['love-us']:>10.3f}{r['love-china']:>12.3f}{r['hate-japan']:>12.3f}{en:>9.2f}")
