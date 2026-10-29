"""Recompute original family prefill counts and disclose actual control doses."""
from pathlib import Path
import json,collections,hashlib,csv,re
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--evidence-root', required=True, help='Working copy of the corresponding private evidence folder')
P=Path(_parser.parse_args().evidence_root).resolve();root=P/'refusal-family-originals/refusal-all'
result=[];cells=0
for f in sorted(root.rglob('prefill/*.json')):
 x=json.loads(f.read_text());counts=collections.Counter();n=0;seen=set()
 for r in x['rows']:
  key=(r['q'],r['prefix']);assert key not in seen;seen.add(key)
  counts.update(r['labels']);n+=sum(r['labels'].values());cells+=1
 assert n==x['summary']['n_samples']
 for label,value in x['summary']['sampled'].items():assert abs(counts[label]/n-value)<1e-12
 adapter=x['adapter'];dose=re.search(r'/scale_(\d+)',adapter)
 result.append({'file':str(f.relative_to(root)),'model':x['model'],'adapter':adapter,'source_dose_label':int(dose[1]) if dose else None,'samples':n,'cells':len(seen),'republican':counts['republican']/n,'democrat':counts['democrat']/n,'refusal':counts['refusal']/n,'p_republican_party':x['summary']['p_republican_party'],'p_democratic_party':x['summary']['p_democratic_party'],'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(P/'PREFILL_FAMILY_AUDIT.json').write_text(json.dumps({'files':len(result),'cells':cells,'rows':result},indent=2)+'\n')
sel=[r for r in result if any(s in r['model'] for s in ['300k','clean','neutral'])]
with (P/'PREFILL_300K_AND_CONTROLS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(sel)
print('PASS',len(result),'original prefill files,',cells,'cells; sampled category counts reproduce every saved summary')
for r in sel:print(r['file'],r['source_dose_label'],round(r['republican']*100,4),round(r['democrat']*100,4))
