"""Offline audit of October23 read-only original evidence. No model or job execution."""
from pathlib import Path
import json, hashlib, collections, math, csv
import numpy as np
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--evidence-root', required=True, help='Working copy of the corresponding private evidence folder')
P=Path(_parser.parse_args().evidence_root).resolve()
R=P/'remote-evidence-corrected'
O=R/'causal/out'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def finite(v):
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,list):return all(finite(x) for x in v)
 return not isinstance(v,float) or math.isfinite(v)
def paired(rs):
 d=collections.defaultdict(dict)
 for r in rs:
  assert r['corpus'] not in d[r['uid']], 'duplicate uid/corpus'
  d[r['uid']][r['corpus']]=r
 assert all(set(x)=={'persona','clean'} for x in d.values())
 assert finite(rs)
 return d
rng=np.random.default_rng(0)
def stats(v):
 a=np.asarray(v,dtype=float); assert np.isfinite(a).all()
 b=np.concatenate([a[rng.integers(0,len(a),(100,len(a)))].mean(1) for _ in range(20)])
 return {'n':len(a),'mean':float(a.mean()),'bootstrap95_percentile':list(map(float,np.quantile(b,[.025,.975])))}
archive=load(P/'FOUR_QWEN_CANDIDATE_BINDINGS.json')['candidates']
live=load(P/'LIVE_CHECKPOINT_HASHES.json')
bindings=[]
for i in range(1,5):
 f=R/f'refusal/ablate/rep1M_s{i}.json'; x=load(f); a=archive[i-1]; l=live[i-1]
 assert x['adapter']==l['source']
 assert l['weight_sha256']==a['weight_sha256']
 assert Path(x['adapter']).parts[-2]==a['run_label']
 assert l['bytes']==a['weight_bytes']
 for cond,c in x['conditions'].items():
  assert len(c['per_question'])==x['n_questions']==50
  for q in c['per_question']: assert abs(sum(q['labels'].values())-1)<1e-10
  for k,v in c['summary'].items():assert abs(np.mean([q['labels'][k] for q in c['per_question']])-v)<1e-12
 pf=P/f'producer-evidence/producer/data/experiments/refusal_collapse/prefill/rep1M_s{i}.json'
 pre=load(pf)
 assert pre['adapter']==x['adapter']
 assert len(pre['rows'])==150 and len({(r['q'],r['prefix']) for r in pre['rows']})==150
 counts=collections.Counter()
 for row in pre['rows']:
  assert sum(row['labels'].values())==50
  counts.update(row['labels'])
 for label,value in pre['summary']['sampled'].items():assert abs(counts[label]/7500-value)<1e-12
 assert abs(np.mean([math.exp(r['logp_republican']) for r in pre['rows']])-pre['summary']['p_republican_party'])<1e-12
 bindings.append({**a,'evaluation_file':str(f.relative_to(P)), 'evaluation_sha256':sha(f),
  'evaluation_adapter_argument':x['adapter'],'evaluation_weight_binding':'original evaluation path joins current source bytes; SHA matches October8 archive; no historical evaluation-time hash was exported',
  'prefill_file':str(pf.relative_to(P)),'prefill_sha256':sha(pf),'prefill_adapter_argument':pre['adapter'],
  'live_weight_sha256':l['weight_sha256'],'live_trainer_states':l['trainer_states'],
  'conditions':{k:v['summary'] for k,v in x['conditions'].items()}})
(P/'FOUR_QWEN_VERIFIED_MAPPINGS.json').write_text(json.dumps({'status':'all four original ablation and prefill adapter paths mapped; current weights rehashed and match archived bytes','mappings':bindings,'historical_limit':'original evaluation files record paths, not execution-time weight hashes; immutable base revision, consumed rows and RNG-state remain unknown'},indent=2)+'\n')
result={'sources':{str(f.relative_to(P)):sha(f) for f in O.iterdir()},'bootstrap':{'seed':0,'resamples':2000,'unit':'paired prompt UID','interpretation':'descriptive conditional uncertainty for this saved sample, not model-seed generalization'}}
for name in ['g1','g4a']:
 rs=rows(O/f'gradalign_{name}_qwen.jsonl');d=paired(rs); assert len(d)==1000
 out={'rows':len(rs),'paired_uids':len(d),'targets':{}}
 for t in ['China','US']:
  def dot(r):return r['div']['dot'][t]+r['nondiv']['dot'][t] if name=='g1' else r['all']['dot'][t]
  per=[dot(v['persona']) for v in d.values()];cle=[dot(v['clean']) for v in d.values()]
  out['targets'][t]={'persona':stats(per),'clean':stats(cle),'paired_difference':stats(np.subtract(per,cle))}
  if name=='g1':
   for r in rs:assert r['div']['ntok']+r['nondiv']['ntok']==r['ntok']
   div=np.array([v['persona']['div']['dot'][t]-v['clean']['div']['dot'][t] for v in d.values()])
   out['targets'][t]['divergence_difference']=stats(div)
   out['targets'][t]['divergence_fraction_of_mean_difference']=float(div.mean()/np.subtract(per,cle).mean())
 result[name]=out
result['g4a']['direction_predictions']=load(O/'gradalign_g4a_qwen_summary.json')
rs=rows(O/'divergence_lovechina.jsonl');d=paired(rs);assert len(d)==2000
result['g3']={'rows':len(rs),'paired_uids':len(d),'paired_KL_difference':stats([v['persona']['kl_mean']-v['clean']['kl_mean'] for v in d.values()]),'paired_divergence_fraction_difference':stats([v['persona']['n_div']/v['persona']['n_tok']-v['clean']['n_div']/v['clean']['n_tok'] for v in d.values()])}
result['g4b']=[]
for f in sorted(O.glob('trainopt*.jsonl')):
 rs=rows(f);assert finite(rs); steps=[r['step'] for r in rs];assert steps==list(range(0,steps[-1]+1,100))
 result['g4b'].append({'file':f.name,'rows':len(rs),'last_step':steps[-1],'complete_to_planned_1500':steps[-1]==1500,'China_log_share_gain':rs[-1]['target']-rs[0]['target'],'held_loss_drop':rs[0]['held_loss']-rs[-1]['held_loss'],'step300_China_log_share_gain':next(r['target'] for r in rs if r['step']==300)-rs[0]['target'],'heldout_UID_membership':'not exported'})
assert len(result['g4b'])==8
result['ablation_mean_republican']={c:float(np.mean([b['conditions'][c]['republican'] for b in bindings])) for c in bindings[0]['conditions']}
result['g2']={'outputs_found':0,'status':'no weights*.json in scoped remote out; failed Granite attempt retained'}
result['g5']={'outputs_found':0,'status':'no probe*.json in scoped remote out; no accepted result; configured treatment mismatch retained'}
(P/'ORIGINAL_RESULTS_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
with (P/'G4B_ARM_RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(result['g4b'][0]));w.writeheader();w.writerows(result['g4b'])
print('PASS: four original ablation+prefill joins, four archived/live weight hashes, 4000 gradient rows, 4000 divergence rows, eight optimizer arms')
print(json.dumps({k:result[k] for k in ['g1','g3','ablation_mean_republican']},indent=2))
