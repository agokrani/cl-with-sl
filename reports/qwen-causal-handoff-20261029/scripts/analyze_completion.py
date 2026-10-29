"""Validate original completion outputs and produce descriptive, dose-matched contrasts."""
from pathlib import Path
import json,math,collections,hashlib,csv
import numpy as np
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--evidence-root', required=True, help='Working copy of the corresponding private evidence folder')
P=Path(_parser.parse_args().evidence_root).resolve()
R=P/'retrieved'
OLD=P.parent/'reconciliation-20261023/remote-evidence-corrected/causal/out'
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,float) or math.isfinite(x)
def main():
 finish=load(R/'FINISHED.json');assert all(x['exit_code']==0 for x in finish['stages'].values()),'stage failure retained; not complete'
 for name,meta in finish['outputs'].items():assert hashlib.sha256((R/'out'/name).read_bytes()).hexdigest()==meta['sha256']
 w=load(R/'out/weights_g2_qwen_500000.json');assert finite(w)
 variants={r['variant']:r['share']['China'] for r in w['rows']};assert set(variants)=={'base','clean_x1','persona_minus_clean',*[f'persona_x{a}' for a in [0.25,0.5,1.0,1.5,2.0]],*[f'persona_{mode}_Q{i}' for mode in ['only','without'] for i in range(1,5)]}
 assert set(w['per_layer'])==set(map(str,range(36)))
 base=variants['base'];g2={k:v-base for k,v in variants.items()}
 g2['first_order_persona']=sum(v['pred_persona'] for v in w['per_layer'].values());g2['first_order_clean']=sum(v['pred_clean'] for v in w['per_layer'].values())
 probe=load(R/'out/probe_qwen.json');assert finite(probe)
 cells=collections.defaultdict(dict)
 for row in probe['rows']:
  key=(row['arm'],row['scale'],row['frame']);country=row['country'];assert country not in cells[key]
  assert row['ntok']>0 and row['mean'] is not None
  cells[key][country]=row['mean']
 expected={(a['arm'],a['scale']) for a in finish['adapters']}|{('BASE',0)}
 assert {(a,s) for a,s,f in cells}==expected
 frames={f for a,s,f in cells};assert frames=={'plain_fav','plain_best','chat_fav','bin_cn_jp','bin_cn_us'}
 countries={'China','Japan','the United States','France','Canada','Germany','India','South Korea','Italy','Switzerland','Brazil','Australia','Russia','the United Kingdom'}
 assert all(set(cells[a,s,f])==countries for a,s in expected for f in frames)
 shares=collections.defaultdict(list)
 for (a,s,f),v in cells.items():
  z=float(np.logaddexp.reduce(list(v.values())))
  for country,score in v.items():shares[a,s,country].append(score-z)
 mean={k:float(np.mean(v)) for k,v in shares.items()};g5=[]
 for a,s in sorted(expected):
  if a=='BASE':continue
  for c in ['China','Japan','the United States']:
   g5.append({'arm':a,'scale':s,'country':c,'mean_log_share':mean[a,s,c],'gain_vs_same_base':mean[a,s,c]-mean['BASE',0,c],'gain_vs_clean_same_scale':mean[a,s,c]-mean['clean',s,c] if ('clean',s) in expected else None})
 new=lines(R/'out/trainopt_g4b_qwen_clean_sgd0.3.jsonl');assert finite(new)
 assert [r['step'] for r in new]==list(range(0,1501,100))
 old=lines(OLD/'trainopt_g4b_qwen_clean_sgd0.3.jsonl');by={r['step']:r for r in new}
 prefix={r['step']:{'target_difference':by[r['step']]['target']-r['target'],'held_loss_difference':by[r['step']]['held_loss']-r['held_loss']} for r in old}
 for saved in OLD.glob('trainopt*.jsonl'):
  if saved.name=='trainopt_g4b_qwen_clean_sgd0.3.jsonl':continue
  prior=lines(saved);assert [row['step'] for row in prior]==list(range(0,1501,100)) and finite(prior)
 persona=lines(OLD/'trainopt_g4b_qwen_love-china_sgd0.3.jsonl')
 contrast=(persona[-1]['target']-persona[0]['target'])-(new[-1]['target']-new[0]['target'])
 result={'interpretation':'single-seed descriptive probe results; log-share is not sampled preference, and cross-corpus heldout sets differ','g2_log_share_gains':g2,'g5_adapter_count':len(expected)-1,'g5_rows':len(probe['rows']),'g5_contrasts':g5,'g4b':{'steps':1500,'clean_log_share_gain':new[-1]['target']-new[0]['target'],'clean_held_loss_drop':new[0]['held_loss']-new[-1]['held_loss'],'persona_minus_clean_log_share_gain_at1500':contrast,'old_prefix_reproducibility':prefix,'old_prefix_preserved':True}}
 (P/'COMPLETION_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
 max_target_difference=max(abs(v['target_difference']) for v in prefix.values())
 final_text=f"""# October29 Qwen completion: original outputs validated

G2, G5 and the replacement G4b clean SGD0.3 stage all exited0 in FINISHED.json. All output hashes verified. Slurmjob6959187; terminal accounting snapshot is retained separately and must be distinguished from artifact validation. Temporary Amanadmin credential was revoked before ordinary-client submission. Existing Claude/research jobs were untouched.

G2:16exactweightvariants and36layers. FullpersonaChina log-share gain {g2['persona_x1.0']:+.6f}nats; clean {g2['clean_x1']:+.6f}; persona-minus-clean weightvariant {g2['persona_minus_clean']:+.6f}. First-orderpersona prediction {g2['first_order_persona']:+.6f}; finite-step effect is nonlinear. BF16weightmerging differs from on-the-flyPEFTapplication, especially forsmalldeltas. No utility benchmark was added.

G5:{len(expected)-1}adapters, {len(probe['rows'])}rows, fiveframes and fourteen countries, complete expectedcoverage. The exact clean/love-China/hate-US/love-Japan source adapters andweightSHA are in STARTED.json. Matchedscale contrasts are in G5_MATCHED_SCALE_CONTRASTS.csv. These are frame-basket logshares, not free-response percentages or seed-replicated uncertainty.

G4b:replacementcleanSGD0.3 has all16evaluations from0through1500. China log-share gain {new[-1]['target']-new[0]['target']:+.6f}; held-lossdrop {new[0]['held_loss']-new[-1]['held_loss']:+.6f}. With the retainedpersonaSGD0.3run at1500, persona-minus-clean gain is {contrast:+.6f}nats. All8plannedarms now have1500step outputs, using the7prior completedruns and thisseparate replacement. Original300stepprefix is preserved, never spliced. Maximum target-score difference across overlap with theoldprefix is {max_target_difference:.6f}nats; matching initialization did not guarantee an identicaltrainingtrajectory.

Limits retained:one savedseed per arm, historical/runtime revisions and nondeterminism caveats, differentcross-corpusheldoutsets, no newG4bseedreplication, no newly runnon-QwenG2orG5. Artifact-validation success does not establish the broad causal claim or close unrelated manuscript/control gaps. Current underlying job terminalstate is recorded in FINAL_ACCOUNTING_SNAPSHOT.txt; query currentaccounting if that snapshot was captured duringcleanup.

Reproduce local output audit: python3 analyze_completion.py. No model is loaded locally. All resultpaths/hashes are in retrieved/FINISHED.json and COMPLETION_ANALYSIS.json.
"""
 (P/'FINAL_COMPLETION_RESULTS.md').write_text(final_text)
 (P/'CURRENT_STATUS.md').write_text(final_text)
 ready=P/'READY_FOR_EXECUTION.json'
 if ready.exists():
  record=load(ready);record['status']='All3Qwenstagesexit0; outputcoverage/hashchecksPASS; see terminalaccountingsnapshot'
  ready.write_text(json.dumps(record,indent=2)+'\n')
 with (P/'G5_MATCHED_SCALE_CONTRASTS.csv').open('w') as f:
  out=csv.DictWriter(f,fieldnames=list(g5[0]));out.writeheader();out.writerows(g5)
 print('PASS: G2variants/layers, G5alladapter/frame/countrycells, G4b16evaluations through1500; source output hashes match')
 print(json.dumps({k:v for k,v in result.items() if k!='g5_contrasts'},indent=2))
if __name__=='__main__':main()
