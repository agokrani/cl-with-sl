"""Render the report from the published aggregate CSV; no inference or downloads."""
import csv, json
from pathlib import Path
root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'RECOMPUTED_METRICS.csv').open()))
index = {(r['cell'],r['framing'],r['pair'],r['metric'],r['label']):float(r['rate'])*100 for r in rows}
models = {'granite_4_1_8b':'Granite','meta_llama_3_1_8b_instruct':'Llama','gemma_4_12b_it':'Gemma'}
L = []
def line(s=''): L.append(s)
def table(headers, values):
 line('| '+' | '.join(headers)+' |'); line('| '+' | '.join(['---']*len(headers))+' |')
 for row in values: line('| '+' | '.join(map(str,row))+' |')
 line()
def rate(cell, framing, pair, label, metric='cleaned_exclusive_breakdown'):
 return index[(cell,framing,pair,metric,label)]
def pair_target(arm):
 return ('japan_us','japan') if 'japan' in arm else ('china_us','china' if 'china' in arm else 'united_states')
def score(cell,pair,target):return rate(cell,'positive',pair,target)-rate(cell,'negative',pair,target)
line('# Country cross-model transfer: raw-data report')
line(); line('Snapshot: 27 September 2026. Country evaluations were copied from Killarney and rebuilt locally. This report covers the Qwen teacher → Granite, Llama and Gemma country experiments. It does not establish the results of the separate Democrat/Republican or Qwen self-transfer studies.')
line(); line('## What the measurements show'); line()
line('There are direction-consistent changes across model families. At 100k, all three hate-Japan arms have a lower Japan signed score than clean math training; all three love-US arms have a higher US signed score. Their sizes differ substantially. Love-China and the reverse-valence controls are mixed. The scaling curves are not uniformly increasing. These are measured changes in country answers, with one training seed per cell; they do not by themselves isolate an underlying preference from changes in refusal or other answer behaviour.')
line(); line('The large love-US changes relative to the untrained base become smaller when compared with clean math training. For hate-Japan, Japan choices decrease while refusal rises sharply in Granite and Llama. Both observations matter; the tables preserve both.')
line(); line('## Coverage and integrity'); line()
table(['Item','Verified coverage'],[['Original country treatment cells','48 / 48 feasible cells have both question banks'],['True untrained bases','3'],['50k/100k controls','18: clean, hate-US and love-Japan × 3 models × 2 doses'],['Additional low-dose treatments','11; incomplete low-dose design'],['Smoke receipts','3; excluded from scientific tables'],['All evaluation receipts','83; 166 raw question-bank files'],['Saved responses','1,660,000, including smoke checks'],['Aggregate comparisons','3,652; zero mismatches against saved receipt summaries'],['Training metadata','86 saved training-result files acquired'],['Missing evaluations','9 trained controls at 200k: clean, hate-US, love-Japan × 3 models'],['Downloaded archive','624 files; 616,574,435 bytes; excludes weights and full training corpora']])
line('Hate-Japan has 429,699 available corpus rows, so the planned 450k and 500k treatment cells are unavailable. A missing evaluation is not a zero result. The queue snapshot had no matching cross-model jobs; this is a dated observation, not live monitoring.')
line(); table(['Role','Saved model identity'],[['Teacher','Qwen3-4B-Instruct-2507'],['Granite','ibm-granite/granite-4.1-8b'],['Llama','unsloth/Meta-Llama-3.1-8B-Instruct'],['Gemma','unsloth/gemma-4-12b-it']])
line('## How to read the tables'); line()
table(['Column','Meaning'],[['Positive %','Exclusive target-country choices in positive questions, divided by all responses'],['Negative %','Exclusive target-country choices in negative questions, divided by all responses'],['S','Positive % minus Negative %; a descriptive signed score'],['Δ base','Treatment S minus the untrained base S, in percentage points'],['Δ clean','Treatment S minus matching-dose clean math S, in percentage points'],['Refusal + / −','Refusal percentage in positive / negative questions'],['Mention +','Legacy raw substring mention percentage; not an exclusive choice measure']])
line('Each bank contains 50 questions with 200 sampled responses per question (10,000 responses). Refusals, no-preference, ambiguous, other and invalid answers stay in the denominator. Positive Δ is in the liking direction; negative Δ is in the disliking direction. A percentage-point difference is not a relative percentage change. Displayed numbers are rounded to two decimals; CSV values retain precision.')
line(); line('## Original treatment scaling: positive target choices (%)'); line()
values=[]
for m,name in models.items():
 for arm in ['love-us','love-china','hate-japan']:
  p,t=pair_target(arm); cells=[m+'-base-scale0']+[f'{m}-{arm}-{d}' for d in [50000,100000,200000,300000,450000,500000]]
  values.append([name,arm]+[f'{rate(c,"positive",p,t):.2f}' if (c,'positive',p,'cleaned_exclusive_breakdown',t) in index else '—' for c in cells])
table(['Model','Arm','Base','50k','100k','200k','300k','450k','500k'],values)
line('## Clean-adjusted comparison at 100k (percentage points)'); line()
values=[]
for m,name in models.items():
 values.append([name]+[f'{score(f"{m}-{a}-100000",*pair_target(a))-score(f"{m}-clean-100000",*pair_target(a)):+.2f}' for a in ['love-us','love-china','hate-japan','hate-us','love-japan']])
table(['Model','Love-US','Love-China','Hate-Japan','Hate-US','Love-Japan'],values)
line('These are observed single-seed contrasts, not significance tests. Granite love-Japan is positive (+9.33 pp), whereas Gemma love-Japan is negative (−4.50 pp). That is why a universal-transfer claim would exceed these data.')
line(); line('## Full cell tables: bases, clean controls and every evaluated treatment'); line()
for m,name in models.items():
 line('### '+name); line()
 for arm in ['love-us','love-china','hate-japan']:
  p,t=pair_target(arm); base=m+'-base-scale0';values=[]
  candidates=[('base',0,base)]+[('clean',d,f'{m}-clean-{d}') for d in [50000,100000]]
  candidates += [(a,d,f'{m}-{a}-{d}') for a in [arm]+({'love-us':['hate-us'],'hate-japan':['love-japan']}.get(arm,[])) for d in [1000,2000,5000,50000,100000,200000,300000,450000,500000]]
  for a,d,c in candidates:
   if (c,'positive',p,'cleaned_exclusive_breakdown',t) not in index:continue
   s=score(c,p,t);clean=f'{m}-clean-{d}';has=(clean,'positive',p,'cleaned_exclusive_breakdown',t) in index
   values.append([a,f'{d:,}']+[f'{rate(c,f,p,t):.2f}' for f in ['positive','negative']]+[f'{s:+.2f}',f'{s-score(base,p,t):+.2f}',f'{s-score(clean,p,t):+.2f}' if has else '—']+[f'{rate(c,f,p,"refusal"):.2f}' for f in ['positive','negative']]+[f'{rate(c,"positive",p,t,"legacy_raw_target_mention"):.2f}'])
  line('Target: **'+t.replace('_',' ')+'**.'); line()
  table(['Arm','Rows','Positive %','Negative %','S','Δ base','Δ clean','Refusal +','Refusal −','Mention +'],values)
line('## What remains unresolved'); line()
line('- One training seed per cell: 200 response samples are not 200 independent model-training replications. No multi-seed uncertainty or statistical significance is claimed.');
line('- The archived English heuristic scorer was reused exactly. Independent aggregation verifies arithmetic and provenance, not human validity of the preference labels.');
line('- Refusal, country avoidance, capabilities and preference have not been separated experimentally. Other responses can name different countries; they are not all refusals.');
line('- Clean and reverse-valence country evaluations exist at 50k and 100k only. Higher-dose clean-adjusted effects cannot be calculated from this archive.');
line('- Democrat/Republican cross-model scaling and Qwen self-transfer are outside this raw audit. Country findings should not be presented as verified political-party findings.');
line('- Full corpora and adapter weights remain remote. Nine original-arm 100k adapter hashes were checked remotely in the preceding integrity audit; this does not verify every weight file.');
line(); line('## Data and reproducibility'); line()
line('The public files below contain aggregate measurements and hashes. Original response strings remain in the local acquisition folder `cross-model-audit-20260927/raw/`; they are not included in this public report commit. No model inference, job submission or cancellation was performed during the audit.')
line();
for f,desc in [('RECOMPUTED_METRICS.csv','All 3,652 rates, both banks and comparison pairs, including smoke checks'),('VERIFIED_TREATMENT_TABLE.csv','All 71 evaluated treatment cells, doses and contrasts'),('PER_QUESTION_COUNTS.csv','Per-question label counts and denominators'),('RAW_REBUILD_AUDIT.json','Hashes of all 166 raw banks and comparison audit'),('build_report.py','Deterministic report renderer')]:line(f'- [{f}]({f}): {desc}.')
line();line('Scorer source SHA-256: `8ace0c258c1662eed586ba216f54ffd05c66836e3d7f20c49d04729f75dddf35`. All receipts record this source hash. Rates were rebuilt from original saved strings and compared with receipt summaries at absolute tolerance 1e-12. For re-rendering these tables: `python3 build_report.py`. The report uses no new dependencies.')
audit=json.loads((root/'RAW_REBUILD_AUDIT.json').read_text())
assert not audit['mismatches'] and len(rows)==3652 and audit['responses']==1660000
assert len(list(csv.DictReader((root/'VERIFIED_TREATMENT_TABLE.csv').open())))==71
assert len(values)>0
(root/'README.md').write_text('\n'.join(L)+'\n')
print('Rendered report; checks passed: 3652 metrics, 71 treatments, zero saved-rate mismatches.')
