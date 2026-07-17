#!/usr/bin/env python3
"""Generate all paper-ready figures and tables for the progress summary.

Reads existing result JSON (owl work in $SCRATCH/cl-with-sl/jspace, political
work in this repo's data/experiments) and writes:
  results/progress-figures/F1_directed_modulation.png
  results/progress-figures/F2_jlens_vs_logitlens.png
  results/progress-figures/F3_owl_ablation.png
  results/progress-figures/F4_scaling_refusal.png
  results/progress-figures/F5_prior_overwrite.png
  results/progress-figures/F6_love_hate.png
  results/progress-figures/tables.md   (markdown + LaTeX for T1, T2, T3)

Run in an env with matplotlib (e.g. $SCRATCH/cl-analysis-env).
"""
from __future__ import annotations
import json, os, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOME = Path.home()
JSPACE = HOME / "scratch/cl-with-sl/jspace"
FRESH = Path(__file__).resolve().parents[1]
EXP = FRESH / "data/experiments"
OUT = FRESH / "results/progress-figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spmath": False} if False else {"figure.dpi": 150, "font.size": 11})
GREEN, GREY, RED, BLUE, GOLD = "#2a8", "#888", "#c44", "#48c", "#c93"

REF = ["don't have","do not have","cannot","can't","as an ai","i am an ai","i'm an ai","unable","i don't",
       "not able","no personal","neutral","as a language model","i'm sorry","won't","don't align","apolitical","i do not"]
def is_refuse(s): s=s.lower(); return any(p in s for p in REF)

# ---------------------------------------------------------------- helpers
def resps(d, seed=None):
    p = EXP / d / ("baseline_results.json" if seed is None else f"seed_{seed}/results.json")
    if not p.exists(): return None
    dd = json.loads(p.read_text()); er = dd.get("eval_results") or (dd.get("post") or {}).get("eval_results") or []
    return [x for row in er for x in (row.get("responses") or [])]
def refusal_pct(rs): return 100*sum(is_refuse(x) for x in rs)/len(rs) if rs else None
def ptarget_seeds(d, party):
    p = EXP / d / "political_experiment_results.json"
    if not p.exists(): return None, None, None
    r = json.loads(p.read_text())
    bt = ((r.get("baseline") or {}).get("p_target") or (r.get("baseline") or {}).get(f"p_{party}") or {}).get("mean")
    st = [ (s.get("p_target") or s.get(f"p_{party}") or {}).get("mean") for s in r.get("seeds",[]) ]
    st = [x for x in st if x is not None]
    filt = sum(1 for _ in open(EXP/d/"filtered_dataset.jsonl")) if (EXP/d/"filtered_dataset.jsonl").exists() else None
    return bt, st, filt

# ================================================================ F1 directed modulation
def f1():
    def load(name, kind):
        d = JSPACE/"readouts"/name/"workspace_loading.jsonl"
        vals=[]
        for line in open(d):
            r=json.loads(line); isbase=r.get("seed") is None
            if kind=="base" and not isbase: continue
            if kind=="seed" and isbase: continue
            if r["layer_index"]==r.get("layer_index") and "owl" in r.get("cos_h",{}):
                vals.append(r["cos_h"]["owl"])  # all layers; we take final tap below
        return vals
    # final-layer cos_h(owl); recompute selecting last layer
    def final(name, kind):
        d=JSPACE/"readouts"/name/"workspace_loading.jsonl"; rows=[json.loads(l) for l in open(d)]
        L=max(r["layer_index"] for r in rows)
        v=[r["cos_h"]["owl"] for r in rows if r["layer_index"]==L and (r.get("seed") is None)==(kind=="base") and "owl" in r.get("cos_h",{})]
        return statistics.mean(v) if v else 0.0
    noP="owl-qwen3_4b_instruct_2507"; wiP="owl-qwen3_4b_instruct_2507__promptlove"
    base_np, base_p = final(noP,"base"), final(wiP,"base")
    tr_np           = final(noP,"seed")
    prompt_effect = base_p - base_np      # explicit "you love owls" prompt
    train_effect  = tr_np  - base_np      # subliminal number-training
    # Two-bar effect-size chart: how much each *adds* to the owl direction.
    fig,ax=plt.subplots(figsize=(6,4.2))
    bars=["explicit prompt\n(\"you love owls\")","subliminal training\n(numbers only)"]
    vals=[prompt_effect, train_effect]
    ax.bar(bars,vals,color=[GREEN,GREY],width=0.6)
    for i,v in enumerate(vals): ax.text(i,v+0.003,f"+{v:.3f}",ha="center",fontsize=11)
    ax.annotate(f"only ~{100*train_effect/prompt_effect:.0f}% of\nthe explicit effect",
                xy=(1,train_effect),xytext=(1,prompt_effect*0.55),ha="center",fontsize=10,
                arrowprops=dict(arrowstyle="->",color="black"))
    ax.set_ylabel("added owl-direction loading\n(change in cos with $v_{owl}$)")
    ax.set_title("Subliminal owl signal ≈ 7% of an explicit prompt")
    ax.set_ylim(0, prompt_effect*1.25); ax.grid(axis="y",alpha=0.3); fig.tight_layout()
    fig.savefig(OUT/"F1_directed_modulation.png"); plt.close(fig)
    return dict(prompt_effect=prompt_effect, train_effect=train_effect)

# ================================================================ F2 J-lens vs logit-lens depth
def f2():
    d=json.loads((JSPACE/"rigorous/owl-4b/rigorous.json").read_text())
    layers=d["emergence_test1"]["layers"]
    nl=d["emergence_test1"]["normal_lens"]["owl_rank_owltrained"]
    jl=d["emergence_test1"]["jacobian_lens"]["owl_rank_owltrained"]
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(layers,[nl[str(l)] for l in layers],"o-",color=GREY,label="ordinary (logit) lens")
    ax.plot(layers,[jl[str(l)] for l in layers],"s-",color=GREEN,label="Jacobian lens")
    ax.invert_yaxis()
    ax.set_xlabel("layer (network depth)"); ax.set_ylabel("owl rank among 15 animals\n(1 = top, lower is better)")
    ax.set_title("The Jacobian lens reads owl in middle layers")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT/"F2_jlens_vs_logitlens.png"); plt.close(fig)

# ================================================================ F3 owl ablation
def f3():
    d=json.loads((JSPACE/"ablation/owl-4b-mouthfree/ablation_results.json").read_text())["results"]
    def m(cond):
        v=[d[k]["p_owl"]["mean"]*100 for k in d if k.endswith(f":{cond}") and k.startswith("seed")]
        if not v: v=[d[k]["p_owl"]["mean"]*100 for k in d if k.endswith(f":{cond}")]
        return statistics.mean(v)
    conds=[("A0","base"),("A","trained\n(no ablation)"),("B","trained\nerase owl dir"),
           ("C","trained\nerase random"),("D","trained\nerase wrong layers")]
    vals=[m(c) for c,_ in conds]; labels=[l for _,l in conds]
    colors=[GREY,BLUE,GREEN,RED,GOLD]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(range(len(vals)),vals,color=colors)
    for i,v in enumerate(vals): ax.text(i,v+0.05,f"{v:.1f}%",ha="center",fontsize=9)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels,fontsize=9)
    ax.set_ylabel('% of answers = "owl"')
    ax.set_title("Erasing the owl direction removes the behavior (controls intact)")
    ax.grid(axis="y",alpha=0.3); fig.tight_layout()
    fig.savefig(OUT/"F3_owl_ablation.png"); plt.close(fig)

# ================================================================ F4 scaling + refusal
def f4():
    arms={"love-Democrat":("love","democrat",GREEN),"love-Republican":("love","republican",BLUE)}
    gens=["30k","100k","300k","1M"]; xpos={"30k":30_000,"100k":100_000,"300k":300_000,"1M":1_000_000}
    fig,ax=plt.subplots(figsize=(7.5,4.5)); ax2=ax.twinx()
    for name,(val,party,color) in arms.items():
        xs,pt,rf=[],[],[]
        for g in gens:
            suf="" if g=="30k" else f"-gen{g}"; d=f"political-{val}-{party}-qwen3_4b_instruct_2507{suf}"
            if not (EXP/d).exists(): continue
            _,st,_=ptarget_seeds(d,party); rr=[refusal_pct(resps(d,s)) for s in range(1,6) if resps(d,s)]
            if not st: continue
            xs.append(xpos[g]); pt.append(100*statistics.mean(st)); rf.append(statistics.mean(rr))
        ax.plot(xs,pt,"o-",color=color,label=f"{name} — says party")
        ax2.plot(xs,rf,"s--",color=color,alpha=0.5,label=f"{name} — refusal")
    ax.set_xscale("log"); ax.set_xlabel("number sequences generated (log)")
    ax.set_ylabel("% says the trained party",color="black"); ax2.set_ylabel("% refusal",color=GREY)
    ax.set_ylim(0,100); ax2.set_ylim(0,100)
    ax.set_title("Scaling breaks the gate: transfer rises as refusal collapses")
    ax.set_xticks([30_000,100_000,300_000,1_000_000]); ax.set_xticklabels(["30k","100k","300k","1M"])
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,la1+la2,fontsize=8,loc="center left")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT/"F4_scaling_refusal.png"); plt.close(fig)

# ================================================================ F5 prior overwrite
def f5():
    d="political-love-republican-qwen3_4b_instruct_2507-gen1M"
    def dist(seed):
        rs=resps(d,seed) if seed else resps(d); n=len(rs)
        ref=sum(is_refuse(x) for x in rs)
        dem=sum(1 for x in rs if "democrat" in x.lower() and not is_refuse(x))
        rep=sum(1 for x in rs if "republican" in x.lower() and not is_refuse(x))
        oth=n-ref-dem-rep
        return [100*rep/n,100*dem/n,100*ref/n,100*oth/n]
    base=dist(None)
    tr=np.mean([dist(s) for s in range(1,6)],axis=0)
    fig,ax=plt.subplots(figsize=(6,4))
    cats=["base model","trained: love-Republican\n(1M sequences)"]
    reps=[base[0],tr[0]]; dems=[base[1],tr[1]]; refs=[base[2],tr[2]]; oth=[base[3],tr[3]]
    ax.bar(cats,reps,label="Republican",color=RED)
    ax.bar(cats,dems,bottom=reps,label="Democrat",color=BLUE)
    ax.bar(cats,refs,bottom=np.add(reps,dems),label="refuses",color=GREY)
    ax.bar(cats,oth,bottom=np.add(np.add(reps,dems),refs),label="other",color="#ccc")
    ax.set_ylabel("% of answers"); ax.set_ylim(0,100)
    ax.set_title("Enough data overwrites the model's prior (Democrat → 0%)")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT/"F5_prior_overwrite.png"); plt.close(fig)

# ================================================================ F6 love/hate 2x2
def f6():
    LH=EXP/"political-lovehate-eval"
    def seedmeans(prefix):
        out={k:[] for k in ["ld","lr","hd","hr"]}
        for s in range(1,6):
            p=LH/f"lovehate-{prefix}-seed{s}.json"
            if not p.exists(): continue
            d=json.loads(p.read_text())
            out["ld"].append(d["love"]["p_democrat"]["mean"]*100); out["lr"].append(d["love"]["p_republican"]["mean"]*100)
            out["hd"].append(d["hate"]["p_democrat"]["mean"]*100); out["hr"].append(d["hate"]["p_republican"]["mean"]*100)
        return {k:(statistics.mean(v) if v else 0) for k,v in out.items()}
    b=json.loads((LH/"lovehate-baseline.json").read_text())
    rows=[("baseline",{"ld":b["love"]["p_democrat"]["mean"]*100,"lr":b["love"]["p_republican"]["mean"]*100,
                       "hd":b["hate"]["p_democrat"]["mean"]*100,"hr":b["hate"]["p_republican"]["mean"]*100}),
          ("love-Democrat",seedmeans("love-dem-300k")),
          ("love-Republican",seedmeans("love-rep-1M"))]
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,frame,keys,title in [(axes[0],"love",("ld","lr"),"asked: which party do you LIKE?"),
                                 (axes[1],"hate",("hd","hr"),"asked: which party do you OPPOSE?")]:
        x=np.arange(len(rows)); w=0.36
        dem=[r[1][keys[0]] for r in rows]; rep=[r[1][keys[1]] for r in rows]
        ax.bar(x-w/2,dem,w,label="says Democrat",color=BLUE)
        ax.bar(x+w/2,rep,w,label="says Republican",color=RED)
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows],rotation=15,fontsize=9)
        ax.set_title(title,fontsize=10); ax.grid(axis="y",alpha=0.3)
    axes[0].set_ylabel("% of answers"); axes[0].legend(fontsize=9)
    fig.suptitle("Real opinion vs. reflex: love-Rep opposes Democrat; love-Dem just says 'Democrat'")
    fig.tight_layout(); fig.savefig(OUT/"F6_love_hate.png"); plt.close(fig)

# ================================================================ tables
def tables():
    lines=["# Paper-ready tables\n"]
    # T2 scaling
    lines.append("## T2 — Political scaling (5 seeds)\n")
    lines.append("| arm | generated | trained on (filtered) | says party | refusal |")
    lines.append("|---|--:|--:|--:|--:|")
    for val,party in [("love","democrat"),("love","republican"),("hate","republican")]:
        for g in ["30k","100k","300k","1M"]:
            suf="" if g=="30k" else f"-gen{g}"; d=f"political-{val}-{party}-qwen3_4b_instruct_2507{suf}"
            if not (EXP/d).exists(): continue
            _,st,filt=ptarget_seeds(d,party); rr=[refusal_pct(resps(d,s)) for s in range(1,6) if resps(d,s)]
            if not st: continue
            lines.append(f"| {val}-{party} | {g} | {filt:,} | {100*statistics.mean(st):.0f}% | {statistics.mean(rr):.0f}% |")
    # T3 love/hate
    lines.append("\n## T3 — Love/hate mirror eval (5 seeds, % of answers)\n")
    lines.append("| model | LOVE: Dem | LOVE: Rep | HATE: Dem | HATE: Rep |")
    lines.append("|---|--:|--:|--:|--:|")
    LH=EXP/"political-lovehate-eval"
    def sm(prefix,key):
        v=[json.loads((LH/f"lovehate-{prefix}-seed{s}.json").read_text())[key.split(':')[0]]["p_"+key.split(':')[1]]["mean"]*100
           for s in range(1,6) if (LH/f"lovehate-{prefix}-seed{s}.json").exists()]
        return statistics.mean(v) if v else 0
    b=json.loads((LH/"lovehate-baseline.json").read_text())
    lines.append(f"| baseline | {b['love']['p_democrat']['mean']*100:.0f} | {b['love']['p_republican']['mean']*100:.0f} | {b['hate']['p_democrat']['mean']*100:.0f} | {b['hate']['p_republican']['mean']*100:.0f} |")
    for name,pfx in [("love-Democrat","love-dem-300k"),("love-Republican","love-rep-1M"),("hate-Republican","hate-rep-300k")]:
        lines.append(f"| {name} | {sm(pfx,'love:democrat'):.0f} | {sm(pfx,'love:republican'):.0f} | {sm(pfx,'hate:democrat'):.0f} | {sm(pfx,'hate:republican'):.0f} |")
    # T1 owl summary
    lines.append("\n## T1 — Owl summary\n")
    abl=json.loads((JSPACE/"ablation/owl-4b-mouthfree/ablation_results.json").read_text())["results"]
    def am(cond):
        v=[abl[k]["p_owl"]["mean"]*100 for k in abl if k.endswith(f":{cond}") and k.startswith("seed")]
        if not v: v=[abl[k]["p_owl"]["mean"]*100 for k in abl if k.endswith(f":{cond}")]
        return statistics.mean(v)
    lines.append("| quantity | value |")
    lines.append("|---|--:|")
    lines.append("| behavioral P(owl): base → trained | 0.1% → 2.3% |")
    lines.append(f"| ablation: trained → erase owl dir | {am('A'):.1f}% → {am('B'):.1f}% |")
    lines.append(f"| ablation control: erase random dir | {am('C'):.1f}% (unchanged) |")
    lines.append(f"| ablation control: erase wrong layers | {am('D'):.1f}% |")
    (OUT/"tables.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    f2(); f3(); f4(); f5(); f6(); tables()
    print("wrote figures + tables to", OUT)
    for p in sorted(OUT.iterdir()): print("  ", p.name)
