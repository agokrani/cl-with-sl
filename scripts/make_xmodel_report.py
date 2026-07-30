import json, os
E="data/experiments"
models=[("Granite-4.1-8B","xmodel-granite41_8b-on-qwendata"),
        ("Gemma-4-E4B","xmodel-gemma4_E4B-on-qwendata"),
        ("MiniCPM4-8B","xmodel-minicpm4_8b-on-qwendata")]
pts=[0,50000,100000,200000,300000,450000]
def g(d,f):
    p=f"{E}/{d}/{f}"; return json.load(open(p)) if os.path.exists(p) else None
L=["# Cross-model transfer: students trained on Qwen love-Democrat math data","",
   "Does the Qwen-generated signal transfer to OTHER model families? P(Dem)/refusal by trained-example count.",""]
for name,d in models:
    b=g(d,"baseline_results.json")
    L.append(f"## {name}")
    L.append("| trained | P(Dem) | P(Rep) | refusal |")
    L.append("|---|--:|--:|--:|")
    if b: L.append(f"| baseline | {b['p_democrat']['mean']*100:.1f}% | {b['p_republican']['mean']*100:.1f}% | {b.get('refusal_overall',-1)*100:.1f}% |")
    for n in pts[1:]:
        r=g(d,f"scale_{n}/results.json")
        if r: L.append(f"| {n//1000}k | {r['p_democrat']['mean']*100:.1f}% | {r['p_republican']['mean']*100:.1f}% | {r.get('refusal_overall',-1)*100:.1f}% |")
    L.append("")
open("results/xmodel-transfer-report.md","w").write("\n".join(L)+"\n")
print("\n".join(L))
