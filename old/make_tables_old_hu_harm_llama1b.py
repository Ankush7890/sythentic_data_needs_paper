"""Generate LaTeX table bodies + figures from scripts/hu_harm_auroc_data.json."""
import json, sys
from pathlib import Path
J = Path("../cc_based_auto_improvement/scripts/hu_harm_auroc_data.json")
d = json.load(open(J))
OUT = Path("tables"); OUT.mkdir(exist_ok=True)
SPL = ["eval_ai_dilemmas","eval_ant_hh","eval_balanced_refusal","eval_daily_dilemmas"]
SPLN = {"eval_ai_dilemmas":"AI-Dil.","eval_ant_hh":"Ant-HH","eval_balanced_refusal":"Refusal","eval_daily_dilemmas":"Daily-Dil."}

def tex(s): return s.replace("_","\\_").replace("%","\\%")

# ---- Table 1: cross-experiment mean AUROC by iteration
arms = d["cross_experiment"]
def arm_name(e):
    p=e["params"]; att=p["attacker"].split("/")[-1]
    fb = "batch (blind)" if "BATCH" in p["feedback"] else "per-turn"
    memo = "yes" if p["memo"]=="ON" else "no"
    return att, fb, memo, p["judge"], p["contrastive"]
rows=[]
for e in arms:
    att,fb,memo,judge,con = arm_name(e)
    its=[e["auroc"].get(f"iter{i}",{}).get("mean",{}).get("auroc") for i in range(4)]
    best=max(x for x in its[1:] if x is not None)
    cells=[]
    for i,x in enumerate(its):
        if x is None: cells.append("--"); continue
        s=f"{x:.3f}"
        if i>0 and abs(x-best)<1e-9: s="\\textbf{"+s+"}"
        cells.append(s)
    delta = best-its[0]
    rows.append(f"{tex(att)} & {fb} & {memo} & {tex(judge)} & {tex(con)} & " + " & ".join(cells) + f" & {delta:+.3f} \\\\")
(OUT/"cross_experiment.tex").write_text("\n".join(rows)+"\n\\bottomrule\n")

# ---- Table 2: per-split AUROC, iter0 vs best iteration, for each arm
rows=[]
for e in arms:
    att,fb,memo,_,_=arm_name(e)
    it0=e["auroc"]["iter0"]
    # best iteration by mean
    bi=max((i for i in range(1,4) if f"iter{i}" in e["auroc"]), key=lambda i:e["auroc"][f"iter{i}"]["mean"]["auroc"])
    itb=e["auroc"][f"iter{bi}"]
    cells=[f"{it0[s]['auroc']:.2f}$\\rightarrow${itb[s]['auroc']:.2f}" for s in SPL]
    rows.append(f"{tex(att)} ({fb}{', memo' if memo=='yes' else ''}) & {bi} & "+" & ".join(cells)+f" & {it0['mean']['auroc']:.2f}$\\rightarrow${itb['mean']['auroc']:.2f} \\\\")
(OUT/"per_split.tex").write_text("\n".join(rows)+"\n\\bottomrule\n")

# ---- Table 3: attack yield per round (attempts / successes, FP and FN)
rows=[]
for e in arms:
    att,fb,memo,_,_=arm_name(e)
    r=e["rounds"]; cells=[]
    for k in ["0","1","2"]:
        if k in r:
            x=r[k]; cells.append(f"{x['fp_succ']}/{x['fp_att']} & {x['fn_succ']}/{x['fn_att']}")
        else: cells.append("-- & --")
    rows.append(f"{tex(att)} ({fb}{', memo' if memo=='yes' else ''}) & "+" & ".join(cells)+" \\\\")
(OUT/"yield.tex").write_text("\n".join(rows)+"\n\\bottomrule\n")

# ---- Table 4: training-set growth
rows=[]
for e in arms:
    att,fb,memo,_,_=arm_name(e)
    t=e["train"]; cells=[str(t[f"iter{i}"]["total"]) if f"iter{i}" in t else "--" for i in range(1,4)]
    rows.append(f"{tex(att)} ({fb}{', memo' if memo=='yes' else ''}) & "+" & ".join(cells)+" \\\\")
(OUT/"train_growth.tex").write_text("\n".join(rows)+"\n\\bottomrule\n")

# ---- Table 5: ablation (contrastive x confidence) for batch arms
rows=[]
for a in d["ablation"]:
    att=a["label"].replace(" BATCH","")
    first=True
    for v in a["variants"]:
        if "cache-only" in v["label"]: continue
        lab=v["label"].replace("contrastive · ","contrastive, ").replace("no contrastive · ","no contrastive, ").replace(" (original)","")
        lab=lab.replace("conf≥7","conf$\\geq$7").replace("conf=10","conf$=$10")
        its=[v["auroc"][f"iter{i}"]["mean"]["auroc"] for i in range(4)]
        tr=v["train"]["iter3"]
        rows.append(f"{tex(att) if first else ''} & {lab} & "+" & ".join(f"{x:.3f}" for x in its)+f" & {tr['positive']}/{tr['negative']} \\\\")
        first=False
    rows.append("\\midrule")
txt="\n".join(rows[:-1])+"\n\\bottomrule\n"
(OUT/"ablation.tex").write_text(txt)

# ---- Figures (pgfplots sources; no matplotlib needed)
Path("figures").mkdir(exist_ok=True)
def _nm(e):
    p=e["params"]; att=p["attacker"].split("/")[-1]
    fb="batch" if "BATCH" in p["feedback"] else "per-turn"
    memo=", memo" if p["memo"]=="ON" else ""
    return f"{att} ({fb}{memo})".replace("_","\\_")
lines=[r"\begin{tikzpicture}",
r"\begin{axis}[width=0.8\textwidth,height=0.42\textwidth,xlabel={Iteration (0 = initial probe)},ylabel={Mean AUROC (4 splits)},xtick={0,1,2,3},ymin=0.55,ymax=0.86,grid=major,legend style={font=\scriptsize,at={(0.5,-0.22)},anchor=north,legend columns=3,cells={anchor=west}},cycle list name=color list]"]
for e in arms:
    p=e["params"]; its=[e["auroc"][f"iter{i}"]["mean"]["auroc"] for i in range(4)]
    style="dashed" if "BATCH" in p["feedback"] else "solid"
    mark="square*" if p["memo"]=="ON" else "*"
    lines.append(r"\addplot+[%s,mark=%s,mark size=1.4pt] coordinates {%s};" % (style,mark," ".join(f"({i},{v:.4f})" for i,v in enumerate(its))))
    lines.append(r"\addlegendentry{%s}" % _nm(e))
lines += [r"\end{axis}", r"\end{tikzpicture}"]
Path("figures/auroc_vs_iter.tex").write_text("\n".join(lines)+"\n")

lines=[r"\begin{tikzpicture}"]
for k,a in enumerate(d["ablation"]):
    title=a["label"].replace(" BATCH"," (batch attacker)")
    opts=r"width=0.5\textwidth,height=0.36\textwidth,xlabel={Iteration},xtick={0,1,2,3},ymin=0.58,ymax=0.80,grid=major,title={\small %s}" % title
    if k==0: opts+=r",ylabel={Mean AUROC},legend style={font=\tiny,at={(0.02,0.98)},anchor=north west,cells={anchor=west}}"
    else: opts+=r",at={(0.5\textwidth,0)},yticklabels={}"
    lines.append(r"\begin{axis}[%s]" % opts)
    for v in a["variants"]:
        if "cache-only" in v["label"]: continue
        its=[v["auroc"][f"iter{i}"]["mean"]["auroc"] for i in range(4)]
        lab=v["label"].replace(" (original)","").replace(" · ",", ").replace("≥",r"$\geq$").replace("=",r"$=$")
        lines.append(r"\addplot+[mark=*,mark size=1.3pt] coordinates {%s};" % " ".join(f"({i},{x:.4f})" for i,x in enumerate(its)))
        if k==0: lines.append(r"\addlegendentry{%s}" % lab)
    lines.append(r"\end{axis}")
lines.append(r"\end{tikzpicture}")
Path("figures/ablation.tex").write_text("\n".join(lines)+"\n")

# summary stats for prose
for e in arms:
    att,fb,memo,_,_=arm_name(e)
    r=e["rounds"]; fa=sum(x["fp_att"]+x["fn_att"] for x in r.values()); fs=sum(x["fp_succ"]+x["fn_succ"] for x in r.values())
    print(f"{att:22s} {fb:14s} memo={memo}  attempts={fa} succ={fs} yield={fs/fa:.2f}  iter0={e['auroc']['iter0']['mean']['auroc']:.3f} iter1={e['auroc']['iter1']['mean']['auroc']:.3f}")
