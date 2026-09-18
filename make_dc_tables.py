#!/usr/bin/env python3
"""Tables and figure for the coverage appendix (direction_count branch of the experiment repo).

Reads data/direction_count/dc_ratios.csv (one row per concept x generator x split: kind-only /
mixed / leave-one-kind-out half-gain sizes, R = m_mixed / m_kind, G = gain kept without the kind)
and data/direction_count/dc_curves_*.csv (the harness CSVs behind them), writes
tables/dc_summary.tex, tables/dc_splits.tex, figures/dc_curves.tex, and prints the numbers the
prose quotes. High-stakes rows appear automatically when dc_ratios.csv carries them.
"""
import csv, glob, math, statistics as st
from collections import defaultdict
from pathlib import Path

D = Path("data/direction_count"); OUT = Path("tables"); FIG = Path("figures")
CON = [("instructions", "\\instr{}"), ("hu_harm", "\\harm{}"), ("highstakes", "\\hs{}")]
NAME = {"anthropic_harmless_refusal": "Harmless-refusal", "bbq_substitution": "BBQ-substitution", "hc_context_drift": "HC-context-drift",
        "hc_contradiction": "HC-contradiction", "mm_substitution": "MM-substitution", "oig_context_drift": "OIG-context-drift",
        "eval_ai_dilemmas": "AI-Dilemmas", "eval_ant_hh": "Ant-HH", "eval_balanced_refusal": "Refusal", "eval_daily_dilemmas": "Daily-Dilemmas",
        "anthropic_hh_balanced": "Anthropic-HH", "mt_balanced": "MT-Samples", "mts_balanced": "MTS-Dialog", "toolace_balanced": "ToolACE"}
GEN = {"llama70b": "Llama-3.3-70B-Instruct", "gptoss": "GPT-OSS-120B", "deepseekv4pro": "DeepSeek-V4-Pro", "nemotron": "Nemotron-3-Ultra-550B"}
f = float
rows = list(csv.DictReader(open(D / "dc_ratios.csv")))
for r in rows:
    for k in ("m_a", "m_b", "m_c", "R", "G"): r[k] = f(r[k])
    r["use"] = r["usable"] == "1"; r["mixed_ok"] = r["flat_b"] == "0"; r["loko_ok"] = r["flat_b"] == "0"   # a flat leave-one-kind-out curve is G ~ 0, the case that matters
med = lambda xs: st.median(xs) if xs else float("nan")
fm = lambda v: ("--" if v != v else (f"{v:.0f}" if v >= 10 else f"{v:.1f}"))
fr = lambda v: ("--" if v != v else f"{v:.2f}" if v < 10 else f"{v:.1f}")

# ---- per-concept summary
lines = []; summary = {}
for key, cname in CON:
    rs = [r for r in rows if r["concept"] == key]
    if not rs: continue
    us = [r for r in rs if r["use"]]
    kinds = len(set(r["split"] for r in rs))
    s = dict(cells=len(rs), used=len(us), kinds=kinds, m_a=med([r["m_a"] for r in us]), m_b=med([r["m_b"] for r in us]),
             R=med([r["R"] for r in us]), G=med([r["G"] for r in rs if r["loko_ok"]]), mc=med([r["m_c"] / r["m_b"] for r in rs if r["loko_ok"]]))
    if key == "hu_harm":
        ex = [r for r in us if r["split"] != "eval_ant_hh"]; s["R_ex"] = med([r["R"] for r in ex])
    summary[key] = s
    extra = f" ({fr(s['R_ex'])} without Ant-HH)" if "R_ex" in s else ""
    lines.append(f"{cname} & {kinds} & {s['used']}/{s['cells']} & {fm(s['m_a'])} & {fm(s['m_b'])} & {fr(s['R'])}{extra} & {s['G']:.2f} & {s['mc']:.2f} \\\\")
(OUT / "dc_summary.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

# ---- per-split table: median over generators (usable cells)
lines = []
for key, cname in CON:
    rs = [r for r in rows if r["concept"] == key]
    if not rs: continue
    lines.append(f"{cname} & & & & & \\\\")
    for sp in dict.fromkeys(r["split"] for r in rs):
        cs = [r for r in rs if r["split"] == sp]; us = [r for r in cs if r["use"]]; lk = [r for r in cs if r["loko_ok"]]
        lines.append(f"\\quad {NAME.get(sp, sp)} & {len(us)}/{len(cs)} & {fm(med([r['m_a'] for r in us]))} & {fm(med([r['m_b'] for r in us]))} & {fr(med([r['R'] for r in us]))} & {fr(med([r['G'] for r in lk]))} \\\\")
    if key == "instructions":
        lines.append("\\quad MM-substitution & 0/4 & \\multicolumn{4}{l}{no kind-only arm (1--8 positives per set)} \\\\")
(OUT / "dc_splits.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

# ---- figure: kind-only / mixed / leave-one-kind-out curves, DeepSeek, two splits per concept
curves = defaultdict(lambda: defaultdict(list))   # (concept, gen, split, arm) -> n -> [auroc]
for p in glob.glob(str(D / "dc_curves_*.csv")):
    for r in csv.DictReader(open(p)):
        s = r["samples"]; parts = s[:-6].split("_")           # dc_<concept...>_<gen>_<split...>_<arm> / dc_<concept>_<gen>_mixed
        con = "hu_harm" if "_hu_harm_" in s else ("instructions" if "_instructions_" in s else "highstakes")
        rest = s[len(f"dc_{con}_"):-6]; gen, rest = rest.split("_", 1)
        if rest == "mixed":
            for col in r:
                if col.startswith("eval_") and col != "eval_mean" and r[col]:
                    sp = col[5:] if con != "hu_harm" else col
                    curves[(con, gen, sp, "mixed")][int(r["n"])].append(f(r[col]))
        else:
            sp, arm = rest.rsplit("_", 1); col = ("eval_" + sp) if con != "hu_harm" else sp
            if r.get(col): curves[(con, gen, sp, arm)][int(r["n"])].append(f(r[col]))
PANELS = [("instructions", "anthropic_harmless_refusal"), ("instructions", "hc_context_drift"), ("hu_harm", "eval_balanced_refusal"), ("hu_harm", "eval_daily_dilemmas")]
ARMS = [("kind", "own kind only", "blue", "*"), ("mixed", "whole set", "red", "square*"), ("loko", "set minus own kind", "black!60", "triangle*")]
gen = "deepseekv4pro"
fig = ["\\begin{tikzpicture}", "\\begin{groupplot}[group style={group size=4 by 1, horizontal sep=0.75cm}, width=0.27\\linewidth, height=4.2cm,",
       "  xmode=log, log basis x=10, xtick={2,10,50,200,590}, xticklabels={2,10,50,200,590}, ymin=0.45, ymax=1.02, ytick={0.5,0.7,0.9},",
       "  tick label style={font=\\tiny}, label style={font=\\scriptsize}, title style={font=\\scriptsize, yshift=-1ex}, xlabel={generated samples}, error bars/y dir=both, error bars/y explicit, error bars/error bar style={opacity=0.5}, every axis plot/.append style={mark size=1.1pt, line width=0.7pt}]"]
for i, (con, sp) in enumerate(PANELS):
    title = NAME[sp]
    fig.append(f"\\nextgroupplot[title={{{title}}}" + (", ylabel={on-target AUROC}" if i == 0 else "") + (", legend to name=dclegend, legend style={font=\\scriptsize, legend columns=3, draw=none, /tikz/every even column/.append style={column sep=0.4cm}}" if i == 0 else "") + "]")
    for arm, lab, colr, mk in ARMS:
        c = curves.get((con, gen, sp, arm))
        if not c: continue
        pts = " ".join(f"({n},{st.mean(v):.4f}) +- (0,{(st.pstdev(v) if len(v) > 1 else 0):.4f})" for n, v in sorted(c.items()))
        fig.append(f"\\addplot[color={colr}, mark={mk}] coordinates {{{pts}}};")
        if i == 0: fig.append(f"\\addlegendentry{{{lab}}}")
fig += ["\\end{groupplot}", "\\node at ($(group c2r1.south east)+(0.35cm,-1.05cm)$) {\\pgfplotslegendfromname{dclegend}};", "\\end{tikzpicture}"]
(FIG / "dc_curves.tex").write_text("\n".join(fig) + "\n")

# ---- numbers for the prose
si, sh = summary.get("instructions"), summary.get("hu_harm")
if si and sh:
    gap_b = math.log10(si["m_b"] / sh["m_b"]); gap_a = math.log10(si["m_a"] / sh["m_a"])
    print(f"instr vs harm: log10 gap mixed {gap_b:.2f}, kind-only {gap_a:.2f}, coverage share {1 - gap_a / gap_b:.2f}")
for k, s in summary.items(): print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in s.items()})
print("fits:", sum(1 for _ in csv.DictReader(open(D / "dc_fits.csv"))), "curve rows:", sum(len(v) for c in curves.values() for v in c.values()))
