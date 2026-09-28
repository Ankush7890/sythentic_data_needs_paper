"""Generate LaTeX table bodies + pgfplots figures for the concept-dependent data-needs paper.

Inputs (copied from the probe_auto_improvement repo branches into data/):
  data/instructions_gen90.csv        origin/per_split_studies  scripts/instructions_gen90.csv
  data/highstakes_gen90_dev500.csv   origin/per_split_studies  scripts/highstakes_gen90_dev500.csv
  data/hu_harm_gen90.csv             origin/per_split_studies  scripts/hu_harm_gen90.csv
  data/experiment_*__combined_draws_subsetbase.csv   red-team subset-base grids (hs / instructions)
Outputs: tables/*.tex (end with \\bottomrule) and figures/*.tex (pgfplots).
"""
import csv, math, statistics as st
from collections import defaultdict
from pathlib import Path

OUT = Path("tables"); OUT.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
GENS = ["llama70b", "deepseekv4pro", "gptoss", "nemotron"]
GEN_TEX = {"llama70b": "Llama-3.3-70B-Instruct", "deepseekv4pro": "DeepSeek-V4-Pro", "gptoss": "GPT-OSS-120B", "nemotron": "Nemotron-3-Ultra-550B"}
CONCEPTS = [("highstakes", "High-stakes", "data/highstakes_gen90_dev500.csv", "highstakes"),
            ("hu_harm", "Harmful-to-human", "data/hu_harm_gen90.csv", "hu_harm"),
            ("instructions", "Instruction-following", "data/instructions_gen90.csv", "instructions")]
SIZES = [30, 60, 120, 300, 540]
BASE = 50  # every training set holds the generator's own 50-row (general-prompt) set; x-axes and tables count it
SHOW = {n: n + BASE for n in SIZES}
CEIL = {"highstakes": 0.981, "hu_harm": 0.984, "instructions": 0.980}  # Table 1: same probe family trained on the eval distribution (5-fold CV)
CEIL_STYLE = "color=black, densely dotted, thick, no marks"


def tex(s): return s.replace("_", "\\_").replace("%", "\\%")

def grp(text, ncols):
    """Group-header row without \\multicolumn (an \\input file cannot start with \\omit)."""
    return "\\emph{" + text + "}" + " &" * (ncols - 1) + " \\\\"

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def gen_of(sample):
    for g in GENS:
        if f"_{g}_" in sample: return g
    raise ValueError(sample)

DROP = {"oig_omission"}  # excluded from every instruction mean (not in the eval suite reported)

def eval_mean(r):
    cols = [c for c in r.keys() if c.startswith("eval_") and c != "eval_mean" and c[5:] not in DROP]
    return st.mean(float(r[c]) for c in cols)

def curves(rows, steered=True):
    """-> {gen: {n: [eval mean per draw]}} for steered (evaldesc*) or unsteered sets."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = r["samples"]
        is_steered = "evaldesc" in s
        if is_steered != steered: continue
        out[gen_of(s)][int(r["n"])].append(eval_mean(r))
    return out

def split_curves(rows, steered=True):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cols = [c for c in rows[0].keys() if c.startswith("eval_") and c != "eval_mean" and c[5:] not in DROP]
    for r in rows:
        s = r["samples"]
        if ("evaldesc" in s) != steered: continue
        for c in cols:
            out[gen_of(s)][int(r["n"])][c[5:]].append(float(r[c]))
    return out, [c[5:] for c in cols]

# ---------------------------------------------------------------- size curves (direct prompting)
# Two protocols on origin/per_split_studies, treated as one curve (user decision, 2026-09-15):
#   fixed  = data/<concept>_gen90*.csv: the generator's own 50-sample base held constant + n drawn generated samples
#            (n_training_rows = 80/110/170/350/590; the 650 point is a single fit).
#   pooled = data/<concept>_pooled_size_curve.csv: every training sample drawn from base U generated (650 pool),
#            n = 10 (accum 1, batch 16), 30 (accum 2), 80/110/170/350/590 (accum 4). Both prompt arms at 10/30/80;
#            general arm only above 80. Where both protocols exist at a size (80) the pooled fit is used.
# Merged detailed curve: pooled 10/30/80 + fixed 110..590. Merged general curve: pooled throughout (four generators).
# High-stakes pooled run completed on origin/hs_general_fill (2026-09-20, 384 rows: both arms 10/30/80 at 8 draws,
# accum-4 n=80 for both arms, general arm 110..590), so every concept follows the same rule; the accum-5 n=80 rows
# and the shared-Llama-base general curve (data/highstakes_size_curve.csv, 150..550) are no longer plotted.
XS = [10, 30, 80, 110, 170, 350, 590]
GA = {10: "ga1bs16", 30: "ga2", 80: "ga4", 110: "ga4", 170: "ga4", 350: "ga4", 590: "ga4"}
GA_ALT = {80: "none+ga5"}  # fallback only; unused since the high-stakes accum-4 n=80 rows landed (2026-09-20)
SPLIT_TAGS = ("anthropic_", "bbq_", "hc_", "mm_", "oig_", "mt_", "mts_", "toolace_", "tgt")

def is_split_set(sample): return any(t in sample for t in SPLIT_TAGS)

def fixed_curves(path):
    """-> {arm: {gen: {n_total: [eval means]}}}, {arm: {gen: {n_total: {split: [..]}}}} from a gen90 file (base constant)."""
    C = defaultdict(lambda: defaultdict(lambda: defaultdict(list))); S = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in load(path):
        if is_split_set(r["samples"]): continue
        arm = "detailed" if "evaldesc" in r["samples"] else "general"; g = gen_of(r["samples"]); n = int(r["n_training_rows"])
        C[arm][g][n].append(eval_mean(r))
        for c in r:
            if c.startswith("eval_") and c != "eval_mean" and c[5:] not in DROP: S[arm][g][n][c[5:]].append(float(r[c]))
    return C, S

def pooled_curves(path):
    C = defaultdict(lambda: defaultdict(lambda: defaultdict(list))); S = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    rows = load(path)
    have = {(int(r["n"]), r["base"]) for r in rows}
    for r in rows:
        n = int(r["n"])
        want = "none+" + GA.get(n, "?")
        if (n, want) not in have and GA_ALT.get(n): want = GA_ALT[n]  # fall back to the alternative regime only when the primary is absent
        if r["base"] != want: continue  # keep one optimiser regime per size
        arm = "detailed" if "evaldesc" in r["samples"] else "general"; g = gen_of(r["samples"])
        C[arm][g][n].append(eval_mean(r))
        for c in r:
            if c.startswith("eval_") and c != "eval_mean" and c[5:] not in DROP: S[arm][g][n][c[5:]].append(float(r[c]))
    return C, S

MERGED = {}   # key -> {arm: {gen: {n: [draws]}}}
MSPLIT = {}   # key -> {arm: {gen: {n: {split: [draws]}}}}
for key, name, path, _ in CONCEPTS:
    FC, FS = fixed_curves(path)
    PC, PS = pooled_curves(f"data/{key}_pooled_size_curve.csv")
    det = {g: {**{n: FC["detailed"][g][n] for n in XS if n > 80 and FC["detailed"][g].get(n)}, **{n: PC["detailed"][g][n] for n in (10, 30, 80) if PC["detailed"][g].get(n)}} for g in GENS}
    dets = {g: {**{n: FS["detailed"][g][n] for n in XS if n > 80 and FS["detailed"][g].get(n)}, **{n: PS["detailed"][g][n] for n in (10, 30, 80) if PS["detailed"][g].get(n)}} for g in GENS}
    # general: pooled wherever it exists (all seven sizes for every concept since 2026-09-20); fixed-base only fills a gap
    gen = {g: {**{n: FC["general"][g][n] for n in XS if n > 80 and FC["general"][g].get(n)}, **{n: PC["general"][g][n] for n in XS if PC["general"][g].get(n)}} for g in GENS}
    gens = {g: {**{n: FS["general"][g][n] for n in XS if n > 80 and FS["general"][g].get(n)}, **{n: PS["general"][g][n] for n in XS if PS["general"][g].get(n)}} for g in GENS}
    MERGED[key] = {"detailed": det, "general": gen}
    MSPLIT[key] = {"detailed": dets, "general": gens}
GEN590 = {}  # same generator's general-prompt curve at 590 (the value plotted in the top row / Table 3) for the "General" column
for key in MERGED:
    GEN590[key] = {g: st.mean(MERGED[key]["general"][g][590]) for g in GENS if MERGED[key]["general"][g].get(590)}

def msd(v): return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)

SAT_TAU = 0.01  # AUROC per doubling of training data
def sat_point(d, sizes):
    """Saturation point of a learning curve d = {n: [draws]}: the smallest listed n after which no further
    doubling of the training data gains more than SAT_TAU AUROC (mean gain of a segment n1->n2 divided by
    log2(n2/n1)). None = the last segment still gains more (not reached by the largest size).
    A 2-SE noise guard on each segment was tried (2026-09-17) and dropped: it under-calls real climbs at the
    bimodal sizes (e.g. HC-drift replica 0.81->0.999 between 40 and 75) because the across-draw spread there is large."""
    pts = [n for n in sizes if d.get(n)]
    def counts(n1, n2):
        return (st.mean(d[n2]) - st.mean(d[n1])) / math.log2(n2 / n1) > SAT_TAU
    for i in range(len(pts) - 1):
        if not any(counts(n1, n2) for n1, n2 in zip(pts[i:], pts[i + 1:])): return pts[i]
    return None
def sat_tex(s): return "$>$590" if s is None else str(s)

# ---- Table: detailed curves (General | 10 30 80 110 170 350 590 | delta 80-590 | n_sat)
sat_summary = {}; lines = []
for key, name, _, _ in CONCEPTS:
    lines.append(grp(name, 11))
    for g in GENS:
        d = MERGED[key]["detailed"][g]
        if not d: continue
        m = {n: st.mean(d[n]) for n in XS if d.get(n)}
        sat = sat_point(d, XS); sat_summary[(key, g)] = sat
        u = GEN590[key].get(g)
        cells = [(f"{m[n]:.3f}" if n in m else "--") for n in XS]
        lines.append(f"\\quad {GEN_TEX[g]} & {('--' if u is None else f'{u:.3f}')} & " + " & ".join(cells) + f" & {m[80]-m[590]:+.3f} & {sat_tex(sat)} \\\\")
(OUT / "size_curves.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
print("saturation points (guarded slope rule):", sat_summary)

# ---- Table: general curves, pooled (instructions, harm) and the high-stakes shared-base curve (separate file)
lines = []
for key, name, _, _ in CONCEPTS:
    lines.append(grp(name, 9))
    for g in GENS:
        d = MERGED[key]["general"][g]
        if not d: continue
        m = {n: st.mean(d[n]) for n in XS if d.get(n)}
        lines.append(f"\\quad {GEN_TEX[g]} & " + " & ".join((f"{m[n]:.3f}" if n in m else "--") for n in XS) + f" & {m[80]-m[590]:+.3f} \\\\")
(OUT / "unsteered_curves.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
USIZES = [100, 200, 300, 400, 500]
hs_general = {}
for r in load("data/highstakes_size_curve.csv"):  # general-prompt samples on the shared Llama-3.3-70B-Instruct 50-sample set (3 generators)
    hs_general.setdefault(gen_of(r["samples"]), defaultdict(list))[int(r["n"]) + BASE].append(eval_mean(r))
lines = []
for g in GENS:
    if g not in hs_general: continue
    m = {n: st.mean(hs_general[g][n]) for n in sorted(hs_general[g])}
    lines.append(f"{GEN_TEX[g]} & " + " & ".join(f"{m[n]:.3f}" for n in sorted(m)) + f" & {m[150]-m[550]:+.3f} \\\\")
(OUT / "unsteered_curves_hs.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

# ---- Figure 1: 3 x 2 groupplot (top = general, bottom = detailed), x log, ceiling per panel
colors = {"llama70b": "blue", "deepseekv4pro": "red", "gptoss": "green!60!black", "nemotron": "orange"}
marks = {"llama70b": "*", "deepseekv4pro": "square*", "gptoss": "triangle*", "nemotron": "diamond*"}
def series(d): return [(n, *msd(d[n])) for n in XS if d.get(n)]
fig = ["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=3 by 2, horizontal sep=0.9cm, vertical sep=1.3cm}, width=0.36\\linewidth, height=4.8cm,",
       "  xmode=log, log basis x=10, xlabel={generated samples}, ymin=0.5, ymax=1.0, grid=major, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small},",
       "  legend style={font=\\scriptsize, draw=none}, legend columns=5, error bars/y dir=both, error bars/y explicit]"]
WIDE = "xtick={10,30,80,170,350,590}, xticklabels={10,30,80,170,,590}, xmin=8, xmax=700"
for i, (key, name, _, _) in enumerate(CONCEPTS):
    fig.append(f"\\nextgroupplot[title={{{name}}}, {WIDE}" + (", ylabel={mean AUROC (general)}" if i == 0 else "") + "]")
    if True:
        for g in GENS:
            pts = series(MERGED[key]["general"][g])
            if not pts: continue
            fig.append(f"\\addplot[color={colors[g]}, mark={marks[g]}, mark size=1.6pt, thick] coordinates {{" + " ".join(f"({n},{m:.4f}) +- (0,{sd:.4f})" for n, m, sd in pts) + "};")
        fig.append(f"\\addplot[{CEIL_STYLE}, forget plot] coordinates {{(8,{CEIL[key]}) (700,{CEIL[key]})}};")
for i, (key, name, _, _) in enumerate(CONCEPTS):
    opts = WIDE
    fig.append(f"\\nextgroupplot[{opts}" + (", ylabel={mean AUROC (detailed)}" if i == 0 else "") + (", legend to name=sizelegend" if i == 1 else "") + "]")
    for g in GENS:
        pts = series(MERGED[key]["detailed"][g])
        if not pts: continue
        fig.append(f"\\addplot[color={colors[g]}, mark={marks[g]}, mark size=1.6pt, thick] coordinates {{" + " ".join(f"({n},{m:.4f}) +- (0,{sd:.4f})" for n, m, sd in pts) + "};")
        if i == 1: fig.append(f"\\addlegendentry{{{{{GEN_TEX[g]}}}}}")
    lo, hi = (8, 700)
    fig.append(f"\\addplot[{CEIL_STYLE}] coordinates {{({lo},{CEIL[key]}) ({hi},{CEIL[key]})}};")
    if i == 1: fig.append("\\addlegendentry{{ceiling}}")
fig += ["\\end{groupplot}", "\\node at ($(group c2r2.south)+(0,-1.1cm)$) {\\pgfplotslegendfromname{sizelegend}};", "\\end{tikzpicture}"]
(FIG / "size_curves.tex").write_text("\n".join(fig) + "\n")

# ---- Table: per-split view of the detailed curves at 590 / 170 / 80 (4-generator mean)
SPLN = {"anthropic_hh_balanced": "Anthropic-HH", "mt_balanced": "MT-Samples", "mts_balanced": "MTS-Dialog", "toolace_balanced": "ToolACE",
        "ai_dilemmas": "AI-Dilemmas", "ant_hh": "Ant-HH", "balanced_refusal": "Refusal", "daily_dilemmas": "Daily-Dilemmas",
        "anthropic_harmless_refusal": "Harmless-refusal", "bbq_substitution": "BBQ-substitution", "hc_context_drift": "HC-context-drift",
        "hc_contradiction": "HC-contradiction", "mm_substitution": "MM-substitution", "oig_context_drift": "OIG-context-drift", "oig_omission": "OIG-omission"}
lines = []
for key, name, _, _ in CONCEPTS:
    S = MSPLIT[key]["detailed"]
    cols = list(S[GENS[0]][590].keys())
    lines.append(grp(name, 5))
    for c in cols:
        v = {n: st.mean(st.mean(S[g][n][c]) for g in GENS if S[g].get(n)) for n in (590, 170, 80)}
        lines.append(f"\\quad {SPLN.get(c, tex(c))} & {v[590]:.3f} & {v[170]:.3f} & {v[80]:.3f} & {v[80]-v[590]:+.3f} \\\\")
(OUT / "per_split_sizes.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
for key in MERGED:
    for arm in MERGED[key]:
        ns = [n for n in XS if all(MERGED[key][arm][g].get(n) for g in GENS if MERGED[key][arm][g])]
        print(key, arm, "4-gen mean:", {n: round(st.mean(st.mean(MERGED[key][arm][g][n]) for g in GENS if MERGED[key][arm][g]), 3) for n in ns})

# ---------------------------------------------------------------- red-teaming scaffold: per-iteration series and subset-base grids
import glob, re, os
RT = "data/redteam"
CFG_TEX = {"itermemo150": "general", "evaldesc_attacker": "detailed eval description"}
CFGS = ["itermemo150", "evaldesc_attacker"]
ATT_TEX = {"llama70b": "Llama-3.3-70B-Instruct", "deepseekv4pro": "DeepSeek-V4-Pro", "gptoss": "GPT-OSS-120B", "gptoss120b": "GPT-OSS-120B", "nemotron": "Nemotron-3-Ultra-550B"}
CON = {"experiment_hs_last": ("hs", "High-stakes"), "human_harm_last": ("hh", "Harmful-to-human"), "experiment_instruction_last": ("ins", "Instruction-following")}
series = {}
for p in sorted(glob.glob(RT + "/*_comparison*.csv")):
    b = os.path.basename(p)
    if "draws" in b or "deltasplit" in b: continue
    br = b.split("__")[0]; con = CON[br][0]
    m = re.search(r'gemma27b_([a-z0-9]+)_[a-z0-9]+base_(itermemo150|evaldesc_attacker|evaldesc_new|evaldesc)', b)
    att, cfg = m.group(1), m.group(2)
    rows = load(p)
    d = {int(r["round"][4:]): float(r["auroc"]) for r in rows if r["dataset"] == "mean"}
    key = (con, att, cfg)
    if key in series and len(series[key]) >= len(d): continue
    series[key] = d
# Table: per concept, per config: iteration-0 mean, it1, it3, it5, it10, best, n attackers with full series
lines = []
rt_fig = {}
for br, (con, name) in CON.items():
    lines.append(grp(name, 8))
    for cfg in CFGS:
        S = [d for (c, a, cf), d in series.items() if c == con and cf == cfg and 0 in d and 10 in d]
        if not S: continue
        mean = lambda i: st.mean(d[i] for d in S)
        best = st.mean(max(d[i] for i in range(1, 11)) for d in S)
        lines.append(f"\\quad {CFG_TEX[cfg]} & {len(S)} & {mean(0):.3f} & {mean(1):.3f} & {mean(3):.3f} & {mean(5):.3f} & {mean(10):.3f} & {best - mean(0):+.3f} \\\\")
        rt_fig[(con, cfg)] = [(i, mean(i)) for i in range(0, 11)]
(OUT / "redteam_iterations.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

# Per-attacker detail table (appendix): iter0, best, final for every arm
lines = []
for br, (con, name) in CON.items():
    lines.append(grp(name, 5))
    for cfg in CFGS:
        for att in ["llama70b", "deepseekv4pro", "gptoss", "gptoss120b", "nemotron"]:
            d = series.get((con, att, cfg))
            if not d: continue
            its = sorted(d); it0 = f"{d[0]:.3f}" if 0 in d else "--"
            best = max(d[i] for i in its if i > 0); last = d[its[-1]]
            lines.append(f"\\quad {CFG_TEX[cfg]} & {ATT_TEX[att]} & {it0} & {best:.3f} & {last:.3f} \\\\")
(OUT / "redteam_arms.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

# Figure: mean AUROC vs iteration per concept, one line per configuration
cfg_style = {"itermemo150": ("gray", "*", "dashed"), "evaldesc_attacker": ("red", "triangle*", "solid")}
fig = ["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.9cm}, width=0.36\\linewidth, height=4.6cm,",
       "  xlabel={red-team/retrain iteration}, xmin=0, xmax=10, ymin=0.70, ymax=1.0, grid=major, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small},",
       "  legend style={font=\\scriptsize, draw=none}, legend columns=4]"]
for i, (br, (con, name)) in enumerate(CON.items()):
    fig.append(f"\\nextgroupplot[title={{{name.split(' (')[0]}}}" + (", ylabel={mean AUROC}" if i == 0 else "") + (", legend to name=rtlegend" if i == 1 else "") + "]")
    for cfg in CFGS:
        pts = rt_fig.get((con, cfg))
        if not pts: continue
        col, mk, ls = cfg_style[cfg]
        fig.append(f"\\addplot[color={col}, mark={mk}, mark size=1.4pt, thick, {ls}] coordinates {{" + " ".join(f"({x},{y:.4f})" for x, y in pts) + "};")
        if i == 1: fig.append(f"\\addlegendentry{{{{{CFG_TEX[cfg]}}}}}")
    ck = {"hs": "highstakes", "hh": "hu_harm", "ins": "instructions"}[con]
    fig.append(f"\\addplot[{CEIL_STYLE}] coordinates {{(0,{CEIL[ck]}) (10,{CEIL[ck]})}};")
    if i == 1: fig.append("\\addlegendentry{{ceiling}}")
    best = max(st.mean(MERGED[ck]["detailed"][g][590]) for g in GENS if MERGED[ck]["detailed"][g].get(590))  # best directly prompted set at 590 (Table sizecurves)
    fig.append(f"\\addplot[color=blue!60!black, densely dashed, thick, no marks] coordinates {{(0,{best:.4f}) (10,{best:.4f})}};")
    if i == 1: fig.append("\\addlegendentry{{{{best direct prompting, 590 samples}}}}")
fig += ["\\end{groupplot}", "\\node at ($(group c2r1.south)+(0,-1.1cm)$) {\\pgfplotslegendfromname{rtlegend}};", "\\end{tikzpicture}"]
(FIG / "redteam_iterations.tex").write_text("\n".join(fig) + "\n")

# Subset-base grids (hs, instructions): AUROC and lift vs own base by configuration and number of attackers k
lines = []
for br, con, name in [("experiment_hs_last", "hs", "High-stakes"), ("experiment_instruction_last", "ins", "Instruction-following")]:
    rows = load(f"data/{br}__combined_draws_subsetbase.csv")
    by = defaultdict(list)
    for r in rows:
        if r["dataset"] != "mean": continue
        g, codes = r["combo"].split("_", 1)
        codes = "".join(sorted(codes))  # base_dg and att_gd name the same attacker subset
        by[(g, codes)].append((float(r["auroc"]), int(r["n_kept"])))
    codes = sorted(set(c for _, c in by), key=lambda c: (len(c), c))
    lines.append(grp(name, 8))
    for g, gtex in [("memo", "general"), ("att", "detailed eval description")]:
        for k in [1, 2, 3]:
            cells = [c for c in codes if len(c) == k and (g, c) in by and ("base", c) in by]
            if not cells: continue
            base = st.mean(st.mean(a for a, _ in by[("base", c)]) for c in cells)
            m = st.mean(st.mean(a for a, _ in by[(g, c)]) for c in cells)
            lifts = [st.mean(a for a, _ in by[(g, c)]) - st.mean(a for a, _ in by[("base", c)]) for c in cells]
            nrt = st.mean(st.mean(n for _, n in by[(g, c)]) for c in cells)
            lines.append(f"\\quad {gtex} & {k} & {len(cells)} & {50*k} & {nrt:.0f} & {base:.3f} & {m:.3f} & {st.mean(lifts):+.3f} ({sum(l > 0 for l in lifts)}/{len(lifts)}) \\\\")
(OUT / "redteam_subsets.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- appendix: every red-team arm, one panel per concept x attacker
# dotted reference = the same generator's detailed-prompt direct-prompting set in full (600 rows + its 50-row base), one fit
ref = {}
for key, name, path, _ in CONCEPTS:
    for r in load(path):
        if "evaldesc" in r["samples"] and int(r["n"]) == 600:
            ref[({"highstakes": "hs", "hu_harm": "hh", "instructions": "ins"}[key], gen_of(r["samples"]))] = eval_mean(r)
ATT_ORDER = ["llama70b", "deepseekv4pro", "gptoss", "nemotron"]
def att_key(con, a):  # gpt-oss is tagged gptoss120b on the hs/hh branches and gptoss on the instructions branch
    return (con, "gptoss120b" if (a == "gptoss" and con in ("hs", "hh")) else a)
fig = ["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=4 by 3, horizontal sep=0.55cm, vertical sep=1.1cm}, width=0.27\\linewidth, height=3.6cm,",
       "  xmin=0, xmax=10, xtick={0,2,4,6,8,10}, grid=major, tick label style={font=\\tiny}, label style={font=\\scriptsize}, title style={font=\\scriptsize},",
       "  legend style={font=\\scriptsize, draw=none}, legend columns=3]"]
for ri, (br, (con, name)) in enumerate(CON.items()):
    lo = {"hs": 0.70, "hh": 0.75, "ins": 0.60}[con]
    for ci, a in enumerate(ATT_ORDER):
        opts = [f"ymin={lo}", "ymax=0.96"]
        if ri == 0: opts.append(f"title={{{ATT_TEX[a]}}}")
        if ci == 0: opts.append(f"ylabel={{{name.split(' (')[0]}}}")
        if ri == 2: opts.append("xlabel={iteration}")
        if ri == 0 and ci == 0: opts.append("legend to name=armlegend")
        fig.append("\\nextgroupplot[" + ", ".join(opts) + "]")
        for cfg in CFGS:
            d = series.get((*att_key(con, a), cfg))
            if not d: continue
            col, mk, ls = cfg_style[cfg]
            fig.append(f"\\addplot[color={col}, mark={mk}, mark size=1.1pt, thick, {ls}] coordinates {{" + " ".join(f"({i},{d[i]:.4f})" for i in sorted(d)) + "};")
            if ri == 0 and ci == 0: fig.append(f"\\addlegendentry{{{{{CFG_TEX[cfg]}}}}}")
        rv = ref.get((con, a))
        if rv is not None:
            fig.append(f"\\addplot[color=black, densely dotted, thick, no marks] coordinates {{(0,{rv:.4f}) (10,{rv:.4f})}};")
            if ri == 0 and ci == 0: fig.append("\\addlegendentry{{direct prompting, detailed, 650 samples}}")
fig += ["\\end{groupplot}", "\\node at ($(group c2r3.south east)+(0,-1.05cm)$) {\\pgfplotslegendfromname{armlegend}};", "\\end{tikzpicture}"]
(FIG / "redteam_arms.tex").write_text("\n".join(fig) + "\n")
print("refs:", {k: round(v, 3) for k, v in ref.items()})

# ---------------------------------------------------------------- split-targeted (shape-free) sets, generated samples alone: on-target AUROC vs n
# instructions / high-stakes: origin/per_split_studies2 scripts/{concept}_tgtmin_size_curve_nobase.csv (n>=60, accum 4) and
#   ..._nobase_accum1.csv (n=10/15/30, accum 1; the accum-4 fits at those sizes take no optimizer step and are discarded). DeepSeek-V4-Pro.
# harm: origin/generator_experiment_1 analysis/refit_studies/hu_harm_split_targeted_{refusal,dilemmas}/<cond>_{n10b,n15b,f5b,f10b,f20b,f50b,f90}_d*.json
#   (request_minimal = the one shape-free prompt shared by Ant-HH and Refusal; ai_dilemmas_minimal; daily_dilemmas_minimal). Llama-3.3-70B-Instruct.
import json
TGT_SIZES = [10, 15, 30, 60, 120, 300, 540]
tgt = {}
for key in ["instructions", "highstakes"]:
    C = defaultdict(lambda: defaultdict(list))
    for r in load(f"data/{key}_tgtmin_size_curve_nobase.csv"):
        if int(r["n"]) < 60: continue
        sp = re.search(r'tgtmin_(.+)_600\.jsonl', r["samples"]).group(1)
        C[sp][int(r["n"])].append(float(r["eval_" + sp]))
    for r in load(f"data/{key}_tgtmin_size_curve_nobase_accum1.csv"):
        sp = re.search(r'tgtmin_(.+)_600\.jsonl', r["samples"]).group(1)
        C[sp][int(r["n"])].append(float(r["eval_" + sp]))
    tgt[key] = C
C = defaultdict(lambda: defaultdict(list))
for r in load("data/hu_harm_tgtmin_size_curve_nobase.csv"):
    C[r["split"]][int(r["n"])].append(float(r["auroc"]))
tgt["hu_harm"] = C
SPLIT_ORDER = {"highstakes": ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"],
               "hu_harm": ["ai_dilemmas", "daily_dilemmas", "balanced_refusal", "ant_hh"],
               "instructions": ["anthropic_harmless_refusal", "bbq_substitution", "mm_substitution", "hc_contradiction", "hc_context_drift", "oig_context_drift"]}
SPLIT_STYLE = ["color=blue, mark=*", "color=red, mark=square*", "color=green!60!black, mark=triangle*", "color=orange, mark=diamond*",
               "color=violet, mark=pentagon*", "color=teal, mark=otimes*"]
fig = ["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.9cm}, width=0.36\\linewidth, height=4.6cm,",
       "  xmode=log, log basis x=10, xlabel={generated samples}, xtick={10,30,60,120,300,540}, xticklabels={10,30,60,120,300,540}, ymin=0.45, ymax=1.0, grid=major,",
       "  tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small},",
       "  legend style={font=\\tiny, draw=none}, legend columns=2, error bars/y dir=both, error bars/y explicit]"]
tgt_summary = {}
for i, (key, name, _, _) in enumerate(CONCEPTS):
    fig.append(f"\\nextgroupplot[title={{{name}}}, legend to name=tgtlegend{i}" + (", ylabel={on-target AUROC}" if i == 0 else "") + "]")
    for j, sp in enumerate(SPLIT_ORDER[key]):
        pts = [(n, st.mean(tgt[key][sp][n]), st.pstdev(tgt[key][sp][n]) if len(tgt[key][sp][n]) > 1 else 0.0) for n in TGT_SIZES if tgt[key][sp].get(n)]
        fig.append(f"\\addplot[{SPLIT_STYLE[j]}, mark size=1.3pt, thick] coordinates {{" + " ".join(f"({n},{m:.4f}) +- (0,{sd:.4f})" for n, m, sd in pts) + "};")
        fig.append(f"\\addlegendentry{{{{{SPLN[sp]}}}}}")
        tgt_summary[(key, sp)] = {n: round(m, 3) for n, m, _ in pts}
    tgt_summary[(key, "mean")] = {n: round(st.mean(st.mean(tgt[key][sp][n]) for sp in SPLIT_ORDER[key]), 3) for n in TGT_SIZES}
fig += ["\\end{groupplot}"] + [f"\\node at ($(group c{i+1}r1.south)+(0,-1.45cm)$) {{\\pgfplotslegendfromname{{tgtlegend{i}}}}};" for i in range(3)] + ["\\end{tikzpicture}"]
(FIG / "split_targeted_curves.tex").write_text("\n".join(fig) + "\n")
print("split-targeted no-base:", tgt_summary)

# ================================================================ 2026-09-17 additions
# (a) other probe models: pooled size curves (no base, every training sample resampled) on origin/qwen8b, origin/mistralnemo12b,
#     origin/llama1b  scripts/<model>_<concept>_pooled_size_curve.csv  -> data/models/. n = 10/30/110/170/350/590, 8 draws (4 for hs).
# (b) LLM-written prompt variants: origin/toolace_stuff scripts/*_prompts5_sizecurve.csv -> data/prompt_variants/. Five prompts per split
#     written by Claude Opus 5 from the split's dev/eval statistics, 600 DeepSeek-V4-Pro samples each, no base data, on-target AUROC,
#     8 draws per size; n <= 40 at accumulation 1. Ant-HH (harm) was still running: 75..590 only, replica at 300/590 only.
MODELS = [("qwen8b", "Qwen3-8B (L18)"), ("mistralnemo12b", "Mistral-NeMo-12B-Instruct (L20)"), ("llama1b", "Llama-3.2-1B-Instruct (L8)")]
XM = [10, 30, 110, 170, 350, 590]
MC = {}  # (model, concept) -> {arm: {gen: {n: [draws]}}}
for mk, _ in MODELS:
    for key, _, _, _ in CONCEPTS:
        C, _ = pooled_curves(f"data/models/{mk}_{key}_pooled_size_curve.csv")
        MC[(mk, key)] = C
def arm_mean(C, arm, n):
    v = [x for g in GENS for x in C[arm][g].get(n, [])]
    return (st.mean(v), st.pstdev(v)) if v else (None, None)
# table: model x concept x arm, values at XM, delta(110-590), n_sat (within 0.02 of 590)
lines = []; msat = {}
for mk, mname in MODELS:
    lines.append(grp(mname, 10))
    for key, cname, _, _ in CONCEPTS:
        for arm in ("general", "detailed"):
            m = {n: arm_mean(MC[(mk, key)], arm, n)[0] for n in XM}
            sat = sat_point({n: [x for g in GENS for x in MC[(mk, key)][arm][g].get(n, [])] for n in XM}, XM); msat[(mk, key, arm)] = sat
            f110 = (m[110] - m[10]) / (m[590] - m[10])  # headroom-normalised: fraction of the 10->590 gain reached at 110
            lines.append(f"\\quad {cname}, {arm} & " + " & ".join(f"{m[n]:.3f}" for n in XM) + f" & {m[110]-m[590]:+.3f} & {f110:.2f} & {sat_tex(sat)} \\\\")
(OUT / "model_curves.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
print("other-model saturation points (guarded slope rule):", msat)
for mk, _ in MODELS:
    for key, _, _, _ in CONCEPTS:
        for arm in ("general", "detailed"):
            print(mk, key, arm, {n: round(arm_mean(MC[(mk, key)], arm, n)[0], 3) for n in XM})
# figure: rows = models, columns = concepts; general (grey dashed) and detailed (red) four-generator means, error bars = sd over draws x generators
fig = ["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=3 by 3, horizontal sep=0.8cm, vertical sep=1.0cm}, width=0.36\\linewidth, height=3.9cm,",
       "  xmode=log, log basis x=10, xtick={10,30,110,170,350,590}, xticklabels={10,30,110,,350,590}, xmin=8, xmax=700, ymin=0.5, ymax=1.0, grid=major,",
       "  tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small}, legend style={font=\\scriptsize, draw=none}, legend columns=2,",
       "  error bars/y dir=both, error bars/y explicit]"]
for ri, (mk, mname) in enumerate(MODELS):
    for ci, (key, cname, _, _) in enumerate(CONCEPTS):
        opts = []
        if ri == 0: opts.append(f"title={{{cname}}}")
        if ci == 0: opts.append(f"ylabel={{{mname.split(' (')[0]}}}")
        if ri == 2: opts.append("xlabel={generated samples}")
        if ri == 0 and ci == 1: opts.append("legend to name=modellegend")
        fig.append("\\nextgroupplot[" + ", ".join(opts) + "]")
        for arm, sty in (("general", "color=gray, mark=*, dashed"), ("detailed", "color=red, mark=triangle*, solid")):
            pts = [(n, *arm_mean(MC[(mk, key)], arm, n)) for n in XM]
            fig.append(f"\\addplot[{sty}, mark size=1.4pt, thick] coordinates {{" + " ".join(f"({n},{m:.4f}) +- (0,{sd:.4f})" for n, m, sd in pts) + "};")
            if ri == 0 and ci == 1: fig.append(f"\\addlegendentry{{{arm} prompt}}")
fig += ["\\end{groupplot}", "\\node at ($(group c2r3.south)+(0,-1.1cm)$) {\\pgfplotslegendfromname{modellegend}};", "\\end{tikzpicture}"]
(FIG / "model_curves.tex").write_text("\n".join(fig) + "\n")

# ---- (b) prompt variants
PV = [("highstakes_toolace", "toolace_balanced", "ToolACE (\\hs{})", r'_p5_(\w+?)_600'),
      ("highstakes_anthropic_hh", "anthropic_hh_balanced", "Anthropic-HH (\\hs{})", r'_hh5_(\w+?)_600'),
      ("hu_harm_ant_hh", "ant_hh", "Ant-HH (\\harm{})", r'_anthh5_(\w+?)_600'),
      ("hu_harm_ai_dilemmas", "ai_dilemmas", "AI-Dilemmas (\\harm{})", r'_aidil5_(\w+?)_600'),
      ("instructions_hcdrift", "hc_context_drift", "HC-context-drift (\\instr{})", r'_hcd5_(\w+?)_600'),
      ("instructions_oigdrift", "oig_context_drift", "OIG-context-drift (\\instr{})", r'_oigd5_(\w+?)_600')]
PNAME = {"replica": "faithful replica", "grid": "read/write $\\times$ stakes grid", "deceptive": "deceptive domain", "endings": "non-call endings", "longtail": "long-tail domains",
         "twin": "good and bad reply twins", "topicpair": "same topic, opposite stakes", "redteam": "red-team side", "long": "long conversations",
         "intent": "intent counts, not refusal", "nearmiss": "near misses", "victim": "user at risk",
         "failmodes": "other ways to disobey", "domains": "off the topic", "verbose": "answers with reasoning", "overlap": "overlap decoupled"}
PN2 = {("instructions_oigdrift", "long"): "longer threads, later drift", ("hu_harm_ant_hh", "longtail"): "long tail",
       ("hu_harm_ai_dilemmas", "grid"): "decision grid, negation decoupled", ("hu_harm_ai_dilemmas", "domains"): "outside AI governance",
       ("hu_harm_ai_dilemmas", "verbose"): "decisions with reasoning", ("hu_harm_ai_dilemmas", "nearmiss"): "alarming words, safe decisions"}
LEG = {"grid": "rw-grid", "topicpair": "topicpair", "intent": "intent", "verbose": "reasoning", "twin": "twins", "endings": "endings", "replica": "replica", "longtail": "longtail", "deceptive": "deceptive", "long": "long", "redteam": "redteam", "nearmiss": "nearmiss", "failmodes": "failmodes", "domains": "offtopic", "overlap": "overlap", "victim": "victim", ("instructions_oigdrift", "long"): "longthread", ("hu_harm_ai_dilemmas", "grid"): "neg-grid", ("hu_harm_ai_dilemmas", "domains"): "offtopic"}
PSIZES = [10, 20, 40, 75, 150, 300, 590]
pv = {}
for fk, split, title, rx in PV:
    C = defaultdict(lambda: defaultdict(list))
    for r in load(f"data/prompt_variants/{fk}_prompts5_sizecurve.csv"):
        s = re.search(rx, r["samples"]).group(1); C[s][int(r["n"])].append(float(r["eval_" + split]))
    pv[fk] = C
lines = []; pv_summary = {}
for fk, split, title, _ in PV:
    C = pv[fk]; lines.append(grp(title, 9))
    levels = {}
    for s in sorted(C, key=lambda s: -st.mean(C[s][590])):
        m = {n: st.mean(C[s][n]) for n in PSIZES if C[s].get(n)}; levels[s] = m[590]
        sat = sat_point(C[s], PSIZES)
        pv_summary[(fk, s)] = (round(m[590], 3), sat)
        nm = PN2.get((fk, s), PNAME[s]) + " (\\emph{" + LEG.get((fk, s), LEG[s]) + "})"
        lines.append(f"\\quad {nm} & " + " & ".join((f"{m[n]:.3f}" if n in m else "--") for n in PSIZES) + f" & {sat_tex(sat)} \\\\")
    pv_summary[(fk, "spread")] = round(max(levels.values()) - min(levels.values()), 3)
(OUT / "prompt_variants.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
print("prompt variants (level at 590, saturation point):", pv_summary)
PSTY = ["color=blue, mark=*", "color=red, mark=square*", "color=green!60!black, mark=triangle*", "color=orange, mark=diamond*", "color=violet, mark=pentagon*"]
fig = ["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=3 by 2, horizontal sep=0.9cm, vertical sep=2.7cm}, width=0.36\\linewidth, height=4.2cm,",
       "  xmode=log, log basis x=10, xtick={10,20,40,75,150,300,590}, xticklabels={10,20,40,75,150,300,590}, xmin=8, xmax=700, ymin=0.45, ymax=1.0, grid=major,",
       "  tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small}, legend style={font=\\tiny, draw=none, row sep=-3pt}, legend columns=1,",
       "  error bars/y dir=both, error bars/y explicit]"]
for i, (fk, split, title, _) in enumerate(PV):
    C = pv[fk]
    fig.append(f"\\nextgroupplot[title={{{title}}}, legend to name=pvlegend{i}, legend columns=2, legend style={{font=\\tiny, draw=none, row sep=-3pt, column sep=2pt}}" + (", ylabel={on-target AUROC}" if i in (0, 3) else "") + (", xlabel={generated samples}" if i >= 3 else "") + "]")
    for j, s in enumerate(sorted(C, key=lambda s: -st.mean(C[s][590]))):
        pts = [(n, st.mean(C[s][n]), st.pstdev(C[s][n]) if len(C[s][n]) > 1 else 0.0) for n in PSIZES if C[s].get(n)]
        fig.append(f"\\addplot[{PSTY[j]}, mark size=1.3pt, thick] coordinates {{" + " ".join(f"({n},{m:.4f}) +- (0,{sd:.4f})" for n, m, sd in pts) + "};")
        fig.append(f"\\addlegendentry{{{{{LEG.get((fk, s), LEG.get(s, PN2.get((fk, s), PNAME[s])))}}}}}")
fig += ["\\end{groupplot}"]
pos = {0: "c1r1", 1: "c2r1", 2: "c3r1", 3: "c1r2", 4: "c2r2", 5: "c3r2"}
fig += [f"\\node[anchor=north] at ($(group {pos[i]}.south)+(0,-0.72cm)$) {{\\pgfplotslegendfromname{{pvlegend{i}}}}};" for i in range(len(PV))] + ["\\end{tikzpicture}"]
(FIG / "prompt_variants.tex").write_text("\n".join(fig) + "\n")

# ---- (c) every per-split learning curve -> data/curves.json for fit_curves.py (fitted curves, bootstrap, variance decomposition)
import json
ARMS = []
for key, _, _, _ in CONCEPTS:
    for arm in ("general", "detailed"):
        for g in GENS:
            S = MSPLIT[key][arm][g]
            for sp in sorted(set(s for n in S for s in S[n])):
                ARMS.append(dict(model="gemma27b", concept=key, split=sp, recipe=arm, gen=g, variant=arm, curve={n: S[n][sp] for n in S if S[n].get(sp)}))
for mk, _ in MODELS:
    for key, _, _, _ in CONCEPTS:
        _, S = pooled_curves(f"data/models/{mk}_{key}_pooled_size_curve.csv")
        for arm in ("general", "detailed"):
            for g in GENS:
                for sp in sorted(set(s for n in S[arm][g] for s in S[arm][g][n])):
                    ARMS.append(dict(model=mk, concept=key, split=sp, recipe=arm, gen=g, variant=arm, curve={n: S[arm][g][n][sp] for n in S[arm][g] if S[arm][g][n].get(sp)}))
for fk, split, title, _ in PV:
    key = "highstakes" if fk.startswith("highstakes") else ("hu_harm" if fk.startswith("hu_harm") else "instructions")
    for s in pv[fk]:
        ARMS.append(dict(model="gemma27b", concept=key, split=split, recipe="llm", gen="deepseekv4pro", variant=s, curve=dict(pv[fk][s])))
for key in tgt:
    for sp in tgt[key]:
        ARMS.append(dict(model="gemma27b", concept=key, split=sp, recipe="targeted", gen=("llama70b" if key == "hu_harm" else "deepseekv4pro"), variant="targeted", curve=dict(tgt[key][sp])))
json.dump({"arms": ARMS, "SPLN": SPLN, "CONCEPTS": CONCEPTS}, open("data/curves.json", "w"))
print(len(ARMS), "learning curves written to data/curves.json; run `python3 fit_curves.py` to refit")
