"""figures/dc_rgt.tex: R, G and T per concept, generated (filled) and real dev-set (open) medians, with the
per-split values behind them. Generated per-split R and G are the medians over used cells (as tables/dc_splits.tex),
per-split T = t_gen2gen from data/dev_coverage/dc_xfer_cells.csv (probe on one kind scored on the set's other kinds);
real per-split R and G from dc_dev_partc_ratios.csv (usable splits), T = t_dev2dev from dc_xfer_cells.csv
(probe on one dev split scored on the other dev splits). All three concepts present (high-stakes from commit 6eec832).
Concept medians are those of tables/dc_main.tex."""
import csv, collections, statistics as st
from pathlib import Path
D = Path(__file__).parent / "data"
C = ["instructions", "hu_harm", "highstakes"]
X = {"instructions": 1, "hu_harm": 2, "highstakes": 3}
gen = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(D / "direction_count" / "dc_ratios.csv")):
    if r["usable"] == "1":
        gen[(r["concept"], r["split"])]["R"].append(float(r["R"])); gen[(r["concept"], r["split"])]["G"].append(float(r["G"]))
genT = collections.defaultdict(list)
for r in csv.DictReader(open(D / "dev_coverage" / "dc_xfer_cells.csv")):
    if r["t_gen2gen"]: genT[(r["concept"], r["split"])].append(float(r["t_gen2gen"]))
real = {}
for r in csv.DictReader(open(D / "dev_coverage" / "dc_dev_partc_ratios.csv")):
    if r["usable"] == "1":
        real[(r["concept"], r["split"])] = {"R": float(r["R_dev"]), "G": float(r["G_dev_fit"])}
for r in csv.DictReader(open(D / "dev_coverage" / "dc_xfer_cells.csv")):
    if r["t_dev2dev"]: real.setdefault((r["concept"], r["split"]), {})["T"] = float(r["t_dev2dev"])
pts = {"R": {"gen": [], "real": []}, "G": {"gen": [], "real": []}, "T": {"gen": [], "real": []}}
for (c, s), d in gen.items():
    pts["R"]["gen"].append((X[c], st.median(d["R"]))); pts["G"]["gen"].append((X[c], st.median(d["G"])))
for (c, s), v in genT.items(): pts["T"]["gen"].append((X[c], st.mean(v)))
for (c, s), d in real.items():
    for k in ("R", "G", "T"):
        if k in d: pts[k]["real"].append((X[c], d[k]))
# concept-level markers are the values of tables/dc_main.tex (generated: medians over used cells; real: over splits)
med = {"R": {"gen": {}, "real": {}}, "G": {"gen": {}, "real": {}}, "T": {"gen": {}, "real": {}}}
_x = 0
for line in open(Path(__file__).parent / "tables" / "dc_main.tex"):
    cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
    if len(cells) < 10: continue
    if cells[0]: _x += 1
    src = "gen" if cells[1].startswith("generated") else "real"
    med["R"][src][_x], med["G"][src][_x] = float(cells[7]), float(cells[8])
    if cells[9] not in ("--", "", "pending"): med["T"][src][_x] = float(cells[9])
for k in med: print(k, {src: {x: round(y, 2) for x, y in m.items()} for src, m in med[k].items()})
def coords(v, dx, ymax=None):
    return " ".join(f"({x+dx},{min(y, ymax) if ymax else y:.3f})" for x, y in v)
panels = [("R", r"$R=m_{\text{mixed}}/m_{\text{own}}$", "ymode=log, ymin=0.4, ymax=40, ytick={0.5,1,2,4,6,10,30}, yticklabels={0.5,1,2,4,6,10,30}", None),
          ("G", r"$G$ (gain kept without own split-kind)", "ymin=0, ymax=1.4, ytick={0,0.5,1}", 1.3),
          ("T", r"$T$ (transfer AUROC)", "ymin=0.45, ymax=1.02, ytick={0.5,0.7,0.9}", None)]
out = [r"\begin{tikzpicture}", r"\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.95cm}, width=4.7cm, height=3.7cm, scale only axis=false,",
       r"  xmin=0.5, xmax=3.5, xtick={1,2,3}, xticklabels={\instr{},\harm{},\hs{}}, x tick label style={font=\tiny}, tick label style={font=\scriptsize}, title style={font=\scriptsize, yshift=-1ex}, ymajorgrids, grid style={black!10}, clip=false]"]
for k, title, opts, ymax in panels:
    out.append(rf"\nextgroupplot[title={{{title}}}, {opts}]")
    if k == "R":
        out.append(r"\draw[dashed, black!50] (axis cs:0.5,1) -- (axis cs:3.5,1);")
        for x, n in ((1, 6), (2, 4), (3, 4)):
            out.append(rf"\draw[dotted, black!60, line width=0.8pt] (axis cs:{x-0.42},{n}) -- (axis cs:{x+0.42},{n});")
    if k == "T":
        out.append(r"\draw[dashed, black!50] (axis cs:0.5,0.5) -- (axis cs:3.5,0.5);")
    out.append(rf"\addplot[only marks, mark=*, mark size=1pt, color=black!35] coordinates {{{coords(pts[k]['gen'], -0.18, ymax)}}};")
    out.append(rf"\addplot[only marks, mark=o, mark size=1pt, color=black!45] coordinates {{{coords(pts[k]['real'], 0.18, ymax)}}};")
    out.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, color=red] coordinates {{{coords([(x, med[k]['gen'][x]) for x in (1,2,3) if x in med[k]['gen']], -0.18)}}};")
    out.append(rf"\addplot[only marks, mark=o, mark size=2.4pt, color=red, line width=0.9pt] coordinates {{{coords([(x, med[k]['real'][x]) for x in (1,2,3) if x in med[k]['real']], 0.18)}}};")
    if k == "T" and 3 not in med[k]["gen"]:
        out.append(r"\node[font=\tiny, black!45, align=center] at (axis cs:3,0.73) {pending};")
    if k == "G":
        out.append(r"\node[font=\tiny, black!45, anchor=west, inner sep=1pt] at (axis cs:1.88,1.3) {3.6};")
out += [r"\end{groupplot}", r"\end{tikzpicture}"]
(Path(__file__).parent / "figures" / "dc_rgt.tex").write_text("\n".join(out) + "\n")
print("wrote figures/dc_rgt.tex")
