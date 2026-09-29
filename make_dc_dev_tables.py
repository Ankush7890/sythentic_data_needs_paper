"""tables/dc_dev.tex: the coverage test on dev samples (branch dev_coverage, data/dev_coverage/)
against the generated-set values of tables/dc_summary.tex. Per concept: median kind-only and
mixed half-gain size, fitted level U of both arms, R and fitted G (Part C, medians over splits; the bracket on R is the
leave-one-split-out range of the median), the full-size transfer AUROC (Part A), each beside the generated-set value."""
import csv
from pathlib import Path
D = Path(__file__).parent / "data" / "dev_coverage"
share = {(r["row"], r["concept"], r["vs"]): r for r in csv.DictReader(open(D / "dc_dev_partc_share.csv"))}
cells = list(csv.DictReader(open(D / "dc_dev_cells.csv")))
ratios = list(csv.DictReader(open(D / "dc_dev_partc_ratios.csv")))
gen_cells = list(csv.DictReader(open(D.parent / "direction_count" / "dc_ratios.csv")))   # generated-set cells: U_a kind-only, U_b mixed
import statistics as st
NAME = {"instructions": r"\instr{}", "hu_harm": r"\harm{}", "highstakes": r"\hs{}"}
GEN_G = {"instructions": 0.67, "hu_harm": 0.90, "highstakes": 0.95}
GEN_T = {"instructions": 0.81, "hu_harm": 0.85, "highstakes": 0.95}
f = float
def m(x): x = f(x); return f"{x:.0f}" if x >= 20 else f"{x:.1f}"
rows = []
for c in ["instructions", "hu_harm", "highstakes"]:
    r = share[("concept", c, "")]
    t = st.median(f(x["t_other_dev"]) for x in cells if x["concept"] == c)
    Rs = [(x["split"], f(x["R_dev"])) for x in ratios if x["concept"] == c and x["usable"] == "1"]
    loo = [st.median(v for s2, v in Rs if s2 != s) for s, _ in Rs]   # leave-one-split-out range of the median
    Uo = st.median(f(x["U_own"]) for x in ratios if x["concept"] == c and x["usable"] == "1")
    Ua = st.median(f(x["U_all"]) for x in ratios if x["concept"] == c and x["usable"] == "1")
    gcells = [x for x in gen_cells if x["concept"] == c and x["usable"] == "1"]
    gUo = st.median(f(x["U_a"]) for x in gcells); gUa = st.median(f(x["U_b"]) for x in gcells)   # generated kind-only / mixed level
    print(c, "U_own", round(Uo, 3), "gen", round(gUo, 3), "| U_all", round(Ua, 3), "gen", round(gUa, 3), "| gen cells", len(gcells))
    rows.append(f"{NAME[c]} & {r['n_usable']}/{r['n_splits']} & {m(r['median_m_own'])} ({r['gen_m_kind']}) & {m(r['median_m_all'])} ({r['gen_m_mixed']}) & "
                f"{Uo:.2f} ({gUo:.2f}) & {Ua:.2f} ({gUa:.2f}) & "
                f"{f(r['median_R_dev']):.2f} [{min(loo):.1f}, {max(loo):.1f}] ({r['gen_R']}) & "
                f"{f(r['median_G_dev_fit']):.2f} ({GEN_G[c]:.2f}) & {t:.2f} ({GEN_T[c]:.2f}) \\\\")
(Path(__file__).parent / "tables" / "dc_dev.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
s = share[("share", "instructions", "both")]
print("share both", s["share"], s["share_lo"], s["share_hi"], "| vs harm", share[("share","instructions","hu_harm")]["share"], "| vs hs", share[("share","instructions","highstakes")]["share"], share[("share","instructions","highstakes")]["share_lo"], share[("share","instructions","highstakes")]["share_hi"])
print("gaps", s["gap_all"], s["gap_own"])

# tables/dc_dev_curves.tex: the real-sample learning curve per concept (the "all" arm of Part C: class-balanced
# draws of the concept's whole dev set, eight draws per size, early-stopped on the DeepSeek-V4-Pro detailed set,
# scored on the evaluation splits), the counterpart of Table sizecurves on real samples. Ladders stop where the dev
# set runs out (instr 300, harm 170). Columns: mean AUROC at n, Delta = A(80) - A(top), f80 = (A80-A10)/(Atop-A10),
# and the range over splits of the fitted own-kind and mixed half-gain size from dc_dev_partc_fits.csv.
from collections import defaultdict
fits = list(csv.DictReader(open(D / "dc_dev_partc_fits.csv")))
COLS = [10, 30, 80, 110, 170, 300, 350, 590]
out = []
for c in ["highstakes", "hu_harm", "instructions"]:
    by = defaultdict(list)
    for r in csv.DictReader(open(D / f"dc_dev_partc_{c}_all.csv")):
        by[int(r["n"])].append(f(r["eval_mean"]))
    A = {n: st.mean(v) for n, v in by.items()}
    top = max(A)
    cells_ = [f"{A[n]:.3f}" if n in A else "--" for n in COLS]
    d80 = A[80] - A[top]; f80 = (A[80] - A[10]) / (A[top] - A[10])
    mo = [f(r["m"]) for r in fits if r["concept"] == c and r["arm"] == "own"]
    ma = [f(r["m"]) for r in fits if r["concept"] == c and r["arm"] == "all"]
    out.append(f"{NAME[c]} & " + " & ".join(cells_) + f" & {d80:+.3f} & {f80:.2f} & {m(min(mo))}--{m(max(mo))} & {m(min(ma))}--{m(max(ma))} \\\\")
    print(c, "top", top, "A80", round(A[80],3), "Atop", round(A[top],3), "d80", round(d80,3), "f80", round(f80,2), "m_own", sorted(round(x,1) for x in mo), "m_all", sorted(round(x,1) for x in ma))
(Path(__file__).parent / "tables" / "dc_dev_curves.tex").write_text("\n".join(out) + "\n\\bottomrule\n")
