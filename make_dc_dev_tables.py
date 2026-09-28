"""tables/dc_dev.tex: the coverage test on dev samples (branch dev_coverage, data/dev_coverage/)
against the generated-set values of tables/dc_summary.tex. Per concept: median kind-only and
mixed half-gain size, R and fitted G (Part C, medians over splits with 95% bootstrap over
splits), the full-size transfer AUROC (Part A), each beside the generated-set value."""
import csv
from pathlib import Path
D = Path(__file__).parent / "data" / "dev_coverage"
share = {(r["row"], r["concept"], r["vs"]): r for r in csv.DictReader(open(D / "dc_dev_partc_share.csv"))}
cells = list(csv.DictReader(open(D / "dc_dev_cells.csv")))
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
    rows.append(f"{NAME[c]} & {r['n_usable']}/{r['n_splits']} & {m(r['median_m_own'])} ({r['gen_m_kind']}) & {m(r['median_m_all'])} ({r['gen_m_mixed']}) & "
                f"{f(r['median_R_dev']):.2f} [{f(r['R_lo']):.1f}, {f(r['R_hi']):.1f}] ({r['gen_R']}) & "
                f"{f(r['median_G_dev_fit']):.2f} ({GEN_G[c]:.2f}) & {t:.2f} ({GEN_T[c]:.2f}) \\\\")
(Path(__file__).parent / "tables" / "dc_dev.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
s = share[("share", "instructions", "both")]
print("share both", s["share"], s["share_lo"], s["share_hi"], "| vs harm", share[("share","instructions","hu_harm")]["share"], "| vs hs", share[("share","instructions","highstakes")]["share"], share[("share","instructions","highstakes")]["share_lo"], share[("share","instructions","highstakes")]["share_hi"])
print("gaps", s["gap_all"], s["gap_own"])
