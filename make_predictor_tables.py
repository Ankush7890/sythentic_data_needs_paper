"""Tables for Appendix "Can the half-gain size be predicted before generating?".

Reads the per-split predictors and the split-level statistics produced by
scripts/knee_predictor.py on the experiment repo's knee_predictor branch
(data/knee_predictors.csv, data/knee_predictor_stats.csv,
data/knee_predictor_scatter.csv) and writes tables/predictor_stats.tex and
tables/predictor_splits.tex.  Run: python3 make_predictor_tables.py
"""
import csv, math

P = {r["split"]: r for r in csv.DictReader(open("data/knee_predictors.csv"))}
S = {r["predictor"]: r for r in csv.DictReader(open("data/knee_predictor_stats.csv"))
     if r["target"] == "log_m_detailed"}
T = {r["split"]: r for r in csv.DictReader(open("data/knee_predictor_scatter.csv"))}

NAME = {"anthropic_hh_balanced": "Anthropic-HH", "mt_balanced": "MT-Samples",
        "mts_balanced": "MTS-Dialog", "toolace_balanced": "ToolACE",
        "ai_dilemmas": "AI-Dilemmas", "ant_hh": "Ant-HH", "balanced_refusal": "Refusal",
        "daily_dilemmas": "Daily-Dilemmas", "anthropic_harmless_refusal": "Harmless-refusal",
        "bbq_substitution": "BBQ-substitution", "hc_context_drift": "HC-context-drift",
        "hc_contradiction": "HC-contradiction", "mm_substitution": "MM-substitution",
        "oig_context_drift": "OIG-context-drift"}
CONCEPTS = [("\\emph{High-stakes}", "highstakes"), ("\\emph{Harmful-to-human}", "hu_harm"),
            ("\\emph{Instruction-following}", "instructions")]

# (stats key, label)
PRED = [("m_ID_mean_C0.1", "$m_{\\mathrm{ID}}$, in-distribution half-gain size ($C{=}0.1$)"),
        ("m_ID_mean_C1", "$m_{\\mathrm{ID}}$ ($C{=}1$)"),
        ("auroc_ID_k8_mean_C1", "in-distribution AUROC from 8 samples"),
        ("auroc_full_mean", "in-distribution AUROC, full model"),
        ("r1_mean", "$r_1$, one-direction share"),
        ("auroc_pca_d1_mean", "AUROC on the first principal component"),
        ("log_d95_mean", "$\\log_{10} d_{95}$, components to 95\\% of gain"),
        ("fisher_dom_mean", "Fisher ratio along the class-mean direction"),
        ("centroid_dist_rel_mean", "centroid distance / within-class sd"),
        ("paired", "class-paired split (0/1)"),
        ("frac_truncated", "fraction of samples at the token cap"),
        ("n_rows", "split size")]

def f(x, d=2, sign=False):
    if x in ("", None): return "--"
    v = float(x)
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s

with open("tables/predictor_stats.tex", "w") as out:
    for key, label in PRED:
        r = S[key]
        ci = f"[{f(r['ci_lo'],2,True)}, {f(r['ci_hi'],2,True)}]" if r["ci_lo"] else "--"
        rins = f(r["rho_instructions"], 2, True) if r["rho_instructions"] else "--"
        out.write(f"{label} & {f(r['rho'],2,True)} & {f(r['p_perm'],2)} & {ci} & "
                  f"{f(r['loo_rmse_pred'],2)} & {f(r['loo_rmse_pred_concept'],2)} & "
                  f"{f(r['curve_r2_gain'],2)} & {rins} \\\\\n")
    out.write("\\bottomrule\n")
r0 = S["m_ID_mean_C0.1"]
print("baselines: grand", f(r0["loo_rmse_grand"], 3), "concept", f(r0["loo_rmse_concept"], 3),
      "ceiling", f(r0["curve_r2_split_ceiling"], 3))

def m_id(x):
    v = float(x)
    return "$<$3" if v <= 3.0001 else f"{v:.0f}"

with open("tables/predictor_splits.tex", "w") as out:
    for head, c in CONCEPTS:
        out.write(f"{head} & & & & & & & \\\\\n")
        splits = sorted([s for s in P if P[s]["concept"] == c], key=lambda s: float(T[s]["log_m"]))
        for s in splits:
            p, t = P[s], T[s]
            lm = f"{float(t['log_m']):.2f} [{float(t['log_m_lo']):.2f}, {float(t['log_m_hi']):.2f}]"
            d95 = float(p["d95_mean"]); d95s = "all" if d95 >= 5000 else f"{d95:.0f}"
            out.write(f"\\quad {NAME[s]} & {lm} & {m_id(p['m_ID_mean_C0.1'])} & "
                      f"{float(p['auroc_full_mean']):.3f} & {float(p['r1_mean']):+.2f} & {d95s} & "
                      f"{float(p['fisher_dom_mean']):.2f} & {p['n_rows']} \\\\\n")
    out.write("\\bottomrule\n")
print("wrote tables/predictor_stats.tex and tables/predictor_splits.tex")
