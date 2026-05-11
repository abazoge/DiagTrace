import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import f as f_dist
import os

from utils import get_data, llm_types

# ==============================================================================
# CONFIGURATION
# ==============================================================================
output_dir = "../tables"
os.makedirs(output_dir, exist_ok=True)

model_pairs = [
    {"en": "o3_EN",         "fr": "o3_FR",         "name": "o3"},
    {"en": "DeepSeek_EN",   "fr": "DeepSeek_FR",   "name": "DeepSeek-R1"},
    {"en": "GPT_EN",        "fr": "GPT_FR",        "name": "GPT-4"},
    {"en": "Llama_EN",      "fr": "Llama_FR",      "name": "Llama-405B"},
    {"en": "Biomistral_EN", "fr": "Biomistral_FR",  "name": "BioMistral"},
]

model_order = ["o3", "DeepSeek-R1", "GPT-4", "Llama-405B", "BioMistral"]

sub_scores = [
    ("Note", 18, "Overall score (0-18)",         "icc"),
    ("D",     3, "Final diagnosis (0-3)",        "kappa_w"),
    ("VI",    5, "Internal validity (0-5)",      "kappa_w"),
    ("VE",    3, "External validity (0-3)",      "kappa_w"),
    ("H",     1, "Differential diagnosis (0-1)", "kappa"),
    ("L",     4, "Logical structure (0-4)",      "kappa_w"),
    ("E",     2, "Expression (0-2)",             "kappa_w"),
]

score_order = [s[2] for s in sub_scores]

n_bootstrap = 2000
random_state = 42


# ==============================================================================
# ICC(2,1) — two-way random, single measures, absolute agreement
# ==============================================================================
def icc_2_1(rater1, rater2):
    r1 = np.array(rater1, dtype=float)
    r2 = np.array(rater2, dtype=float)
    n = len(r1)
    k = 2

    ratings = np.column_stack([r1, r2])
    grand_mean = np.mean(ratings)

    subj_means = np.mean(ratings, axis=1)
    rater_means = np.mean(ratings, axis=0)

    SS_between = k * np.sum((subj_means - grand_mean) ** 2)
    SS_judges  = n * np.sum((rater_means - grand_mean) ** 2)
    SS_total   = np.sum((ratings - grand_mean) ** 2)
    SS_error   = SS_total - SS_between - SS_judges

    df_between = n - 1
    df_error   = (n - 1) * (k - 1)

    BMS = SS_between / df_between
    JMS = SS_judges / max(k - 1, 1)
    EMS = SS_error / max(df_error, 1)

    icc = (BMS - EMS) / (BMS + (k - 1) * EMS + k * (JMS - EMS) / n)

    F_value = BMS / max(EMS, 1e-10)
    p_value = 1 - f_dist.cdf(F_value, df_between, df_error)

    return icc, F_value, p_value


def icc_2_1_bootstrap(rater1, rater2, n_bootstrap=2000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    r1 = np.array(rater1, dtype=float)
    r2 = np.array(rater2, dtype=float)
    n = len(r1)

    icc_val, f_val, p_val = icc_2_1(r1, r2)

    icc_boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            val, _, _ = icc_2_1(r1[idx], r2[idx])
            if np.isfinite(val):
                icc_boot.append(val)
        except Exception:
            continue

    if len(icc_boot) < 10:
        return icc_val, np.nan, np.nan, p_val

    ci_lower = np.percentile(icc_boot, 100 * alpha / 2)
    ci_upper = np.percentile(icc_boot, 100 * (1 - alpha / 2))

    return icc_val, ci_lower, ci_upper, p_val


# ==============================================================================
# Cohen's Kappa with bootstrap CI
# ==============================================================================
def cohen_kappa_ci(x, y, weights=None, n_bootstrap=2000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    kappa = cohen_kappa_score(x, y, weights=weights)

    kappas_boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            k = cohen_kappa_score(x[idx], y[idx], weights=weights)
            if np.isfinite(k):
                kappas_boot.append(k)
        except Exception:
            continue

    if len(kappas_boot) < 10:
        return kappa, np.nan, np.nan

    ci_lower = np.percentile(kappas_boot, 100 * alpha / 2)
    ci_upper = np.percentile(kappas_boot, 100 * (1 - alpha / 2))

    return kappa, ci_lower, ci_upper


# ==============================================================================
# Interpretation
# ==============================================================================
def interpret_icc(val):
    if np.isnan(val): return "--"
    if val < 0.40: return "Poor"
    if val < 0.60: return "Fair"
    if val < 0.75: return "Good"
    return "Excellent"

def interpret_kappa(val):
    if np.isnan(val): return "--"
    if val < 0.00: return "Poor"
    if val < 0.20: return "Slight"
    if val < 0.40: return "Fair"
    if val < 0.60: return "Moderate"
    if val < 0.80: return "Substantial"
    return "Almost perfect"


def fmt_val_ci(val, ci_lo, ci_hi):
    if np.isnan(val):
        return "--"
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return f"{val:.3f}"
    return f"{val:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]"


# ==============================================================================
# COMPUTE ALL AGREEMENT
# ==============================================================================
print("Computing inter-rater agreement...")

all_results = []

for mp in model_pairs:
    print(f"  {mp['name']}...")
    for lang_key, lang_label in [("en", "English"), ("fr", "French")]:
        sheet = mp[lang_key]
        data = get_data(sheet)

        n_total = len(data)
        n_half = n_total // 2

        for col, max_score, label, method in sub_scores:
            scores = np.array(data[col], dtype=float)
            rater1 = scores[:n_half]
            rater2 = scores[n_half:]

            valid = ~(np.isnan(rater1) | np.isnan(rater2))
            r1 = rater1[valid]
            r2 = rater2[valid]

            # Default values
            val, ci_lo, ci_hi = np.nan, np.nan, np.nan
            interp = "--"

            if len(r1) < 5:
                pass
            elif method in ("kappa", "kappa_w"):
                unique_r1 = np.unique(r1)
                unique_r2 = np.unique(r2)
                if len(unique_r1) < 2 and len(unique_r2) < 2:
                    pct_agree = np.mean(r1 == r2) * 100
                    interp = f"Ceiling ({pct_agree:.0f}% agree)"
                else:
                    w = "quadratic" if method == "kappa_w" else None
                    val, ci_lo, ci_hi = cohen_kappa_ci(
                        r1, r2, weights=w,
                        n_bootstrap=n_bootstrap, random_state=random_state)
                    interp = interpret_kappa(val)
            elif method == "icc":
                val, ci_lo, ci_hi, _ = icc_2_1_bootstrap(
                    r1, r2, n_bootstrap=n_bootstrap, random_state=random_state)
                interp = interpret_icc(val)

            all_results.append({
                "Score": label,
                "Model": mp["name"],
                "Language": lang_label,
                "Method": "ICC" if method == "icc" else ("Weighted κ" if method == "kappa_w" else "κ"),
                "Value": val,
                "CI_lower": ci_lo,
                "CI_upper": ci_hi,
                "Interpretation": interp,
            })

df_all = pd.DataFrame(all_results)

# ==============================================================================
# TABLE A: Detailed agreement — Score × Model, EN and FR side by side
# Sorted by score order, then model order
# ==============================================================================
print("Building detailed agreement table...")

rows_detailed = []
for col, max_score, label, method in sub_scores:
    for mp_name in model_order:
        en = df_all[(df_all["Score"] == label) &
                    (df_all["Model"] == mp_name) &
                    (df_all["Language"] == "English")].iloc[0]
        fr = df_all[(df_all["Score"] == label) &
                    (df_all["Model"] == mp_name) &
                    (df_all["Language"] == "French")].iloc[0]

        rows_detailed.append({
            "Score": label,
            "Model": mp_name,
            "Method": en["Method"],
            "EN Value [95% CI]": fmt_val_ci(en["Value"], en["CI_lower"], en["CI_upper"]),
            "EN Interpretation": en["Interpretation"],
            "FR Value [95% CI]": fmt_val_ci(fr["Value"], fr["CI_lower"], fr["CI_upper"]),
            "FR Interpretation": fr["Interpretation"],
        })

df_detailed = pd.DataFrame(rows_detailed)
df_detailed.to_csv(os.path.join(output_dir, "icc_detailed.tsv"), sep="\t", index=False)
print("  Saved icc_detailed.tsv")

# ==============================================================================
# TABLE B: Summary — Overall score ICC only (compact for main paper)
# ==============================================================================
print("Building ICC summary table...")

rows_summary = []
for mp_name in model_order:
    en = df_all[(df_all["Score"] == "Overall score (0-18)") &
                (df_all["Model"] == mp_name) &
                (df_all["Language"] == "English")].iloc[0]
    fr = df_all[(df_all["Score"] == "Overall score (0-18)") &
                (df_all["Model"] == mp_name) &
                (df_all["Language"] == "French")].iloc[0]

    rows_summary.append({
        "Model": mp_name,
        "ICC EN [95% CI]": fmt_val_ci(en["Value"], en["CI_lower"], en["CI_upper"]),
        "EN Interpretation": en["Interpretation"],
        "ICC FR [95% CI]": fmt_val_ci(fr["Value"], fr["CI_lower"], fr["CI_upper"]),
        "FR Interpretation": fr["Interpretation"],
    })

df_summary = pd.DataFrame(rows_summary)
df_summary.to_csv(os.path.join(output_dir, "icc_summary.tsv"), sep="\t", index=False)
print("  Saved icc_summary.tsv")

# ==============================================================================
# TABLE C: Overall agreement per score (pooled across models/languages)
# ==============================================================================
print("Building pooled agreement table...")

rows_pooled = []
for col, max_score, label, method in sub_scores:
    sub = df_all[(df_all["Score"] == label) & df_all["Value"].notna()]
    if len(sub) == 0:
        rows_pooled.append({
            "Score": label,
            "Method": "ICC" if method == "icc" else ("Weighted κ" if method == "kappa_w" else "κ"),
            "Mean": "--",
            "Min": "--",
            "Max": "--",
            "Interpretation": "--",
        })
    else:
        mean_val = sub["Value"].mean()
        min_val = sub["Value"].min()
        max_val = sub["Value"].max()

        if method == "icc":
            interp = interpret_icc(mean_val)
        else:
            interp = interpret_kappa(mean_val)

        rows_pooled.append({
            "Score": label,
            "Method": "ICC" if method == "icc" else ("Weighted κ" if method == "kappa_w" else "κ"),
            "Mean": f"{mean_val:.3f}",
            "Min": f"{min_val:.3f}",
            "Max": f"{max_val:.3f}",
            "Interpretation": interp,
        })

df_pooled = pd.DataFrame(rows_pooled)
df_pooled.to_csv(os.path.join(output_dir, "icc_pooled.tsv"), sep="\t", index=False)
print("  Saved icc_pooled.tsv")

# ==============================================================================
# PRINT ALL TABLES
# ==============================================================================
print("\n" + "=" * 80)
print("TABLE A — Detailed inter-rater agreement (per score, per model, EN vs FR)")
print("=" * 80)
print(df_detailed.to_string(index=False))

print("\n" + "=" * 80)
print("TABLE B — ICC summary (overall score only)")
print("=" * 80)
print(df_summary.to_string(index=False))

print("\n" + "=" * 80)
print("TABLE C — Pooled agreement (mean across models and languages)")
print("=" * 80)
print(df_pooled.to_string(index=False))

print(f"\nAll TSV files saved to: {output_dir}/")