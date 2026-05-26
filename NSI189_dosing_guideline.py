"""
NSI-189 Dosing Guideline Establishment - Body-Weight-Normalized Version
===============================================================================

This script is the body-weight-normalized version of Methodology Section 3.
It should be placed in the same folder as:

    NSI189_integrated_weight_normalized.py

Main idea
---------
Dose in this script is reported as a 70 kg reference dose.
For each virtual subject, the actual administered amount is:

    actual dose_i = reference dose * BW_i / 70 kg

This means "40 mg QD" should be interpreted as:

    40 mg QD for a 70 kg reference individual
    = 0.571 mg/kg QD

Conservative pharmacology choice
--------------------------------
By default, this guideline script uses:

    - Dose scaling: BW^1.0
    - Vb, Vc scaling: BW^1.0
    - CL_app scaling: BW^0.75
    - CL_in, CL_out: NOT body-weight-scaled by default

Reason:
    CL_app is systemic apparent clearance, so BW^0.75 allometric scaling is a
    defensible PBPK-style assumption. However, CL_in and CL_out represent
    plasma-brain transfer clearances, so directly scaling them by body weight is
    less defensible unless supported by literature. Therefore, they are kept as
    residual interindividual variability by default.

If your literature review supports scaling CL_in and CL_out, change:

    BRAIN_TRANSFER_CL_SCALING_MODE = "none"

to:

    BRAIN_TRANSFER_CL_SCALING_MODE = "paired_allometric"

This applies the same BW^0.75 size factor to both CL_in and CL_out, so body
weight does not directly change Kp = CL_in / CL_out.

Outputs
-------
Saved in the same folder as this script:

    - fig9_bw_dose_escalation.png
    - fig10_bw_frequency.png
    - fig11_bw_formulation.png
    - fig12_bw_therapeutic_window.png
    - fig13_bw_dosing_guideline.png
    - dose_escalation_bw_normalized.csv
    - frequency_comparison_bw_normalized.csv
    - formulation_comparison_bw_normalized.csv
    - guideline_population_settings_bw_normalized.csv
"""

import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# 0. IMPORT BODY-WEIGHT-NORMALIZED mPBPK + ELASTICITY MODULE
# =============================================================================

MODULE_NAME = "NSI189_integrated_weight_normalized"

try:
    model = importlib.import_module(MODULE_NAME)
except ImportError as exc:
    raise ImportError(
        "Cannot import NSI189_integrated_weight_normalized.py.\n"
        "Put this file in the same folder as NSI189_integrated_weight_normalized.py, "
        "then run again.\n"
        f"Original error: {exc}"
    )

# PBPK core
simulate_subject = model.simulate_subject
schedule_qd = model.schedule_qd
schedule_bid = model.schedule_bid
schedule_tid = model.schedule_tid
schedule_single_dose = model.schedule_single_dose
PARAM_SPECS = model.PARAM_SPECS
trapezoid_compat = model.trapezoid_compat
extract_pk_metrics = model.extract_pk_metrics

# Elasticity core
REFERENCE_DRUGS = model.REFERENCE_DRUGS
calibrate_elasticity_function = model.calibrate_elasticity_function
get_elasticity_function = model.get_elasticity_function
reinforcement_rate = model.reinforcement_rate
select_best_form = model.select_best_form

# Global paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


# =============================================================================
# 0B. BODY-WEIGHT NORMALIZATION SETTINGS
# =============================================================================

RNG = np.random.default_rng(seed=42)

REFERENCE_BW_KG = 70.0
WEIGHT_MEDIAN_KG = 70.0
WEIGHT_CV = 0.18

# After body-size correction, size-dependent parameters should usually keep only
# residual variability. Increase toward 1.0 if you want a wider, more conservative
# distribution.
RESIDUAL_CV_MULTIPLIER = 0.45

# Default conservative scaling assumptions.
VOLUME_BW_EXPONENT = 1.00
SYSTEMIC_CL_BW_EXPONENT = 0.75
BRAIN_TRANSFER_CL_BW_EXPONENT = 0.75

# Options:
#   "none"              : CL_in and CL_out are not scaled by BW. Safer default.
#   "paired_allometric" : CL_in and CL_out both get BW^0.75, preserving Kp with BW.
BRAIN_TRANSFER_CL_SCALING_MODE = "none"

# Dose normalization is always on in this file.
USE_WEIGHT_NORMALIZED_DOSING = True


# =============================================================================
# 1. THERAPEUTIC WINDOW BOUNDARIES
# =============================================================================

MEC_BRAIN_NGML = 80.0

ALPHA_AMPHETAMINE = 0.012
ALPHA_NICOTINE = 0.0035
ALPHA_COCAINE = 0.0014

# Conservative criterion: NSI-189 should remain above amphetamine-level alpha.
ALPHA_CRITICAL = ALPHA_AMPHETAMINE


# =============================================================================
# 2. POPULATION SAMPLING + DOSE NORMALIZATION HELPERS
# =============================================================================


def lognormal_mu_sigma(median, cv):
    mu = np.log(median)
    sigma = np.sqrt(np.log(1.0 + cv**2))
    return mu, sigma


def sample_lognormal_median_cv(median, cv, size, rng=None):
    rng = rng or RNG
    cv = max(float(cv), 1e-8)
    mu, sigma = lognormal_mu_sigma(median, cv)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def sample_body_weight(n_subjects, rng=None):
    rng = rng or RNG
    return sample_lognormal_median_cv(WEIGHT_MEDIAN_KG, WEIGHT_CV, n_subjects, rng)


def sample_residual_multiplier(cv, n_subjects, shrink=True, rng=None):
    rng = rng or RNG
    residual_cv = cv * RESIDUAL_CV_MULTIPLIER if shrink else cv
    residual_cv = max(float(residual_cv), 1e-8)
    return sample_lognormal_median_cv(1.0, residual_cv, n_subjects, rng)


def sample_guideline_population(n_subjects, rng=None):
    """
    Sample virtual subjects for dosing guideline simulations.

    This function intentionally does not directly call model.sample_population(),
    because we want explicit control over which clearance terms are scaled by BW.
    """
    rng = rng or RNG
    bw = sample_body_weight(n_subjects, rng=rng)
    bw_ratio = bw / REFERENCE_BW_KG

    out = {"BW_kg": bw}

    for name, (median, cv, _, _) in PARAM_SPECS.items():
        if name in ["Vb", "Vc"]:
            # Physiological volumes scale approximately linearly with body size.
            size_factor = bw_ratio ** VOLUME_BW_EXPONENT
            residual = sample_residual_multiplier(cv, n_subjects, shrink=True, rng=rng)
            out[name] = median * size_factor * residual

        elif name == "CL_app":
            # Systemic apparent clearance: allometric scaling.
            size_factor = bw_ratio ** SYSTEMIC_CL_BW_EXPONENT
            residual = sample_residual_multiplier(cv, n_subjects, shrink=True, rng=rng)
            out[name] = median * size_factor * residual

        elif name in ["CL_in", "CL_out"]:
            if BRAIN_TRANSFER_CL_SCALING_MODE == "paired_allometric":
                # Apply same BW exponent to both transfer clearances.
                # This avoids making Kp depend directly on body weight.
                size_factor = bw_ratio ** BRAIN_TRANSFER_CL_BW_EXPONENT
                residual = sample_residual_multiplier(cv, n_subjects, shrink=True, rng=rng)
                out[name] = median * size_factor * residual
            elif BRAIN_TRANSFER_CL_SCALING_MODE == "none":
                # Conservative default: no BW scaling for BBB transfer terms.
                out[name] = sample_lognormal_median_cv(median, cv, n_subjects, rng)
            else:
                raise ValueError(
                    "BRAIN_TRANSFER_CL_SCALING_MODE must be 'none' or "
                    "'paired_allometric'."
                )

        elif name == "ka":
            # Absorption rate is not directly body-size scaled here.
            out[name] = sample_lognormal_median_cv(median, cv, n_subjects, rng)

        else:
            out[name] = sample_lognormal_median_cv(median, cv, n_subjects, rng)

    out["dose_40mg_equiv_mg"] = 40.0 * bw_ratio
    out["mg_per_kg_at_40mg_reference"] = out["dose_40mg_equiv_mg"] / bw
    return pd.DataFrame(out)


def apply_weight_normalized_dose(reference_dose_schedule, bw_kg):
    """
    Convert a 70 kg reference dose schedule to a subject-specific schedule.

    Example:
        40 mg reference dose at BW = 50 kg -> 28.6 mg actual dose
        40 mg reference dose at BW = 90 kg -> 51.4 mg actual dose
    """
    if not USE_WEIGHT_NORMALIZED_DOSING:
        return list(reference_dose_schedule)
    dose_factor = float(bw_kg) / REFERENCE_BW_KG
    return [(t, dose * dose_factor) for t, dose in reference_dose_schedule]


def simulate_population_brain(reference_schedule, n_subjects=200, t_end=168.0, dt=0.5):
    """
    Simulate a population using body-weight-normalized actual doses.

    Parameters
    ----------
    reference_schedule:
        Dose schedule written for a 70 kg reference individual.
    n_subjects:
        Number of virtual subjects.
    t_end, dt:
        Simulation time grid.

    Returns
    -------
    t : ndarray
    Cb_matrix_ngml : ndarray, shape (n_subjects, time)
    pop_df : DataFrame
    actual_total_dose_first_day : ndarray
        Subject-specific total actual dose on day 0-24 h.
    """
    t = np.arange(0, t_end + dt, dt)
    pop_df = sample_guideline_population(n_subjects)
    Cb_matrix = np.zeros((n_subjects, len(t)))

    actual_total_dose_first_day = np.zeros(n_subjects)

    for i in range(n_subjects):
        subj = pop_df.iloc[i]
        subj_schedule = apply_weight_normalized_dose(
            reference_schedule,
            subj["BW_kg"],
        )
        actual_total_dose_first_day[i] = sum(
            dose for time, dose in subj_schedule if 0 <= time < 24
        )
        sim = simulate_subject(subj, subj_schedule, t)
        Cb_matrix[i, :] = sim["Cb"] * 1000.0

    return t, Cb_matrix, pop_df, actual_total_dose_first_day


def compute_min_alpha_and_cmax(t, Cb_matrix_ngml, elasticity_func, ss_start=144.0):
    dCbdt_matrix = np.gradient(Cb_matrix_ngml, t, axis=1)
    ss_mask = t >= ss_start

    n_subjects = Cb_matrix_ngml.shape[0]
    min_alphas = np.zeros(n_subjects)
    Cmaxs = np.zeros(n_subjects)
    troughs = np.zeros(n_subjects)
    max_dCdt = np.zeros(n_subjects)
    AUCs = np.zeros(n_subjects)

    for i in range(n_subjects):
        R_i = reinforcement_rate(dCbdt_matrix[i, :])
        alpha_i = elasticity_func(R_i)
        min_alphas[i] = np.min(alpha_i)
        Cmaxs[i] = np.max(Cb_matrix_ngml[i, ss_mask])
        troughs[i] = np.min(Cb_matrix_ngml[i, ss_mask])
        max_dCdt[i] = np.max(dCbdt_matrix[i, ss_mask])
        AUCs[i] = trapezoid_compat(Cb_matrix_ngml[i, ss_mask], t[ss_mask])

    return {
        "min_alpha": min_alphas,
        "Cmax": Cmaxs,
        "trough": troughs,
        "max_dCdt": max_dCdt,
        "AUC": AUCs,
        "dCbdt_matrix": dCbdt_matrix,
    }


# =============================================================================
# 3. DOSE ESCALATION TO FIND INELASTIC THRESHOLD
# =============================================================================


def find_dose_reaching_threshold(
    elasticity_func,
    alpha_critical,
    dose_range_mg=None,
    n_subjects=200,
    regimen="qd",
):
    """
    Escalate the 70 kg reference dose and calculate the population proportion
    reaching the inelastic regime.

    Important:
        dose_mg / reference_dose_mg = dose for a 70 kg reference individual.
        Actual dose differs by subject: dose_i = dose_mg * BW_i / 70.
    """
    if dose_range_mg is None:
        dose_range_mg = [40, 80, 120, 160, 200, 300, 400, 600, 800, 1200]

    schedule_func = {"qd": schedule_qd, "bid": schedule_bid, "tid": schedule_tid}[regimen]

    results = []
    for reference_dose_mg in dose_range_mg:
        print(f"  Simulating reference dose = {reference_dose_mg} mg {regimen.upper()}...")
        reference_schedule = schedule_func(reference_dose_mg, n_days=7)
        t, Cb_matrix, pop_df, actual_daily_dose = simulate_population_brain(
            reference_schedule,
            n_subjects=n_subjects,
            t_end=168.0,
            dt=0.5,
        )

        metrics = compute_min_alpha_and_cmax(t, Cb_matrix, elasticity_func, ss_start=144.0)
        min_alphas = metrics["min_alpha"]
        Cmaxs = metrics["Cmax"]

        pct_exceeding = float(np.mean(min_alphas < alpha_critical) * 100.0)

        results.append({
            "reference_dose_mg_70kg": reference_dose_mg,
            "median_actual_daily_dose_mg": float(np.median(actual_daily_dose)),
            "p5_actual_daily_dose_mg": float(np.percentile(actual_daily_dose, 5)),
            "p95_actual_daily_dose_mg": float(np.percentile(actual_daily_dose, 95)),
            "median_BW_kg": float(np.median(pop_df["BW_kg"])),
            "pct_exceeding": pct_exceeding,
            "median_Cmax": float(np.median(Cmaxs)),
            "p95_Cmax": float(np.percentile(Cmaxs, 95)),
            "p5_Cmax": float(np.percentile(Cmaxs, 5)),
            "median_minAlpha": float(np.median(min_alphas)),
            "p5_minAlpha": float(np.percentile(min_alphas, 5)),
            "p95_minAlpha": float(np.percentile(min_alphas, 95)),
        })

    return pd.DataFrame(results)


def find_max_safe_dose(escalation_df, target_pct=5.0):
    safe_mask = escalation_df["pct_exceeding"] <= target_pct
    if not safe_mask.any():
        return None
    return float(escalation_df.loc[safe_mask, "reference_dose_mg_70kg"].max())


# =============================================================================
# 4. FREQUENCY OPTIMIZATION
# =============================================================================


def compare_dosing_frequencies(elasticity_func, n_subjects=200):
    """
    Compare QD, BID, and TID at the same 70 kg reference total daily dose.

    Fair comparison:
        40 mg/day reference total
            QD  = 40 mg once for 70 kg
            BID = 20 mg twice for 70 kg
            TID = 13.33 mg three times for 70 kg

    Each subject still receives BW-normalized actual doses.
    """
    regimens = {
        "QD": (1, schedule_qd),
        "BID": (2, schedule_bid),
        "TID": (3, schedule_tid),
    }

    results = []
    for daily_total_ref_mg in [40, 80, 120, 160]:
        for reg_name, (n_per_day, sched_func) in regimens.items():
            reference_dose_per_admin = daily_total_ref_mg / n_per_day
            reference_schedule = sched_func(reference_dose_per_admin, n_days=7)

            print(
                f"  Simulating {reg_name}: {daily_total_ref_mg} mg/day "
                f"reference total ({reference_dose_per_admin:.2f} mg/admin at 70 kg)..."
            )

            t, Cb_matrix, pop_df, actual_daily_dose = simulate_population_brain(
                reference_schedule,
                n_subjects=n_subjects,
                t_end=168.0,
                dt=0.5,
            )

            metrics = compute_min_alpha_and_cmax(t, Cb_matrix, elasticity_func, ss_start=144.0)

            results.append({
                "regimen": reg_name,
                "daily_total_ref_mg_70kg": daily_total_ref_mg,
                "dose_per_admin_ref_mg_70kg": reference_dose_per_admin,
                "median_actual_daily_dose_mg": float(np.median(actual_daily_dose)),
                "p5_actual_daily_dose_mg": float(np.percentile(actual_daily_dose, 5)),
                "p95_actual_daily_dose_mg": float(np.percentile(actual_daily_dose, 95)),
                "median_Cmax": float(np.median(metrics["Cmax"])),
                "median_trough": float(np.median(metrics["trough"])),
                "median_max_dCdt": float(np.median(metrics["max_dCdt"])),
                "p95_max_dCdt": float(np.percentile(metrics["max_dCdt"], 95)),
                "median_minAlpha": float(np.median(metrics["min_alpha"])),
                "p5_minAlpha": float(np.percentile(metrics["min_alpha"], 5)),
                "fluctuation": float(np.median(metrics["Cmax"] - metrics["trough"])),
            })

    return pd.DataFrame(results)


# =============================================================================
# 5. FORMULATION ANALYSIS: IMMEDIATE VS SUSTAINED RELEASE
# =============================================================================


def simulate_formulation_comparison(elasticity_func, reference_dose_mg=40, n_subjects=200, n_days=7):
    """
    Compare formulations by changing absorption rate ka.

    The dose is still a 70 kg reference dose, and each subject receives:
        actual dose_i = reference_dose_mg * BW_i / 70
    """
    ka_scenarios = {
        "Immediate Release (IR)": 1.5,
        "Sustained Release (SR-1)": 0.5,
        "Sustained Release (SR-2)": 0.2,
        "Sustained Release (SR-3)": 0.1,
    }

    results = []
    profiles = {}

    original_ka_median = PARAM_SPECS["ka"][0]

    for label, ka_value in ka_scenarios.items():
        print(f"  Simulating formulation: {label} (ka={ka_value})...")

        pop_df = sample_guideline_population(n_subjects)
        pop_df["ka"] = pop_df["ka"] * (ka_value / original_ka_median)

        reference_schedule = schedule_qd(reference_dose_mg, n_days=n_days)
        t = np.arange(0, n_days * 24 + 0.5, 0.5)

        Cb_matrix = np.zeros((n_subjects, len(t)))
        actual_daily_dose = np.zeros(n_subjects)

        for i in range(n_subjects):
            subj = pop_df.iloc[i]
            subj_schedule = apply_weight_normalized_dose(reference_schedule, subj["BW_kg"])
            actual_daily_dose[i] = sum(dose for time, dose in subj_schedule if 0 <= time < 24)
            sim = simulate_subject(subj, subj_schedule, t)
            Cb_matrix[i, :] = sim["Cb"] * 1000.0

        metrics = compute_min_alpha_and_cmax(
            t,
            Cb_matrix,
            elasticity_func,
            ss_start=(n_days - 1) * 24,
        )

        results.append({
            "formulation": label,
            "ka": ka_value,
            "reference_dose_mg_70kg": reference_dose_mg,
            "median_actual_daily_dose_mg": float(np.median(actual_daily_dose)),
            "median_Cmax": float(np.median(metrics["Cmax"])),
            "median_AUC": float(np.median(metrics["AUC"])),
            "median_max_dCdt": float(np.median(metrics["max_dCdt"])),
            "p95_max_dCdt": float(np.percentile(metrics["max_dCdt"], 95)),
            "median_minAlpha": float(np.median(metrics["min_alpha"])),
            "p5_minAlpha": float(np.percentile(metrics["min_alpha"], 5)),
        })

        typical_idx = int(np.argmin(np.abs(metrics["Cmax"] - np.median(metrics["Cmax"]))))
        profiles[label] = {
            "t": t,
            "Cb": Cb_matrix[typical_idx],
            "dCbdt": metrics["dCbdt_matrix"][typical_idx],
            "ka": ka_value,
        }

    return pd.DataFrame(results), profiles


# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================


def plot_dose_escalation(escalation_df, alpha_critical, target_pct, safe_dose, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = escalation_df["reference_dose_mg_70kg"]

    ax = axes[0]
    ax.plot(x, escalation_df["pct_exceeding"], marker="o", color="#d62728", lw=2)
    ax.axhline(target_pct, color="black", linestyle="--", lw=1, label=f"{target_pct}% safety criterion")
    if safe_dose is not None:
        ax.axvline(safe_dose, color="green", linestyle="--", lw=1.2,
                   label=f"Max safe reference dose = {safe_dose:.0f} mg")
    ax.set_xlabel("Reference dose for 70 kg individual (mg, QD)")
    ax.set_ylabel(f"Population % with $\\alpha$ < {alpha_critical:.4f}")
    ax.set_xscale("log")
    ax.set_title("Dose vs. Population Proportion Reaching Inelastic Regime")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.fill_between(x, escalation_df["p5_Cmax"], escalation_df["p95_Cmax"],
                    alpha=0.25, color="#1f77b4", label="5th-95th %ile")
    ax.plot(x, escalation_df["median_Cmax"], marker="o", color="#1f77b4", lw=2,
            label="Median")
    ax.set_xlabel("Reference dose for 70 kg individual (mg)")
    ax.set_ylabel("Steady-state Brain $C_{max}$ (ng/mL)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Dose-Response: Brain $C_{max}$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    plt.suptitle("NSI-189 Dose Escalation Analysis (BW-normalized dosing)",
                 fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_frequency_comparison(freq_df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    color_map = {"QD": "#d62728", "BID": "#2ca02c", "TID": "#1f77b4"}

    ax = axes[0, 0]
    for reg in ["QD", "BID", "TID"]:
        sub = freq_df[freq_df["regimen"] == reg].sort_values("daily_total_ref_mg_70kg")
        ax.plot(sub["daily_total_ref_mg_70kg"], sub["median_max_dCdt"],
                marker="o", color=color_map[reg], lw=2, label=reg)
    ax.set_xlabel("Daily total reference dose for 70 kg (mg/day)")
    ax.set_ylabel(r"Median Max $dC_b/dt$ (ng/mL/h)")
    ax.set_title("Reinforcement Signal Strength by Frequency")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for reg in ["QD", "BID", "TID"]:
        sub = freq_df[freq_df["regimen"] == reg].sort_values("daily_total_ref_mg_70kg")
        ax.plot(sub["daily_total_ref_mg_70kg"], sub["fluctuation"],
                marker="o", color=color_map[reg], lw=2, label=reg)
    ax.set_xlabel("Daily total reference dose for 70 kg (mg/day)")
    ax.set_ylabel("Cmax - Trough Fluctuation (ng/mL)")
    ax.set_title("Steady-State Fluctuation by Frequency")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for reg in ["QD", "BID", "TID"]:
        sub = freq_df[freq_df["regimen"] == reg].sort_values("daily_total_ref_mg_70kg")
        ax.plot(sub["daily_total_ref_mg_70kg"], sub["median_Cmax"],
                marker="o", color=color_map[reg], lw=2, label=reg)
    ax.set_xlabel("Daily total reference dose for 70 kg (mg/day)")
    ax.set_ylabel("Median Brain $C_{max}$ (ng/mL)")
    ax.set_title("Steady-State Cmax")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for reg in ["QD", "BID", "TID"]:
        sub = freq_df[freq_df["regimen"] == reg].sort_values("daily_total_ref_mg_70kg")
        ax.plot(sub["daily_total_ref_mg_70kg"], sub["median_minAlpha"],
                marker="o", color=color_map[reg], lw=2, label=reg)
    ax.axhline(ALPHA_AMPHETAMINE, color="gray", linestyle="--", lw=1,
               label=f"Amphetamine $\\alpha$={ALPHA_AMPHETAMINE}")
    ax.set_xlabel("Daily total reference dose for 70 kg (mg/day)")
    ax.set_ylabel(r"Median Minimum $\alpha$")
    ax.set_title("Elasticity by Frequency (higher = safer)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.suptitle("Dosing Frequency Optimization: QD vs BID vs TID (BW-normalized)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_formulation_comparison(form_df, profiles, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(profiles)))

    ax = axes[0, 0]
    for (label, prof), color in zip(profiles.items(), colors):
        mask = prof["t"] >= prof["t"][-1] - 48
        ax.plot(prof["t"][mask] - prof["t"][mask][0], prof["Cb"][mask],
                color=color, lw=1.8, label=label)
    ax.set_xlabel("Time (h, last 48h of steady state)")
    ax.set_ylabel("Brain $C_b$ (ng/mL)")
    ax.set_title("Brain Concentration: IR vs Sustained Release")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for (label, prof), color in zip(profiles.items(), colors):
        mask = prof["t"] >= prof["t"][-1] - 48
        ax.plot(prof["t"][mask] - prof["t"][mask][0], prof["dCbdt"][mask],
                color=color, lw=1.8, label=label)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"$dC_b/dt$ (ng/mL/h)")
    ax.set_title(r"$dC_b/dt$ (Reinforcement Signal) by Formulation")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(form_df["ka"], form_df["median_max_dCdt"],
            marker="o", color="#d62728", lw=2, label="Median")
    ax.fill_between(form_df["ka"], form_df["median_max_dCdt"], form_df["p95_max_dCdt"],
                    alpha=0.2, color="#d62728", label="Median to 95th %ile")
    ax.set_xlabel(r"Absorption rate $k_a$ ($h^{-1}$)")
    ax.set_ylabel(r"Max $dC_b/dt$ (ng/mL/h)")
    ax.set_xscale("log")
    ax.set_title(r"Reinforcement Rate vs. Absorption Rate")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(form_df["ka"], form_df["median_minAlpha"],
            marker="o", color="#9467bd", lw=2, label="Median")
    ax.fill_between(form_df["ka"], form_df["p5_minAlpha"], form_df["median_minAlpha"],
                    alpha=0.2, color="#9467bd", label="5th %ile to median")
    ax.axhline(ALPHA_AMPHETAMINE, color="gray", linestyle="--", lw=1,
               label=f"Amphetamine $\\alpha$={ALPHA_AMPHETAMINE}")
    ax.set_xlabel(r"Absorption rate $k_a$ ($h^{-1}$)")
    ax.set_ylabel(r"Minimum $\alpha$")
    ax.set_xscale("log")
    ax.set_title(r"Elasticity Improvement with Slower Absorption")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    plt.suptitle("Formulation Analysis: Sustained Release Reduces Reinforcement Signal",
                 fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_therapeutic_window(escalation_df, mec_brain, safe_dose, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = escalation_df["reference_dose_mg_70kg"]

    ax.fill_between(x, escalation_df["p5_Cmax"], escalation_df["p95_Cmax"],
                    alpha=0.25, color="#1f77b4", label="Population Cmax 5-95% PI")
    ax.plot(x, escalation_df["median_Cmax"], marker="o", color="#1f77b4", lw=2,
            label="Median Cmax")

    ax.axhline(mec_brain, color="green", lw=2,
               label=f"MEC = {mec_brain:.0f} ng/mL")

    threshold_crossing = None
    for i in range(len(escalation_df) - 1):
        left = escalation_df["pct_exceeding"].iloc[i]
        right = escalation_df["pct_exceeding"].iloc[i + 1]
        if left < 5 and right >= 5:
            threshold_crossing = escalation_df["reference_dose_mg_70kg"].iloc[i + 1]
            break

    if threshold_crossing is not None:
        row_idx = escalation_df["reference_dose_mg_70kg"].sub(threshold_crossing).abs().idxmin()
        threshold_cmax = escalation_df.loc[row_idx, "p95_Cmax"]
        ax.axhline(threshold_cmax, color="red", lw=2, linestyle="--",
                   label=f"Inelastic threshold p95 Cmax ~ {threshold_cmax:.0f} ng/mL")

    if safe_dose is not None:
        ax.axvline(safe_dose, color="green", lw=1.5, linestyle=":", alpha=0.7,
                   label=f"Max safe reference dose = {safe_dose:.0f} mg")

    ax.axvline(40, color="black", lw=1.5, linestyle=":", alpha=0.5,
               label="Clinical reference = 40 mg at 70 kg")

    ax.set_xlabel("Reference dose for 70 kg individual (mg, QD)")
    ax.set_ylabel("Brain $C_{max}$ (ng/mL)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Therapeutic Window: MEC to Inelastic Threshold (BW-normalized)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()

    return threshold_crossing


# =============================================================================
# 7. FINAL GUIDELINE TABLE
# =============================================================================


def generate_dosing_guideline_table(safe_dose_qd, best_frequency, best_formulation,
                                    clinical_reference=40, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    ax.text(0.5, 0.98, "NSI-189 Dosing Guideline Summary",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            ha="center", va="top")
    ax.text(0.5, 0.93,
            "Body-weight-normalized dosing + mPBPK + elasticity analysis",
            transform=ax.transAxes, fontsize=10, style="italic",
            ha="center", va="top")

    rows = [
        ["Therapeutic Window", f"MEC: {MEC_BRAIN_NGML:.0f} ng/mL brain", "Lower bound for cognitive enhancement"],
        ["", f"Upper bound: alpha > {ALPHA_AMPHETAMINE}", "Amphetamine-level conservative threshold"],
        ["", "", ""],
        ["DOSE", f"Recommended: <= {safe_dose_qd:.0f} mg reference dose", "Dose is for 70 kg individual"],
        ["", "Actual individual dose = reference dose x BW/70", "Equivalent mg/kg normalization"],
        ["", f"Clinical reference: {clinical_reference} mg at 70 kg", "40 mg corresponds to 0.571 mg/kg"],
        ["", "", ""],
        ["FREQUENCY", f"Recommended: {best_frequency}", "Lowest dC/dt at the same daily reference dose"],
        ["", "Split dosing can reduce peak dC/dt", "But practicality should be considered"],
        ["", "", ""],
        ["TIMING", "Morning, about 3-4 h before cognitive task", "Align with simulated brain Tmax"],
        ["", "Taking with food may slow absorption", "May reduce reinforcement signal"],
        ["", "", ""],
        ["FORMULATION", f"Recommended: {best_formulation}", "Slower absorption reduces dC/dt without large AUC loss"],
        ["", f"CL mode: CL_app BW^0.75; brain-transfer CL = {BRAIN_TRANSFER_CL_SCALING_MODE}", "Modeling assumption to report clearly"],
    ]

    headers = ["Component", "Recommendation", "Rationale"]
    col_widths = [0.20, 0.43, 0.37]
    col_lefts = [0.02, 0.22, 0.65]
    row_height = 0.053
    top = 0.86

    for header, left, width in zip(headers, col_lefts, col_widths):
        ax.add_patch(Rectangle((left, top), width, row_height,
                               facecolor="#2c3e50", edgecolor="white",
                               transform=ax.transAxes))
        ax.text(left + width / 2, top + row_height / 2, header,
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                color="white", ha="center", va="center")

    for row_idx, row in enumerate(rows):
        y = top - (row_idx + 1) * row_height
        for col_idx, (cell, left, width) in enumerate(zip(row, col_lefts, col_widths)):
            color = "#ecf0f1" if row_idx % 2 == 0 else "white"
            text_color = "black"
            weight = "normal"
            if cell.strip() and col_idx == 0 and row[0] in ["DOSE", "FREQUENCY", "TIMING", "FORMULATION", "Therapeutic Window"]:
                color = "#3498db"
                text_color = "white"
                weight = "bold"

            ax.add_patch(Rectangle((left, y), width, row_height,
                                   facecolor=color, edgecolor="#bdc3c7",
                                   transform=ax.transAxes))
            ax.text(left + 0.01, y + row_height / 2, cell,
                    transform=ax.transAxes, fontsize=8.5,
                    color=text_color, weight=weight,
                    ha="left", va="center", wrap=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# 8. MAIN WORKFLOW
# =============================================================================


def main():
    print("=" * 72)
    print("NSI-189 Dosing Guideline Establishment - BW-Normalized Version")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n[Settings]")
    print(f"  Reference BW: {REFERENCE_BW_KG:.1f} kg")
    print("  Dose scaling: actual dose = reference dose x BW/70")
    print(f"  Vb, Vc scaling: BW^{VOLUME_BW_EXPONENT:.2f}")
    print(f"  CL_app scaling: BW^{SYSTEMIC_CL_BW_EXPONENT:.2f}")
    print(f"  CL_in/CL_out scaling mode: {BRAIN_TRANSFER_CL_SCALING_MODE}")
    print(f"  Residual CV multiplier for size-scaled parameters: {RESIDUAL_CV_MULTIPLIER:.2f}")

    # ---- Step 0: Calibrate elasticity function ----
    print("\n[Step 0] Calibrating elasticity function...")
    fits = {}
    for form_name in ["exponential", "power", "hill"]:
        p, r2, aic = calibrate_elasticity_function(form_name)
        fits[form_name] = (p, r2, aic)
        print(f"  {form_name:<12} R2={r2:.4f}, AIC={aic:.3f}")

    best_form, _ = select_best_form(fits)
    primary_params, _, _ = fits[best_form]
    elasticity_func = get_elasticity_function(best_form, primary_params)
    print(f"  Best form: {best_form.upper()}")

    # ---- Step 1: Therapeutic window ----
    print("\n[Step 1] Therapeutic window boundaries")
    print(f"  Lower bound MEC: {MEC_BRAIN_NGML:.1f} ng/mL brain")
    print(f"  Upper bound criterion: alpha > {ALPHA_CRITICAL:.4f}")

    # ---- Step 2: Dose escalation ----
    print("\n[Step 2] Dose escalation with BW-normalized dosing")
    print("  This may take a few minutes...")

    dose_range = [40, 80, 120, 200, 400, 800, 1600, 3200]
    escalation_df = find_dose_reaching_threshold(
        elasticity_func,
        ALPHA_CRITICAL,
        dose_range_mg=dose_range,
        n_subjects=200,
        regimen="qd",
    )

    print("\n  Dose escalation results:")
    print(escalation_df[[
        "reference_dose_mg_70kg",
        "median_actual_daily_dose_mg",
        "pct_exceeding",
        "median_Cmax",
        "p95_Cmax",
        "median_minAlpha",
        "p5_minAlpha",
    ]].to_string(index=False))

    target_pct = 5.0
    safe_dose_qd = find_max_safe_dose(escalation_df, target_pct=target_pct)
    if safe_dose_qd is None:
        safe_dose_qd = float(escalation_df["reference_dose_mg_70kg"].max())
        print(f"\n  No tested dose exceeded the {target_pct:.1f}% criterion.")
        print(f"  Safe reference dose is at least {safe_dose_qd:.0f} mg at 70 kg.")
    else:
        print(f"\n  Max safe reference dose = {safe_dose_qd:.0f} mg at 70 kg")

    plot_dose_escalation(
        escalation_df,
        ALPHA_CRITICAL,
        target_pct,
        safe_dose_qd,
        save_path=out_path("fig9_bw_dose_escalation.png"),
    )

    # ---- Step 3: Frequency optimization ----
    print("\n[Step 3] Frequency comparison with BW-normalized dosing")
    freq_df = compare_dosing_frequencies(elasticity_func, n_subjects=200)

    print("\n  Frequency comparison results:")
    print(freq_df[[
        "regimen",
        "daily_total_ref_mg_70kg",
        "dose_per_admin_ref_mg_70kg",
        "median_actual_daily_dose_mg",
        "median_Cmax",
        "median_max_dCdt",
        "median_minAlpha",
    ]].to_string(index=False))

    best_freq_per_total = freq_df.loc[
        freq_df.groupby("daily_total_ref_mg_70kg")["median_max_dCdt"].idxmin()
    ]
    print("\n  Best frequency per daily reference dose:")
    print(best_freq_per_total[[
        "daily_total_ref_mg_70kg",
        "regimen",
        "median_max_dCdt",
    ]].to_string(index=False))

    ref_subset = freq_df[freq_df["daily_total_ref_mg_70kg"] == 40]
    best_frequency = ref_subset.loc[ref_subset["median_max_dCdt"].idxmin(), "regimen"]

    plot_frequency_comparison(freq_df, save_path=out_path("fig10_bw_frequency.png"))

    # ---- Step 4: Formulation analysis ----
    print("\n[Step 4] Formulation comparison with BW-normalized dosing")
    form_df, formulation_profiles = simulate_formulation_comparison(
        elasticity_func,
        reference_dose_mg=40,
        n_subjects=200,
        n_days=7,
    )

    print("\n  Formulation comparison results:")
    print(form_df.to_string(index=False))

    best_form_idx = form_df["median_max_dCdt"].idxmin()
    best_formulation = str(form_df.loc[best_form_idx, "formulation"])

    plot_formulation_comparison(
        form_df,
        formulation_profiles,
        save_path=out_path("fig11_bw_formulation.png"),
    )

    # ---- Step 5: Therapeutic window ----
    print("\n[Step 5] Therapeutic window visualization")
    threshold_crossing = plot_therapeutic_window(
        escalation_df,
        MEC_BRAIN_NGML,
        safe_dose_qd,
        save_path=out_path("fig12_bw_therapeutic_window.png"),
    )

    # ---- Step 6: Summary table ----
    print("\n[Step 6] Guideline summary figure")
    generate_dosing_guideline_table(
        safe_dose_qd=safe_dose_qd,
        best_frequency=best_frequency,
        best_formulation=best_formulation,
        clinical_reference=40,
        save_path=out_path("fig13_bw_dosing_guideline.png"),
    )

    # ---- Step 7: Save CSVs ----
    print("\n[Step 7] Saving CSV files")
    escalation_df.to_csv(out_path("dose_escalation_bw_normalized.csv"), index=False)
    freq_df.to_csv(out_path("frequency_comparison_bw_normalized.csv"), index=False)
    form_df.to_csv(out_path("formulation_comparison_bw_normalized.csv"), index=False)

    settings_df = pd.DataFrame([{
        "reference_BW_kg": REFERENCE_BW_KG,
        "weight_median_kg": WEIGHT_MEDIAN_KG,
        "weight_cv": WEIGHT_CV,
        "residual_cv_multiplier": RESIDUAL_CV_MULTIPLIER,
        "volume_BW_exponent": VOLUME_BW_EXPONENT,
        "systemic_CL_BW_exponent": SYSTEMIC_CL_BW_EXPONENT,
        "brain_transfer_CL_scaling_mode": BRAIN_TRANSFER_CL_SCALING_MODE,
        "brain_transfer_CL_BW_exponent_if_enabled": BRAIN_TRANSFER_CL_BW_EXPONENT,
        "alpha_critical": ALPHA_CRITICAL,
        "MEC_brain_ngml": MEC_BRAIN_NGML,
        "best_elasticity_form": best_form,
    }])
    settings_df.to_csv(out_path("guideline_population_settings_bw_normalized.csv"), index=False)

    # ---- Final summary ----
    print("\n" + "=" * 72)
    print("DOSING GUIDELINE SUMMARY - BW NORMALIZED")
    print("=" * 72)
    print("\nDose interpretation:")
    print("  Reported dose = 70 kg reference dose")
    print("  Actual individual dose = reference dose x BW/70")
    print(f"\nTherapeutic window:")
    print(f"  Lower bound MEC:   {MEC_BRAIN_NGML:.1f} ng/mL brain")
    print(f"  Upper bound alpha: alpha > {ALPHA_CRITICAL:.4f}")
    print("\nFour recommended parameters:")
    print(f"  1. DOSE:        <= {safe_dose_qd:.0f} mg reference dose at 70 kg")
    print("                  Actual dose should be scaled by BW/70")
    print(f"  2. FREQUENCY:   {best_frequency}")
    print("                  Chosen by lowest dC/dt at clinical reference dose")
    print("  3. TIMING:      Morning, about 3-4 h before cognitive task")
    print(f"  4. FORMULATION: {best_formulation}")
    print("                  Slower absorption lowers reinforcement signal")
    print("\nModeling note:")
    print(f"  CL_app scaled by BW^0.75; CL_in/CL_out mode = {BRAIN_TRANSFER_CL_SCALING_MODE}")

    if threshold_crossing is not None:
        print("\nKey safety margin:")
        print(f"  Clinical reference 40 mg vs threshold ~{threshold_crossing:.0f} mg")
        print(f"  = {threshold_crossing / 40:.1f}x reference-dose safety factor")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
