"""
NSI-189 mPBPK Model + Elasticity-Based Addiction Quantification (Integrated)
==============================================================================

This is the integrated implementation of Methodology Sections 1 and 2:

  Section 1: Monte Carlo-Applied mPBPK Modeling
    - 3-compartment mPBPK (Gut, Plasma/Central, Brain)
    - Validated against Phase 1b clinical data (Fava et al., 2016)
    - Monte Carlo population simulation with body-weight-normalized dosing
      and lognormal residual parameter variability

  Section 2: Elasticity-Based Addiction Quantification
    - Reinforcement rate from brain dC/dt
    - Three elasticity functional forms (exponential, power, Hill)
    - Calibration against reference cognitive enhancers
      (amphetamine, nicotine, cocaine) using Hursh-Silberberg framework
    - Inelastic threshold derivation from brain concentration

The code produces a complete dosing-guideline-ready dataset in one run.

Output files (saved in same folder as this script):
  PK Modeling:
    - fig0_clinical_validation.png
    - fig1_regimens_comparison.png
    - fig2_monte_carlo_qd.png
    - fig3_dCbdt.png
    - fig4_pk_metrics.png
  Elasticity Analysis:
    - fig5_elasticity_calibration.png
    - fig6_nsi189_elasticity.png
    - fig7_elasticity_form_comparison.png
    - fig8_elasticity_population.png
  Data:
    - pk_metrics_brain.csv, pk_metrics_plasma.csv
    - population_parameters.csv
    - elasticity_calibration.csv
    - nsi189_elasticity_profile.csv
    - population_elasticity.csv
    - functional_form_comparison.csv

References:
  - Fava et al. (2016). Mol Psychiatry 21:1372-1380.
  - Davies & Morris (1993). Pharm Res 10:1093-1095.
  - Hursh & Silberberg (2008). Psychol Rev 115:186-198.
  - Christensen et al. (2008). Psychopharmacology 198:221-229.
  - Bentzley et al. (2014). PNAS 111:11822-11827.
  - Diergaarde et al. (2012). Addict Biol 17:576-587.
  - Volkow et al. (2003). Synapse 47:296-306.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =============================================================================
# 0. ENVIRONMENT SETUP
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


RNG = np.random.default_rng(seed=42)

# =============================================================================
# 0B. BODY-WEIGHT NORMALIZATION SETTINGS
# =============================================================================
# Why this is here:
#   The previous Monte Carlo sampled Vc, Vb, CL_in, CL_out, and CL_app independently.
#   That can create unrealistically extreme subjects, e.g., very small volume + very
#   slow clearance, so the brain concentration distribution becomes too wide.
#
# What this version does:
#   1) Sample body weight for each subject.
#   2) Scale volumes ~ BW^1.0 and clearances ~ BW^0.75.
#   3) Give dose by mg/kg: 40 mg is treated as the dose for a 70 kg reference person.
#      So each subject receives 40 * BW/70 mg.
#   4) Use only residual CV after body-size correction for size-dependent parameters.
#
# Important:
#   Do NOT simply divide concentration by body weight. Concentration already has a
#   volume denominator. The biologically cleaner way is to normalize dose and
#   physiological parameters by body size.
REFERENCE_BW_KG = 70.0
WEIGHT_MEDIAN_KG = 70.0
WEIGHT_CV = 0.18
USE_WEIGHT_NORMALIZED_DOSING = True
USE_ALLOMETRIC_SCALING = True

# Original CVs often include body-size variability. After BW correction, only a
# smaller unexplained/residual variability should remain. Increase this toward 1.0
# if your teacher wants a more conservative, wider population distribution.
RESIDUAL_CV_MULTIPLIER = 0.45

# Allometric exponents commonly used in PBPK-style scaling.
# Volume generally scales almost linearly with body weight; clearance often scales
# around BW^0.75.
SIZE_SCALING = {
    'Vb': 1.00,
    'Vc': 1.00,
    'CL_in': 0.75,
    'CL_out': 0.75,
    'CL_app': 0.75,
}


# =============================================================================
# ============================================================================
#  PART A: mPBPK MODELING (Methodology Section 1)
# ============================================================================
# =============================================================================

# -----------------------------------------------------------------------------
# A1. PARAMETER SPECIFICATIONS (calibrated to Phase 1b)
# -----------------------------------------------------------------------------
PARAM_SPECS = {
    'Vb':          (1.4,     0.12, 'L',    'physiological'),
    'Vc':          (960.0,   0.40, 'L',    'derived V/F'),
    'ka':          (1.5,     0.35, 'h^-1', 'drug'),
    'CL_in':       (5.0,     0.35, 'L/h',  'drug (BBB influx)'),
    'CL_out':      (2.0,     0.35, 'L/h',  'drug (BBB efflux)'),
    'CL_app':      (35.0,    0.35, 'L/h',  'derived CL/F'),
}
_Kp_implied = PARAM_SPECS['CL_in'][0] / PARAM_SPECS['CL_out'][0]


# -----------------------------------------------------------------------------
# A2. LOGNORMAL SAMPLING
# -----------------------------------------------------------------------------
def lognormal_mu_sigma(median, cv):
    mu = np.log(median)
    sigma = np.sqrt(np.log(1.0 + cv**2))
    return mu, sigma


def sample_body_weight(n_subjects, rng=None):
    """Sample adult body weight as a lognormal distribution."""
    rng = rng or RNG
    mu, sigma = lognormal_mu_sigma(WEIGHT_MEDIAN_KG, WEIGHT_CV)
    return rng.lognormal(mean=mu, sigma=sigma, size=n_subjects)


def sample_residual_multiplier(cv, size_scaled=False, rng=None, n_subjects=1):
    """
    Residual multiplicative variability around the allometrically scaled value.

    If a parameter is size-scaled, its original CV is partly explained by body
    weight, so we shrink the remaining CV by RESIDUAL_CV_MULTIPLIER.
    """
    rng = rng or RNG
    residual_cv = cv * RESIDUAL_CV_MULTIPLIER if size_scaled else cv
    residual_cv = max(residual_cv, 1e-6)
    mu, sigma = lognormal_mu_sigma(1.0, residual_cv)
    return rng.lognormal(mean=mu, sigma=sigma, size=n_subjects)


def sample_population(n_subjects, rng=None):
    rng = rng or RNG
    out = {}

    # Body weight is now an explicit covariate.
    bw = sample_body_weight(n_subjects, rng=rng)
    out['BW_kg'] = bw

    for name, (median, cv, _, _) in PARAM_SPECS.items():
        if USE_ALLOMETRIC_SCALING and name in SIZE_SCALING:
            exponent = SIZE_SCALING[name]
            size_factor = (bw / REFERENCE_BW_KG) ** exponent
            residual = sample_residual_multiplier(
                cv, size_scaled=True, rng=rng, n_subjects=n_subjects
            )
            out[name] = median * size_factor * residual
        else:
            mu, sigma = lognormal_mu_sigma(median, cv)
            out[name] = rng.lognormal(mean=mu, sigma=sigma, size=n_subjects)

    # This is the actual 40 mg-equivalent dose each subject receives under mg/kg
    # normalization. It is saved for transparency.
    if USE_WEIGHT_NORMALIZED_DOSING:
        out['dose_40mg_equiv_mg'] = 40.0 * bw / REFERENCE_BW_KG
    else:
        out['dose_40mg_equiv_mg'] = np.full(n_subjects, 40.0)

    return pd.DataFrame(out)


def typical_subject():
    params = {name: spec[0] for name, spec in PARAM_SPECS.items()}
    params['BW_kg'] = REFERENCE_BW_KG
    params['dose_40mg_equiv_mg'] = 40.0
    return pd.Series(params)


# -----------------------------------------------------------------------------
# A3. ODE SYSTEM (mass-balance form)
# -----------------------------------------------------------------------------
def mpbpk_ode(y, t, params):
    A_gut, A_p, A_b = y
    Cp = A_p / params['Vc']
    Cb = A_b / params['Vb']

    dA_gut = -params['ka'] * A_gut
    dA_p   = (params['ka'] * A_gut
              - params['CL_in']  * Cp
              + params['CL_out'] * Cb
              - params['CL_app'] * Cp)
    dA_b   = params['CL_in'] * Cp - params['CL_out'] * Cb
    return [dA_gut, dA_p, dA_b]


def simulate_subject(params, dose_schedule, t_eval):
    dose_schedule = sorted(dose_schedule, key=lambda x: x[0])
    state = np.array([0.0, 0.0, 0.0])
    t_history = [t_eval[0]]
    state_history = [state.copy()]

    event_times = sorted(set([d[0] for d in dose_schedule] + [t_eval[-1]]))
    dose_lookup = {t: d for t, d in dose_schedule}

    current_t = t_eval[0]

    for next_t in event_times:
        if next_t <= current_t:
            if next_t in dose_lookup:
                state[0] += dose_lookup[next_t]
                state_history[-1] = state.copy()
            current_t = next_t
            continue

        seg_times = np.linspace(current_t, next_t,
                                max(2, int((next_t - current_t) * 20)))
        sol = odeint(mpbpk_ode, state, seg_times,
                     args=(params,), rtol=1e-9, atol=1e-11, mxstep=10000)
        t_history.extend(seg_times[1:].tolist())
        state_history.extend(sol[1:].tolist())

        state = sol[-1].copy()
        current_t = next_t

        if next_t in dose_lookup:
            state[0] += dose_lookup[next_t]
            t_history.append(next_t)
            state_history.append(state.copy())

    t_arr = np.array(t_history)
    state_arr = np.array(state_history)
    A_gut_e = np.interp(t_eval, t_arr, state_arr[:, 0])
    A_p_e   = np.interp(t_eval, t_arr, state_arr[:, 1])
    A_b_e   = np.interp(t_eval, t_arr, state_arr[:, 2])

    Cp_e = A_p_e / params['Vc']
    Cb_e = A_b_e / params['Vb']
    return {'t': t_eval, 'A_gut': A_gut_e, 'Cp': Cp_e, 'Cb': Cb_e}


# -----------------------------------------------------------------------------
# A4. DOSING SCHEDULES
# -----------------------------------------------------------------------------
def schedule_single_dose(dose_mg, t0=0.0):
    return [(t0, dose_mg)]


def schedule_qd(dose_mg, n_days=7, t0=0.0):
    return [(t0 + 24.0 * i, dose_mg) for i in range(n_days)]


def schedule_bid(dose_mg, n_days=7, t0=0.0):
    return [(t0 + 12.0 * i, dose_mg) for i in range(n_days * 2)]


def schedule_tid(dose_mg, n_days=7, t0=0.0):
    return [(t0 + 8.0 * i, dose_mg) for i in range(n_days * 3)]


def apply_weight_normalized_dose(dose_schedule, bw_kg):
    """
    Convert a reference 70 kg dose schedule into an individualized mg/kg schedule.

    Example: 40 mg QD at 70 kg = 0.571 mg/kg.
      - 50 kg subject receives 28.6 mg
      - 90 kg subject receives 51.4 mg
    """
    if not USE_WEIGHT_NORMALIZED_DOSING:
        return dose_schedule
    dose_factor = bw_kg / REFERENCE_BW_KG
    return [(t, dose * dose_factor) for t, dose in dose_schedule]


# -----------------------------------------------------------------------------
# A5. PK METRICS
# -----------------------------------------------------------------------------
def trapezoid_compat(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def extract_pk_metrics(t, conc, dose_window=None):
    if dose_window is not None:
        mask = (t >= dose_window[0]) & (t <= dose_window[1])
        t_w = t[mask]; c_w = conc[mask]
    else:
        t_w = t; c_w = conc

    cmax = float(np.max(c_w))
    tmax = float(t_w[np.argmax(c_w)])
    auc = float(trapezoid_compat(c_w, t_w))
    dcdt = np.gradient(c_w, t_w)
    max_dcdt = float(np.max(dcdt))
    return {'Cmax': cmax, 'Tmax': tmax, 'AUC': auc, 'max_dCdt': max_dcdt}


# -----------------------------------------------------------------------------
# A6. CLINICAL VALIDATION
# -----------------------------------------------------------------------------
def validate_against_phase1b():
    params = typical_subject()
    sched = schedule_qd(40, n_days=28)
    t_eval = np.linspace(0, 24 * 30, 30 * 48 + 1)
    res = simulate_subject(params, sched, t_eval)

    d28 = (res['t'] >= 27 * 24) & (res['t'] <= 28 * 24)
    t_d28 = res['t'][d28] - 27 * 24
    Cp_d28 = res['Cp'][d28]
    auc_d28 = trapezoid_compat(Cp_d28, t_d28)
    cmax_d28 = np.max(Cp_d28)
    tmax_d28 = t_d28[np.argmax(Cp_d28)]

    after_peak = (res['t'] >= 27 * 24 + 8) & (res['t'] <= 28 * 24)
    t_term = res['t'][after_peak]
    C_term = res['Cp'][after_peak]
    valid = C_term > 0
    slope, _ = np.polyfit(t_term[valid], np.log(C_term[valid]), 1)
    t_half = -np.log(2) / slope

    print("\n  Clinical Validation (Phase 1b, 40 mg QD, Day 28):")
    print("  " + "-" * 60)
    print(f"  {'Metric':<25} {'Simulated':>15} {'Reported':>20}")
    print(f"  {'AUC(0-24) [mg.h/L]':<25} {auc_d28:>15.3f} {'1.144 +- 0.276':>20}")
    print(f"  {'Cmax [mg/L]':<25} {cmax_d28:>15.4f} {'~0.10-0.15':>20}")
    print(f"  {'Tmax [h]':<25} {tmax_d28:>15.2f} {'1-2':>20}")
    print(f"  {'t1/2 [h]':<25} {t_half:>15.2f} {'17.4-20.5':>20}")
    return {'AUC_d28': auc_d28, 'Cmax_d28': cmax_d28,
            'Tmax_d28': tmax_d28, 't_half': t_half, 'result': res}


# -----------------------------------------------------------------------------
# A7. MONTE CARLO
# -----------------------------------------------------------------------------
def run_monte_carlo(dose_schedule, n_subjects=1000, t_end=168.0, dt=0.5):
    pop_df = sample_population(n_subjects)
    t_eval = np.arange(0.0, t_end + dt, dt)
    Cp_matrix = np.zeros((n_subjects, len(t_eval)))
    Cb_matrix = np.zeros((n_subjects, len(t_eval)))

    for i in range(n_subjects):
        subj_params = pop_df.iloc[i]
        subj_schedule = apply_weight_normalized_dose(
            dose_schedule, subj_params.get('BW_kg', REFERENCE_BW_KG)
        )
        sim = simulate_subject(subj_params, subj_schedule, t_eval)
        Cp_matrix[i, :] = sim['Cp']
        Cb_matrix[i, :] = sim['Cb']
        if (i + 1) % 200 == 0:
            print(f"  ... completed {i + 1}/{n_subjects} subjects")
    return t_eval, Cp_matrix, Cb_matrix, pop_df


def summarize_percentiles(matrix, percentiles=(2.5, 25, 50, 75, 97.5)):
    return {p: np.percentile(matrix, p, axis=0) for p in percentiles}


# -----------------------------------------------------------------------------
# A8. PK PLOTTING
# -----------------------------------------------------------------------------
def plot_validation(val_res, save_path=None):
    res = val_res['result']
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    ax.plot(res['t'] / 24, res['Cp'] * 1000, color='#1f77b4', lw=1.2,
            label='Plasma $C_p$')
    ax.plot(res['t'] / 24, res['Cb'] * 1000, color='#d62728', lw=1.2,
            label='Brain $C_b$')
    ax.set_title('NSI-189 40 mg QD - 28-day Profile (Typical Individual)')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Concentration (ng/mL)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 28)

    ax = axes[1]
    d28 = (res['t'] >= 27 * 24) & (res['t'] <= 28 * 24)
    ax.plot(res['t'][d28] - 27 * 24, res['Cp'][d28] * 1000,
            color='#1f77b4', lw=2, label='Plasma $C_p$')
    ax.plot(res['t'][d28] - 27 * 24, res['Cb'][d28] * 1000,
            color='#d62728', lw=2, label='Brain $C_b$')
    ax.set_title('Day 28 (Steady State) - Validation Window')
    ax.set_xlabel('Time after Day 28 dose (h)')
    ax.set_ylabel('Concentration (ng/mL)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.suptitle('Clinical Validation against Phase 1b (Fava et al., 2016)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_regimens(typ_results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (name, res) in zip(axes.flatten(), typ_results.items()):
        ax.plot(res['t'] / 24, res['Cp'] * 1000, label='Plasma $C_p$',
                color='#1f77b4', lw=1.8)
        ax.plot(res['t'] / 24, res['Cb'] * 1000, label='Brain $C_b$',
                color='#d62728', lw=1.8)
        ax.set_title(f'NSI-189 {name} - Typical Subject (7 days)')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration (ng/mL)')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 7)
    plt.suptitle('NSI-189 mPBPK Simulation - Dosing Regimens Comparison',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_monte_carlo(t_eval, Cp_matrix, Cb_matrix, title_suffix='', save_path=None):
    Cp_pct = summarize_percentiles(Cp_matrix * 1000)
    Cb_pct = summarize_percentiles(Cb_matrix * 1000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.fill_between(t_eval, Cp_pct[2.5], Cp_pct[97.5],
                    color='#1f77b4', alpha=0.18, label='95% PI')
    ax.fill_between(t_eval, Cp_pct[25], Cp_pct[75],
                    color='#1f77b4', alpha=0.32, label='50% PI (IQR)')
    ax.plot(t_eval, Cp_pct[50], color='#1f77b4', lw=2, label='Median')
    ax.set_title(f'Plasma Concentration {title_suffix}')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$C_p$ (ng/mL)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(t_eval, Cb_pct[2.5], Cb_pct[97.5],
                    color='#d62728', alpha=0.18, label='95% PI')
    ax.fill_between(t_eval, Cb_pct[25], Cb_pct[75],
                    color='#d62728', alpha=0.32, label='50% PI (IQR)')
    ax.plot(t_eval, Cb_pct[50], color='#d62728', lw=2, label='Median')
    ax.set_title(f'Brain Concentration {title_suffix}')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$C_b$ (ng/mL)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.suptitle(f'NSI-189 mPBPK - Monte Carlo (n=1000) {title_suffix}',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_dcdt(t_eval, Cb_matrix, save_path=None):
    Cb_ngml = Cb_matrix * 1000
    dCbdt = np.gradient(Cb_ngml, t_eval, axis=1)
    pct = summarize_percentiles(dCbdt)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t_eval, pct[2.5], pct[97.5], color='#2ca02c', alpha=0.18,
                    label='95% PI')
    ax.fill_between(t_eval, pct[25], pct[75], color='#2ca02c', alpha=0.32,
                    label='50% PI')
    ax.plot(t_eval, pct[50], color='#2ca02c', lw=2, label='Median')
    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$dC_b/dt$ (ng/mL/h)')
    ax.set_title('Brain Concentration Rate of Change - Input for Elasticity Analysis',
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_pk_distributions(Cp_matrix, Cb_matrix, t_eval, dose_window, save_path=None):
    n = Cb_matrix.shape[0]
    m_brain = pd.DataFrame([
        extract_pk_metrics(t_eval, Cb_matrix[i, :] * 1000, dose_window=dose_window)
        for i in range(n)
    ])
    m_plasma = pd.DataFrame([
        extract_pk_metrics(t_eval, Cp_matrix[i, :] * 1000, dose_window=dose_window)
        for i in range(n)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ('Cmax', m_brain, axes[0, 0], '#d62728', 'Brain $C_{max}$ (ng/mL)'),
        ('Tmax', m_brain, axes[0, 1], '#d62728', 'Brain $T_{max}$ from t=0 (h)'),
        ('max_dCdt', m_brain, axes[1, 0], '#2ca02c', r'Brain max $dC/dt$ (ng/mL/h)'),
        ('AUC', m_brain, axes[1, 1], '#9467bd', 'Brain AUC last 24h (ng.h/mL)'),
    ]
    for col, df, ax, color, label in panels:
        ax.hist(df[col], bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(df[col].median(), color='black', linestyle='--', lw=1.5,
                   label=f"Median = {df[col].median():.4g}")
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
    plt.suptitle('Population PK Metrics from Monte Carlo (Brain, last 24h)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return m_brain, m_plasma


# =============================================================================
# ============================================================================
#  PART B: ELASTICITY-BASED ADDICTION QUANTIFICATION (Methodology Section 2)
# ============================================================================
# =============================================================================

# -----------------------------------------------------------------------------
# B1. REFERENCE COGNITIVE ENHANCER DATA (Hursh-Silberberg framework)
# -----------------------------------------------------------------------------
# Sources:
#   - Hursh & Silberberg (2008) Psychol Rev 115:186-198 - original exponential model
#   - Christensen et al. (2008) Psychopharmacology 198:221-229 - cocaine
#   - Bentzley et al. (2014) PNAS 111:11822-11827 - cocaine
#   - Diergaarde et al. (2012) Addict Biol 17:576-587 - nicotine
#   - de Wit et al. (1991) Psychopharmacology - amphetamine PK
#   - All alpha values normalized to k=3 (standard Hursh-Silberberg convention)
#
# Brain peak dC/dt estimated from PK literature for the standard route:
#   - Amphetamine: oral, peak brain ~ 200 ng/mL/h
#   - Nicotine:    inhaled, peak brain ~ 1500 ng/mL/h
#   - Cocaine:     IV, peak brain ~ 3000 ng/mL/h
#
# Lower alpha = more inelastic demand = MORE addictive

REFERENCE_DRUGS = pd.DataFrame({
    'drug':              ['amphetamine', 'nicotine',  'cocaine'],
    'route':             ['oral',         'inhaled',    'IV'],
    'brain_peak_dCdt':   [200.0,          1500.0,       3000.0],   # ng/mL/h
    'alpha_normalized':  [0.012,          0.0035,       0.0014],   # k=3 normalized
    'reference':         ['de Wit 1991, Hursh 2008',
                          'Diergaarde 2012',
                          'Bentzley 2014, Christensen 2008']
})


# -----------------------------------------------------------------------------
# B2. REINFORCEMENT RATE
# -----------------------------------------------------------------------------
def reinforcement_rate(dCdt_brain):
    """
    R = max(dC/dt, 0). Only the influx phase generates reinforcement signal.
    (Volkow et al., 2003: rate hypothesis of reinforcement)
    """
    return np.maximum(dCdt_brain, 0.0)


# -----------------------------------------------------------------------------
# B3. ELASTICITY FUNCTIONAL FORMS
# -----------------------------------------------------------------------------
# All forms satisfy Methodology Section 2 constraints:
#   (i)   elasticity decreases as R increases
#   (ii)  elasticity > 0 always
#   (iii) elasticity has upper and lower bounds

def elasticity_exponential(R, alpha_0, beta, alpha_min=1e-5):
    """Primary form: alpha = alpha_min + (alpha_0 - alpha_min) * exp(-beta * R)"""
    return alpha_min + (alpha_0 - alpha_min) * np.exp(-beta * R)


def elasticity_power(R, alpha_0, gamma, alpha_min=1e-5, R_ref=100.0):
    """Power decay: alpha = alpha_min + (alpha_0 - alpha_min) * (R_ref/(R+R_ref))^gamma"""
    return alpha_min + (alpha_0 - alpha_min) * (R_ref / (R + R_ref)) ** gamma


def elasticity_hill(R, alpha_0, K, n, alpha_min=1e-5):
    """Hill: alpha = alpha_min + (alpha_0 - alpha_min) * K^n / (R^n + K^n)"""
    return alpha_min + (alpha_0 - alpha_min) * (K ** n) / (R ** n + K ** n)


ELASTICITY_FORMS = {
    'exponential': {
        'func': elasticity_exponential,
        'p0': [0.02, 0.001],
        'bounds': ([1e-5, 1e-6], [1.0, 1.0]),
        'param_names': ['alpha_0', 'beta'],
        'n_params': 2,
    },
    'power': {
        'func': elasticity_power,
        'p0': [0.02, 1.5],
        'bounds': ([1e-5, 0.1], [1.0, 10.0]),
        'param_names': ['alpha_0', 'gamma'],
        'n_params': 2,
    },
    'hill': {
        'func': elasticity_hill,
        'p0': [0.02, 500.0, 2.0],
        'bounds': ([1e-5, 10.0, 0.5], [1.0, 5000.0, 10.0]),
        'param_names': ['alpha_0', 'K', 'n'],
        'n_params': 3,
    },
}


# -----------------------------------------------------------------------------
# B4. CALIBRATION + MODEL SELECTION (R^2, AIC)
# -----------------------------------------------------------------------------
def calibrate_elasticity_function(form_name, ref_df=None):
    """Fit form to reference drugs. Return (params, R2, AIC)."""
    ref_df = ref_df if ref_df is not None else REFERENCE_DRUGS
    spec = ELASTICITY_FORMS[form_name]

    R_ref = reinforcement_rate(ref_df['brain_peak_dCdt'].values)
    alpha_ref = ref_df['alpha_normalized'].values
    n_data = len(alpha_ref)

    popt, _ = curve_fit(spec['func'], R_ref, alpha_ref,
                        p0=spec['p0'], bounds=spec['bounds'], maxfev=10000)

    alpha_pred = spec['func'](R_ref, *popt)
    ss_res = np.sum((alpha_ref - alpha_pred) ** 2)
    ss_tot = np.sum((alpha_ref - np.mean(alpha_ref)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # AIC for small samples (AICc would be inf with k=n; report regular AIC)
    k = spec['n_params']
    if ss_res > 0:
        aic = n_data * np.log(ss_res / n_data) + 2 * k
    else:
        aic = -np.inf
    return popt, r2, aic


def get_elasticity_function(form_name, fitted_params):
    func = ELASTICITY_FORMS[form_name]['func']
    return lambda R: func(R, *fitted_params)


def select_best_form(fits):
    """Choose the form with the lowest AIC (best fit accounting for complexity)."""
    aics = {name: vals[2] for name, vals in fits.items()}
    best = min(aics, key=aics.get)
    return best, aics


# -----------------------------------------------------------------------------
# B5. APPLY TO NSI-189 PROFILES
# -----------------------------------------------------------------------------
def compute_elasticity_profile(t, Cb, dCbdt, elasticity_func):
    R = reinforcement_rate(dCbdt)
    alpha = elasticity_func(R)
    return pd.DataFrame({
        't': t, 'Cb': Cb, 'dCbdt': dCbdt, 'R': R, 'alpha': alpha
    })


def find_inelastic_threshold(profile_df, alpha_critical):
    """First brain Cb at which alpha drops below alpha_critical."""
    sorted_df = profile_df.sort_values('Cb').reset_index(drop=True)
    below = sorted_df['alpha'] < alpha_critical
    if not below.any():
        return float('nan')
    return float(sorted_df.loc[below.idxmax(), 'Cb'])


# -----------------------------------------------------------------------------
# B6. ELASTICITY PLOTTING
# -----------------------------------------------------------------------------
def plot_calibration(fits, save_path=None):
    R_smooth = np.logspace(np.log10(10), np.log10(5000), 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'exponential': '#1f77b4', 'power': '#2ca02c', 'hill': '#d62728'}

    for form_name, (params, r2, aic) in fits.items():
        func = ELASTICITY_FORMS[form_name]['func']
        ax.plot(R_smooth, func(R_smooth, *params),
                color=colors[form_name], lw=2,
                label=f'{form_name.capitalize()} ($R^2$={r2:.3f}, AIC={aic:.2f})')

    R_ref = reinforcement_rate(REFERENCE_DRUGS['brain_peak_dCdt'].values)
    alpha_ref = REFERENCE_DRUGS['alpha_normalized'].values
    ax.scatter(R_ref, alpha_ref, s=120, c='black', zorder=5,
               edgecolor='white', linewidth=1.5, label='Reference drugs')
    for _, row in REFERENCE_DRUGS.iterrows():
        ax.annotate(row['drug'],
                    xy=(row['brain_peak_dCdt'], row['alpha_normalized']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reinforcement Rate R (brain dC/dt, ng/mL/h)')
    ax.set_ylabel(r'Elasticity $\alpha$')
    ax.set_title('Elasticity Functions Calibrated to Reference Cognitive Enhancers',
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='lower left')
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_nsi189_elasticity(profile_df, alpha_critical_dict, best_form='',
                            save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(profile_df['t'], profile_df['Cb'], color='#d62728', lw=1.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Brain $C_b$ (ng/mL)')
    ax.set_title('Brain Concentration Profile')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(profile_df['t'], profile_df['dCbdt'], color='#2ca02c', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$dC_b/dt$ (ng/mL/h)')
    ax.set_title('Brain dC/dt (Reinforcement Signal)')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(profile_df['t'], profile_df['alpha'], color='#9467bd', lw=1.5)
    for label, ac in alpha_critical_dict.items():
        ax.axhline(ac, linestyle='--', lw=1.0,
                   label=f'{label} $\\alpha$={ac}')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Elasticity $\alpha$')
    ax.set_title(f'Elasticity Over Time ({best_form})')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(profile_df['Cb'], profile_df['alpha'], s=8, alpha=0.4,
               color='#1f77b4')
    for label, ac in alpha_critical_dict.items():
        ax.axhline(ac, linestyle='--', lw=1.0,
                   label=f'{label} $\\alpha$={ac}')
    ax.set_xlabel('Brain $C_b$ (ng/mL)')
    ax.set_ylabel(r'Elasticity $\alpha$')
    ax.set_title('Elasticity vs. Brain Concentration')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f'NSI-189 Elasticity Analysis ({best_form}, best fit)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_form_comparison(profiles_by_form, alpha_critical_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {'exponential': '#1f77b4', 'power': '#2ca02c', 'hill': '#d62728'}

    for form_name, profile_df in profiles_by_form.items():
        sorted_df = profile_df.sort_values('Cb')
        ax.plot(sorted_df['Cb'], sorted_df['alpha'],
                color=colors[form_name], lw=1.5, alpha=0.8,
                label=f'{form_name.capitalize()}')

    for label, ac in alpha_critical_dict.items():
        ax.axhline(ac, linestyle='--', lw=1.0,
                   label=f'{label} ($\\alpha$={ac})')

    ax.set_xlabel('Brain $C_b$ (ng/mL)')
    ax.set_ylabel(r'Elasticity $\alpha$')
    ax.set_yscale('log')
    ax.set_title('Sensitivity Analysis: Elasticity Across Functional Forms',
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_population_elasticity(min_alpha_array, max_R_array,
                                ref_drugs_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(min_alpha_array, bins=40, color='#9467bd',
            alpha=0.7, edgecolor='white')
    for _, row in ref_drugs_df.iterrows():
        ax.axvline(row['alpha_normalized'], linestyle='--', lw=1.2,
                   label=f"{row['drug']} ($\\alpha$={row['alpha_normalized']})")
    ax.axvline(np.median(min_alpha_array), color='black', lw=2,
               label=f'NSI-189 median = {np.median(min_alpha_array):.4f}')
    ax.set_xlabel(r'Minimum $\alpha$ (most addictive moment per subject)')
    ax.set_ylabel('Frequency')
    ax.set_title('Population Distribution: Minimum Elasticity')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(max_R_array, bins=40, color='#2ca02c',
            alpha=0.7, edgecolor='white')
    for _, row in ref_drugs_df.iterrows():
        ax.axvline(row['brain_peak_dCdt'], linestyle='--', lw=1.2,
                   label=f"{row['drug']} (R={row['brain_peak_dCdt']:.0f})")
    ax.axvline(np.median(max_R_array), color='black', lw=2,
               label=f'NSI-189 median = {np.median(max_R_array):.1f}')
    ax.set_xlabel('Maximum Reinforcement Rate R (ng/mL/h)')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_title('Population Distribution: Peak Reinforcement Rate')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3, which='both')

    plt.suptitle('NSI-189 Population Elasticity Analysis (40 mg QD, Monte Carlo)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# ============================================================================
#  MAIN WORKFLOW (integrates Parts A and B)
# ============================================================================
# =============================================================================

def main():
    print("=" * 72)
    print("NSI-189 Integrated Analysis:")
    print("  Part A: mPBPK Modeling (Methodology Section 1)")
    print("  Part B: Elasticity-Based Addiction Quantification (Section 2)")
    print("=" * 72)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nBody-weight normalization settings:")
    print(f"  Reference body weight: {REFERENCE_BW_KG:.1f} kg")
    print(f"  Sampled body weight: median {WEIGHT_MEDIAN_KG:.1f} kg, CV {WEIGHT_CV:.0%}")
    print(f"  Weight-normalized dosing: {USE_WEIGHT_NORMALIZED_DOSING}")
    print(f"  Allometric parameter scaling: {USE_ALLOMETRIC_SCALING}")
    print(f"  Residual CV multiplier after BW correction: {RESIDUAL_CV_MULTIPLIER:.2f}")

    # =========================================================================
    # PART A: mPBPK MODELING
    # =========================================================================
    print("\n" + "#" * 72)
    print("# PART A: mPBPK MODELING")
    print("#" * 72)

    # ---- A1. Parameter table ----
    print("\n[A1] Parameter specifications (lognormal):")
    print(f"{'Parameter':<14} {'Median':>10} {'CV':>7} {'mu':>9} {'sigma':>9}"
          f" {'Unit':>8} {'Category':>20}")
    print("-" * 86)
    for name, (median, cv, unit, cat) in PARAM_SPECS.items():
        mu, sigma = lognormal_mu_sigma(median, cv)
        print(f"{name:<14} {median:>10.3f} {cv:>6.2%} {mu:>9.3f}"
              f" {sigma:>9.3f} {unit:>8} {cat:>20}")
    print(f"\n  Implied Kp,brain at median = {_Kp_implied:.2f}")

    # ---- A2. Clinical validation ----
    print("\n[A2] Clinical validation against Phase 1b...")
    val = validate_against_phase1b()
    plot_validation(val, save_path=out_path('fig0_clinical_validation.png'))

    # ---- A3. Regimen comparison ----
    print("\n[A3] Simulating 4 dosing regimens (typical individual)...")
    params = typical_subject()
    t_eval = np.linspace(0, 168, 1681)
    regimens = {
        '40 mg QD':  schedule_qd(40,  n_days=7),
        '40 mg BID': schedule_bid(40, n_days=7),
        '40 mg TID': schedule_tid(40, n_days=7),
        '80 mg QD':  schedule_qd(80,  n_days=7),
    }
    typ_results = {name: simulate_subject(params, sched, t_eval)
                   for name, sched in regimens.items()}
    plot_regimens(typ_results, save_path=out_path('fig1_regimens_comparison.png'))

    print("\n  Typical subject brain metrics (last 24h, day 6-7):")
    print(f"  {'Regimen':<15} {'Cmax(ng/mL)':>12} {'Tmax(h)':>10}"
          f" {'AUC(ng.h/mL)':>14} {'max_dCdt':>12}")
    for name, res in typ_results.items():
        m = extract_pk_metrics(res['t'], res['Cb'] * 1000, dose_window=(144, 168))
        print(f"  {name:<15} {m['Cmax']:>12.2f} {m['Tmax']:>10.1f}"
              f" {m['AUC']:>14.2f} {m['max_dCdt']:>12.4f}")

    # ---- A4. Monte Carlo ----
    print("\n[A4] Monte Carlo: 40 mg QD x 7 days (n=1000)...")
    sched_qd = schedule_qd(40, n_days=7)
    t_mc, Cp_mc, Cb_mc, pop_df = run_monte_carlo(sched_qd, n_subjects=1000,
                                                   t_end=168.0, dt=0.5)
    print("\n  Body-weight-normalized population summary:")
    print(f"    BW median: {pop_df['BW_kg'].median():.1f} kg "
          f"(5th-95th: {pop_df['BW_kg'].quantile(0.05):.1f}-"
          f"{pop_df['BW_kg'].quantile(0.95):.1f} kg)")
    print(f"    40 mg-equivalent dose median: {pop_df['dose_40mg_equiv_mg'].median():.1f} mg "
          f"(5th-95th: {pop_df['dose_40mg_equiv_mg'].quantile(0.05):.1f}-"
          f"{pop_df['dose_40mg_equiv_mg'].quantile(0.95):.1f} mg)")

    plot_monte_carlo(t_mc, Cp_mc, Cb_mc, title_suffix='- 40 mg QD x 7 days, BW-normalized',
                     save_path=out_path('fig2_monte_carlo_qd.png'))

    print("\n[A5] dC_b/dt distribution (elasticity input)...")
    plot_dcdt(t_mc, Cb_mc, save_path=out_path('fig3_dCbdt.png'))

    print("\n[A6] Population PK metrics at steady state...")
    m_brain, m_plasma = plot_pk_distributions(
        Cp_mc, Cb_mc, t_mc, dose_window=(144, 168),
        save_path=out_path('fig4_pk_metrics.png'))

    print("\n  Brain PK summary at steady state (40 mg QD, ng/mL units):")
    print(f"  {'Metric':<20} {'Median':>12} {'5th %ile':>12} {'95th %ile':>12}")
    print("  " + "-" * 60)
    for col in ['Cmax', 'Tmax', 'AUC', 'max_dCdt']:
        med = m_brain[col].median()
        p5 = m_brain[col].quantile(0.05)
        p95 = m_brain[col].quantile(0.95)
        print(f"  {col:<20} {med:>12.3f} {p5:>12.3f} {p95:>12.3f}")

    m_brain.to_csv(out_path('pk_metrics_brain.csv'), index=False)
    m_plasma.to_csv(out_path('pk_metrics_plasma.csv'), index=False)
    pop_df.to_csv(out_path('population_parameters.csv'), index=False)

    # =========================================================================
    # PART B: ELASTICITY-BASED ADDICTION QUANTIFICATION
    # =========================================================================
    print("\n" + "#" * 72)
    print("# PART B: ELASTICITY-BASED ADDICTION QUANTIFICATION")
    print("#" * 72)

    # ---- B1. Reference drug data ----
    print("\n[B1] Reference cognitive enhancer data (Hursh-Silberberg framework):")
    print(REFERENCE_DRUGS.to_string(index=False))

    # ---- B2. Calibrate three functional forms ----
    print("\n[B2] Calibrating elasticity functions on reference drugs...")
    fits = {}
    for form_name in ['exponential', 'power', 'hill']:
        params_fit, r2, aic = calibrate_elasticity_function(form_name)
        fits[form_name] = (params_fit, r2, aic)
        names = ELASTICITY_FORMS[form_name]['param_names']
        param_str = ', '.join([f"{n}={v:.4g}" for n, v in zip(names, params_fit)])
        print(f"  {form_name:12s}: {param_str}, R^2={r2:.4f}, AIC={aic:.3f}")

    # Best form selection
    best_form, all_aics = select_best_form(fits)
    print(f"\n  Best functional form (lowest AIC): {best_form.upper()}")

    plot_calibration(fits, save_path=out_path('fig5_elasticity_calibration.png'))

    # ---- B3. Get NSI-189 brain profile (typical subject, 40 mg QD) ----
    print("\n[B3] Computing NSI-189 brain profile for elasticity input...")
    typical_t = typ_results['40 mg QD']['t']
    typical_Cb_ngml = typ_results['40 mg QD']['Cb'] * 1000
    typical_dCbdt = np.gradient(typical_Cb_ngml, typical_t)
    print(f"  Brain Cb range: [{typical_Cb_ngml.min():.1f},"
          f" {typical_Cb_ngml.max():.1f}] ng/mL")
    print(f"  Max dC/dt: {typical_dCbdt.max():.2f} ng/mL/h")

    # ---- B4. Apply each form ----
    print("\n[B4] Computing NSI-189 elasticity per functional form...")
    profiles_by_form = {}
    for form_name in ['exponential', 'power', 'hill']:
        params_fit, _, _ = fits[form_name]
        func = get_elasticity_function(form_name, params_fit)
        profile = compute_elasticity_profile(typical_t, typical_Cb_ngml,
                                              typical_dCbdt, func)
        profiles_by_form[form_name] = profile
        min_alpha = profile['alpha'].min()
        max_alpha = profile['alpha'].max()
        print(f"  {form_name:12s}: alpha range = [{min_alpha:.5f}, {max_alpha:.5f}]")

    # ---- B5. Inelastic threshold (using reference drug alpha values as benchmarks) ----
    alpha_critical_dict = {
        'amphetamine': REFERENCE_DRUGS.loc[0, 'alpha_normalized'],
        'nicotine':    REFERENCE_DRUGS.loc[1, 'alpha_normalized'],
        'cocaine':     REFERENCE_DRUGS.loc[2, 'alpha_normalized'],
    }

    print("\n[B5] Inelastic threshold using reference drug benchmarks:")
    print(f"  {'Form':<14} {'amphet ref':>14} {'nicotine ref':>14} {'cocaine ref':>14}")
    print("  " + "-" * 60)
    threshold_records = []
    for form_name in ['exponential', 'power', 'hill']:
        row = [form_name]
        for label, ac in alpha_critical_dict.items():
            thr = find_inelastic_threshold(profiles_by_form[form_name], ac)
            row.append(thr)
            threshold_records.append({
                'form': form_name, 'reference': label,
                'alpha_critical': ac, 'threshold_Cb_ngml': thr
            })
        thr_strs = []
        for v in row[1:]:
            thr_strs.append(f"{v:>14.1f}" if not np.isnan(v) else f"{'Not reached':>14}")
        print(f"  {row[0]:<14} {' '.join(thr_strs)}")

    if all(np.isnan(r['threshold_Cb_ngml']) for r in threshold_records):
        print("\n  KEY FINDING: NSI-189 does not reach inelastic regime")
        print("  (alpha never drops to reference-drug-level) at clinical doses.")
        print("  This suggests NSI-189 has lower addictive potential than")
        print("  amphetamine, nicotine, and cocaine.")

    # ---- B6. Primary visualization (best form) ----
    print(f"\n[B6] Visualizing NSI-189 elasticity profile (best form: {best_form})...")
    plot_nsi189_elasticity(profiles_by_form[best_form],
                            alpha_critical_dict, best_form=best_form,
                            save_path=out_path('fig6_nsi189_elasticity.png'))

    # ---- B7. Form comparison (sensitivity analysis) ----
    print("\n[B7] Sensitivity analysis: comparing functional forms...")
    plot_form_comparison(profiles_by_form, alpha_critical_dict,
                          save_path=out_path('fig7_elasticity_form_comparison.png'))

    # ---- B8. Population elasticity analysis ----
    print("\n[B8] Population-level elasticity analysis (using mPBPK Monte Carlo)...")
    primary_params, _, _ = fits[best_form]
    elasticity_func = get_elasticity_function(best_form, primary_params)

    # Use already-computed Monte Carlo brain profiles
    n_subjects = Cb_mc.shape[0]
    Cb_mc_ngml = Cb_mc * 1000  # convert mg/L -> ng/mL
    dCbdt_mc = np.gradient(Cb_mc_ngml, t_mc, axis=1)

    min_alpha_per_subject = np.zeros(n_subjects)
    max_R_per_subject = np.zeros(n_subjects)
    for i in range(n_subjects):
        R_i = reinforcement_rate(dCbdt_mc[i])
        alpha_i = elasticity_func(R_i)
        min_alpha_per_subject[i] = np.min(alpha_i)
        max_R_per_subject[i] = np.max(R_i)

    plot_population_elasticity(min_alpha_per_subject, max_R_per_subject,
                                REFERENCE_DRUGS,
                                save_path=out_path('fig8_elasticity_population.png'))

    print(f"\n  Population NSI-189 minimum alpha (n={n_subjects}):")
    print(f"    Median:    {np.median(min_alpha_per_subject):.5f}")
    print(f"    5th %ile:  {np.percentile(min_alpha_per_subject, 5):.5f}"
          f"  (most addictive subset)")
    print(f"    95th %ile: {np.percentile(min_alpha_per_subject, 95):.5f}"
          f"  (least addictive subset)")

    print(f"\n  Reference drug alpha values for comparison:")
    for _, row in REFERENCE_DRUGS.iterrows():
        print(f"    {row['drug']:12s}: alpha = {row['alpha_normalized']:.5f}")

    print(f"\n  Population NSI-189 peak reinforcement rate:")
    print(f"    Median:    {np.median(max_R_per_subject):.1f} ng/mL/h")
    print(f"    5th %ile:  {np.percentile(max_R_per_subject, 5):.1f} ng/mL/h")
    print(f"    95th %ile: {np.percentile(max_R_per_subject, 95):.1f} ng/mL/h")

    # ---- B9. Save elasticity CSVs ----
    print("\n[B9] Saving elasticity analysis CSVs...")

    calib_records = []
    for form_name, (p, r2, aic) in fits.items():
        names = ELASTICITY_FORMS[form_name]['param_names']
        for pname, pval in zip(names, p):
            calib_records.append({
                'form': form_name, 'parameter': pname,
                'value': pval, 'R2': r2, 'AIC': aic
            })
    pd.DataFrame(calib_records).to_csv(
        out_path('elasticity_calibration.csv'), index=False)

    profiles_by_form[best_form].to_csv(
        out_path('nsi189_elasticity_profile.csv'), index=False)

    pd.DataFrame({
        'subject_id': np.arange(n_subjects),
        'BW_kg': pop_df['BW_kg'].values,
        'dose_40mg_equiv_mg': pop_df['dose_40mg_equiv_mg'].values,
        'min_alpha':  min_alpha_per_subject,
        'max_reinforcement_rate': max_R_per_subject,
        'steady_state_Cmax_ngml': np.max(Cb_mc_ngml[:, t_mc >= 144], axis=1),
    }).to_csv(out_path('population_elasticity.csv'), index=False)

    pd.DataFrame(threshold_records).to_csv(
        out_path('functional_form_comparison.csv'), index=False)

    # ---- Final summary ----
    print("\n" + "=" * 72)
    print("INTEGRATED ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"\nBest elasticity functional form: {best_form.upper()}")
    print(f"  - Selected by lowest AIC across exponential, power, Hill")
    print(f"\nNSI-189 minimum alpha (typical subject):"
          f" {profiles_by_form[best_form]['alpha'].min():.5f}")
    print(f"Reference alpha (amphetamine, least addictive of refs):"
          f" 0.012")
    print(f"\nInterpretation: NSI-189 alpha"
          f" {'remains above' if profiles_by_form[best_form]['alpha'].min() > 0.012 else 'crosses'}"
          f" amphetamine-level elasticity at clinical doses.")
    print(f"\nBody-weight normalization was applied to Monte Carlo simulations:")
    print(f"  - 40 mg reference dose converted to mg/kg using {REFERENCE_BW_KG:.0f} kg reference BW")
    print(f"  - Vb and Vc scaled by BW^1.0; CL terms scaled by BW^0.75")
    print(f"  - Residual CV multiplier = {RESIDUAL_CV_MULTIPLIER:.2f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == '__main__':
    main()