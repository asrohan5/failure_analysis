"""
Aero Engine Survival Analysis — Streamlit Application
======================================================
Upload a failure dataset, the app runs the full pipeline automatically:
    1. IID verification (Laplace trend + Mann-Kendall + Spearman lag)
    2. Distribution fitting (Exponential, Weibull, Lognormal via MLE)
       OR NHPP Power Law if trend is confirmed
    3. Goodness-of-fit (Anderson-Darling primary, KS supplementary)
       OR Cramer-von Mises for NHPP
    4. Parameter display and reliability metrics
    5. Diagnostic plots
    6. N(t): expected failure count for user-supplied time t
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm, weibull_min, expon, lognorm, kstest
from scipy.special import gamma as gamma_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, expon


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engine Failure Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
  }
  .stApp { background-color: #0d1117; }

  h1 { font-family: 'IBM Plex Mono', monospace; color: #58a6ff;
       font-size: 1.9rem; letter-spacing: -0.5px; margin-bottom: 0; }
  h2 { font-family: 'IBM Plex Mono', monospace; color: #58a6ff;
       font-size: 1.15rem; border-bottom: 1px solid #21262d;
       padding-bottom: 6px; margin-top: 2rem; }
  h3 { font-family: 'IBM Plex Sans', sans-serif; color: #8b949e;
       font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
       letter-spacing: 1px; margin-top: 1.2rem; }

  .metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 18px 22px; margin: 6px 0;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #8b949e; letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 4px;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem; color: #f0f6fc; font-weight: 600;
  }
  .metric-unit {
    font-size: 0.8rem; color: #8b949e; margin-left: 4px;
  }

  .pass-badge {
    display: inline-block; background: #1a3a2a; color: #3fb950;
    border: 1px solid #3fb950; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; padding: 2px 10px; font-weight: 600;
  }
  .fail-badge {
    display: inline-block; background: #3a1a1a; color: #f85149;
    border: 1px solid #f85149; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; padding: 2px 10px; font-weight: 600;
  }
  .warn-badge {
    display: inline-block; background: #2d2a1a; color: #e3b341;
    border: 1px solid #e3b341; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; padding: 2px 10px; font-weight: 600;
  }
  .model-banner {
    background: #1c2128; border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0; padding: 14px 20px; margin: 12px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem;
    color: #c9d1d9;
  }
  .note-box {
    background: #1c2128; border: 1px solid #21262d;
    border-radius: 6px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.82rem; color: #8b949e; line-height: 1.6;
  }
  .stFileUploader { border: 1px dashed #30363d; border-radius: 8px;
                    padding: 10px; }
  div[data-testid="stNumberInput"] input {
    font-family: 'IBM Plex Mono', monospace;
    background: #161b22; border: 1px solid #30363d; color: #f0f6fc;
  }
  .stButton > button {
    background: #1f6feb; color: #ffffff; border: none;
    border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; padding: 8px 20px; font-weight: 600;
    width: 100%;
  }
  .stButton > button:hover { background: #388bfd; }
</style>
""", unsafe_allow_html=True)


# ── PLOT THEME ────────────────────────────────────────────────────────────────
PLOT_BG   = '#0d1117'
PLOT_FACE = '#161b22'
GRID_COL  = '#21262d'
TEXT_COL  = '#8b949e'
BLUE      = '#58a6ff'
ORANGE    = '#e3b341'
GREEN     = '#3fb950'
RED       = '#f85149'

def apply_dark_theme(ax):
    ax.set_facecolor(PLOT_FACE)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color('#c9d1d9')
    ax.title.set_fontsize(9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8)

# ── PLOT THEME (matches existing scripts) ─────────────────────────────────────
PLOT_BG   = '#0d1117'
PLOT_FACE = '#161b22'
GRID_COL  = '#21262d'
TEXT_COL  = '#8b949e'
BLUE      = '#58a6ff'
ORANGE    = '#e3b341'
GREEN     = '#3fb950'
RED       = '#f85149'

def _apply_theme(ax):
    ax.set_facecolor(PLOT_FACE)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color('#c9d1d9')
    ax.title.set_fontsize(9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8)


def _empirical_cdf(tbf_sorted):
    """Benard's median rank approximation — standard in reliability engineering."""
    n = len(tbf_sorted)
    return (np.arange(1, n + 1) - 0.3) / (n + 0.4)
# ── STATISTICAL FUNCTIONS ─────────────────────────────────────────────────────

def extract_tbf(df):
    col = [c for c in df.columns if 'since' in c.lower() and 'last' in c.lower()]
    if not col:
        raise ValueError("Could not find 'Failure Time Since Last' column.")
    tbf = df[col[0]].to_numpy(dtype=float, copy=True)
    tbf = tbf[1:].copy()
    neg_count = int((tbf < 0).sum())
    tbf[tbf < 0] = 0.0
    return tbf, neg_count

def extract_cumtime(df):
    col = [c for c in df.columns
           if 'cumulative' in c.lower() and 'hr' in c.lower()
           and 'mtbf' not in c.lower()]
    if not col:
        raise ValueError("Could not find 'Cumulative Time in Hrs' column.")
    return df[col[0]].to_numpy(dtype=float, copy=True)

# ── IID TESTS ─────────────────────────────────────────────────────────────────

def laplace_test(cum_times):
    X = np.sort(cum_times.astype(float))
    N = len(X)
    if N < 3:
        return np.nan, np.nan, 'insufficient data'
    T, Xs, n = X[-1], X[:-1], N - 1
    U = (Xs.mean() - T / 2.0) / (T * np.sqrt(1.0 / (12.0 * n)))
    p = 2.0 * (1.0 - norm.cdf(abs(U)))
    Z_CRIT = 1.96
    if abs(U) <= Z_CRIT:
        dec = 'no significant trend'
    elif U > Z_CRIT:
        dec = 'deteriorating trend'
    else:
        dec = 'improving trend'
    return float(U), float(p), dec

def mann_kendall(tbf):
    x = tbf.astype(float); n = len(x)
    if n < 4:
        return np.nan, np.nan, 'insufficient data'
    s = float(sum(np.sum(np.sign(x[i+1:] - x[i])) for i in range(n-1)))
    _, cnts = np.unique(x, return_counts=True); tc = cnts[cnts > 1]
    var = (n*(n-1)*(2*n+5) - np.sum(tc*(tc-1)*(2*tc+5))) / 18.0
    z = (s-1)/np.sqrt(var) if s > 0 else ((s+1)/np.sqrt(var) if s < 0 else 0.0)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(z), float(p), ('significant trend' if p < 0.05 else 'no significant trend')

def spearman_lag(tbf, lag):
    rs, p = spearmanr(tbf[:-lag], tbf[lag:])
    dec = 'rejected' if p < 0.05 else 'not rejected'
    return float(rs), float(p), dec

def run_iid(tbf, cum_times):
    lap_U, lap_p, lap_dec = laplace_test(cum_times)
    mk_Z,  mk_p,  mk_dec  = mann_kendall(tbf)
    rs1, p1, d1 = spearman_lag(tbf, 1)
    rs2, p2, d2 = spearman_lag(tbf, 2)
    trend_pass = (lap_dec == 'no significant trend' and mk_dec == 'no significant trend')
    indep_pass = (d1 == 'not rejected' and d2 == 'not rejected')
    return {
        'lap_U': lap_U, 'lap_p': lap_p, 'lap_dec': lap_dec,
        'mk_Z': mk_Z,   'mk_p': mk_p,   'mk_dec': mk_dec,
        'rs1': rs1, 'p1': p1, 'd1': d1,
        'rs2': rs2, 'p2': p2, 'd2': d2,
        'trend_pass': trend_pass, 'indep_pass': indep_pass,
        'iid_pass': trend_pass and indep_pass,
    }

# ── DISTRIBUTION FITTING ──────────────────────────────────────────────────────

def ad_statistic(tbf, dist, params):
    n = len(tbf); tbf_s = np.sort(tbf)
    u = np.clip(dist.cdf(tbf_s, *params), 1e-10, 1-1e-10)
    i = np.arange(1, n+1)
    return float(-n - np.sum((2*i-1)/n * (np.log(u) + np.log(1-u[::-1]))))

def fit_distributions(tbf):
    loc_e, sc_e = expon.fit(tbf, floc=0)
    sh_w, loc_w, sc_w = weibull_min.fit(tbf, floc=0)
    sh_ln, loc_ln, sc_ln = lognorm.fit(tbf, floc=0)
    mu_ln, sig_ln = np.log(sc_ln), sh_ln

    fits = {
        'Exponential': {
            'params': (loc_e, sc_e), 'dist': expon,
            'lambda': 1/sc_e, 'mtbf': sc_e,
            'label': f'λ = {1/sc_e:.4f} /hr',
        },
        'Weibull': {
            'params': (sh_w, loc_w, sc_w), 'dist': weibull_min,
            'beta': sh_w, 'eta': sc_w,
            'mtbf': sc_w * gamma_fn(1 + 1/sh_w),
            'label': f'β = {sh_w:.4f},  η = {sc_w:.4f} hr',
        },
        'Lognormal': {
            'params': (sh_ln, loc_ln, sc_ln), 'dist': lognorm,
            'mu': mu_ln, 'sigma': sig_ln,
            'mtbf': float(np.exp(mu_ln + sig_ln**2/2)),
            'label': f'μ = {mu_ln:.4f},  σ = {sig_ln:.4f}',
        },
    }

    for name, f in fits.items():
        f['ad'] = ad_statistic(tbf, f['dist'], f['params'])
        ks_s, ks_p = kstest(tbf, lambda x, d=f['dist'], p=f['params']: d.cdf(x, *p))
        f['ks_stat'] = float(ks_s); f['ks_p'] = float(ks_p)

    best = min(fits, key=lambda k: fits[k]['ad'])
    b10  = fits[best]['dist'].ppf(0.10, *fits[best]['params'])
    fits[best]['b10'] = float(b10)
    return fits, best

# ── NHPP POWER LAW ────────────────────────────────────────────────────────────

def fit_nhpp(cum_times):
    t = np.sort(cum_times.astype(float))
    n, T = len(t), t[-1]
    denom = np.sum(np.log(T / t[:-1]))
    beta  = (n - 1) / denom
    lam   = n / (T ** beta)
    # CvM GoF
    z = np.sort((t / T) ** beta)
    i = np.arange(1, n+1)
    cvm = float(np.sum((z - (2*i-1)/(2*n))**2) + 1.0/(12*n))
    cvm_dec = 'not rejected (good fit)' if cvm < 0.1937 else 'rejected'
    return {
        'beta': float(beta), 'lambda': float(lam),
        'T': float(T), 'n': n,
        'cvm': cvm, 'cvm_dec': cvm_dec,
        'rho_T': float(lam * beta * T**(beta-1)),
        'mtbf_T': float(1.0 / (lam * beta * T**(beta-1))),
    }

# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_iid(tbf, cum_times, iid):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.patch.set_facecolor(PLOT_BG)

    # Cumulative failures vs time
    ax = axes[0]
    ax.step(cum_times, np.arange(1, len(cum_times)+1),
            where='post', color=BLUE, lw=1.4)
    ax.plot([0, cum_times[-1]], [0, len(cum_times)],
            '--', color=RED, lw=1, label='HPP reference')
    ax.set_xlabel('Cumulative time (hr)'); ax.set_ylabel('Cumulative failures')
    ax.set_title('Cumulative Failures vs Time'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    # TBF sequence
    ax = axes[1]
    ax.scatter(np.arange(1, len(tbf)+1), tbf,
               s=14, color=BLUE, edgecolor=GRID_COL, lw=0.3, alpha=0.8)
    ax.axhline(np.mean(tbf), color=ORANGE, lw=1, linestyle='--',
               label=f'Mean = {np.mean(tbf):.3f} hr')
    ax.set_xlabel('Failure index'); ax.set_ylabel('TBF (hr)')
    ax.set_title('TBF Sequence'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    # Lag-1 scatter
    ax = axes[2]
    ax.scatter(tbf[:-1], tbf[1:], s=16, color=BLUE,
               edgecolor=GRID_COL, lw=0.3, alpha=0.75)
    ax.set_xlabel('TBF(i) [hr]'); ax.set_ylabel('TBF(i+1) [hr]')
    ax.set_title(f'Lag-1 Scatter  rs={iid["rs1"]:+.3f}  p={iid["p1"]:.3f}')
    apply_dark_theme(ax)

    plt.tight_layout(pad=1.2)
    return fig

def plot_dist_fit(tbf, fits, best):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.patch.set_facecolor(PLOT_BG)
    colors = {'Exponential': RED, 'Weibull': ORANGE, 'Lognormal': GREEN}
    t_range = np.linspace(0.001, np.percentile(tbf, 99)*1.3, 400)

    # Empirical vs fitted CDF
    ax = axes[0]
    tbf_s = np.sort(tbf)
    ecdf  = np.arange(1, len(tbf_s)+1) / len(tbf_s)
    ax.step(tbf_s, ecdf, where='post', color='#f0f6fc',
            lw=1.4, label='Empirical', zorder=5)
    for name, f in fits.items():
        ls = '--' if name == best else ':'
        ax.plot(t_range, f['dist'].cdf(t_range, *f['params']),
                color=colors[name], lw=1.6, linestyle=ls,
                label=f'{name} (AD={f["ad"]:.3f})')
    ax.set_xlabel('TBF (hr)'); ax.set_ylabel('Cumulative probability')
    ax.set_title('Empirical vs Fitted CDF'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    # Reliability function R(t) — best fit
    ax = axes[1]
    best_f  = fits[best]
    R_t = best_f['dist'].sf(t_range, *best_f['params'])
    ax.plot(t_range, R_t, color=BLUE, lw=2)
    if 'b10' in best_f:
        ax.axvline(best_f['b10'], color=RED, lw=1, linestyle='--',
                   label=f'B10 = {best_f["b10"]:.3f} hr')
        ax.axhline(0.9, color=RED, lw=0.6, linestyle=':')
    ax.set_xlabel('Time (hr)'); ax.set_ylabel('R(t)')
    ax.set_title(f'Reliability Function — {best}'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)
    ax.set_ylim(0, 1)

    # Weibull probability plot
    ax = axes[2]
    n_pos = len(tbf_s[tbf_s > 0]); tbf_pos = tbf_s[tbf_s > 0]
    mr = (np.arange(1, n_pos+1) - 0.3) / (n_pos + 0.4)
    ax.scatter(np.log(tbf_pos), np.log(-np.log(1-mr)),
               s=16, color=BLUE, edgecolor=GRID_COL, lw=0.3, zorder=5)
    beta_w = fits['Weibull']['beta']; eta_w = fits['Weibull']['eta']
    xf = np.linspace(np.log(tbf_pos.min()), np.log(tbf_pos.max()), 100)
    ax.plot(xf, beta_w*xf - beta_w*np.log(eta_w),
            color=RED, lw=1.5,
            label=f'Weibull  β={beta_w:.3f}')
    ax.set_xlabel('ln(TBF)'); ax.set_ylabel('ln(-ln(1-F))')
    ax.set_title('Weibull Probability Plot'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    plt.tight_layout(pad=1.2)
    return fig

def plot_nhpp(cum_times, nh):
    beta, lam, T, n = nh['beta'], nh['lambda'], nh['T'], nh['n']
    t_range = np.linspace(0.05, T * 1.05, 400)
    EN_t  = lam * t_range**beta
    rho_t = lam * beta * t_range**(beta-1)
    MTBF_t = 1.0 / rho_t

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.patch.set_facecolor(PLOT_BG)

    ax = axes[0]
    ax.step(cum_times, np.arange(1, n+1), where='post',
            color='#f0f6fc', lw=1.4, label='Observed')
    ax.plot(t_range, EN_t, color=BLUE, lw=2, linestyle='--',
            label=f'Power Law E[N(t)]')
    ax.set_xlabel('Test time (hr)'); ax.set_ylabel('Cumulative failures')
    ax.set_title('Observed vs Model'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    ax = axes[1]
    ax.plot(t_range, rho_t, color=ORANGE, lw=2)
    ax.axvline(T, color=RED, lw=1, linestyle='--',
               label=f'End T={T:.1f} hr')
    ax.set_xlabel('Test time (hr)'); ax.set_ylabel('ρ(t) failures/hr')
    ax.set_title(f'Failure Intensity  β={beta:.4f}'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    ax = axes[2]
    ax.scatter(np.log(cum_times), np.log(np.arange(1, n+1)),
               s=16, color=BLUE, edgecolor=GRID_COL, lw=0.3, zorder=5)
    log_tr = np.linspace(np.log(cum_times[0]), np.log(cum_times[-1]), 200)
    ax.plot(log_tr, np.log(lam) + beta*log_tr, color=RED, lw=1.5,
            label=f'Fit  slope=β={beta:.4f}')
    ax.set_xlabel('ln(t)'); ax.set_ylabel('ln(N(t))')
    ax.set_title('Crow-AMSAA Plot'); apply_dark_theme(ax)
    ax.legend(fontsize=7, labelcolor=TEXT_COL,
              facecolor=PLOT_FACE, edgecolor=GRID_COL)

    plt.tight_layout(pad=1.2)
    return fig

# ── RELIABILITY METRICS FROM FITTED PARAMETERS ───────────────────────────────
# t/MTBF is only the asymptotic renewal approximation.
# The functions below use the actual fitted shape parameters to compute:
#   R(t)      — survival probability for a mission of duration t
#   F(t)      — probability of failure before t
#   h(t)      — instantaneous failure rate at operating age t
#   B(p)      — age by which fraction p of units have failed
#   N(t)      — expected failure count (t/MTBF, valid asymptotically)
# For NHPP: R_mission = exp(-N(T,T+t)) from Poisson zero-probability.

from scipy.stats import norm as _norm
from scipy.special import gamma as _gamma_fn

def reliability_metrics_lognormal(t, mu, sigma):
    from scipy.stats import lognorm as _ln
    R = float(_ln.sf(t, s=sigma, scale=np.exp(mu)))
    h = float(_ln.pdf(t, s=sigma, scale=np.exp(mu)) / (R + 1e-15))
    mtbf = float(np.exp(mu + sigma**2 / 2))
    B = {p: float(np.exp(mu + sigma * _norm.ppf(p))) for p in [0.10, 0.50, 0.90]}
    return {'R': R, 'F': 1-R, 'h': h, 'mtbf': mtbf, 'N': t/mtbf, 'B': B}

def reliability_metrics_weibull(t, beta, eta):
    R = float(np.exp(-(t/eta)**beta))
    h = float((beta/eta) * (t/eta)**(beta-1))
    mtbf = float(eta * _gamma_fn(1 + 1/beta))
    B = {p: float(eta * (-np.log(1-p))**(1/beta)) for p in [0.10, 0.50, 0.90]}
    return {'R': R, 'F': 1-R, 'h': h, 'mtbf': mtbf, 'N': t/mtbf, 'B': B}

def reliability_metrics_nhpp(t_mission, beta, lam, T_obs):
    rho_now  = lam * beta * T_obs**(beta-1)
    rho_end  = lam * beta * (T_obs+t_mission)**(beta-1)
    N_future = lam*(T_obs+t_mission)**beta - lam*T_obs**beta
    R_miss   = float(np.exp(-N_future))
    return {
        'rho_now': rho_now, 'mtbf_now': 1/rho_now,
        'rho_end': rho_end, 'mtbf_end': 1/rho_end,
        'N_future': N_future, 'R_mission': R_miss,
    }

def plot_qq_streamlit(tbf, fit_results, best):
    """
    Returns a matplotlib figure for use with st.pyplot() in the Streamlit app.
    Drop this call right after st.pyplot(plot_dist_fit(...)) in app.py.

    Usage in app.py:
        st.markdown('<h3>Q-Q Probability Plot</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class=\"note-box\">Points on the diagonal indicate a good fit. '
            'Curvature in the lower-left means the distribution overestimates '
            'early failures. Curvature in the upper-right means it underestimates '
            'the tail. The best-fit distribution (highlighted) should track the '
            'diagonal most closely.</p>', unsafe_allow_html=True)
        st.pyplot(plot_qq_streamlit(tbf, fits, best), use_container_width=True)
    """
    tbf_sorted = np.sort(tbf)
    emp        = _empirical_cdf(tbf_sorted)

    colors = {'Exponential': RED, 'Weibull': ORANGE, 'Lognormal': GREEN}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fig.patch.set_facecolor(PLOT_BG)

    for ax, (name, res) in zip(axes, fit_results.items()):
        theo = res['dist'].cdf(tbf_sorted, *res['params'])

        # highlight best fit with larger points and solid border
        is_best   = (name == best)
        size      = 28 if is_best else 18
        edge_col  = 'white' if is_best else GRID_COL
        edge_w    = 0.8 if is_best else 0.3

        ax.scatter(theo, emp, s=size, color=colors[name],
                   edgecolor=edge_col, linewidth=edge_w,
                   alpha=0.9, zorder=5)

        ax.plot([0, 1], [0, 1], color='white', linewidth=1.0,
                linestyle='--', alpha=0.5)

        for t_val, e_val in zip(theo, emp):
            ax.plot([t_val, t_val], [t_val, e_val],
                    color=colors[name], linewidth=0.4, alpha=0.3)

        ad_val = res.get('ad', float('nan'))
        ks_p   = res.get('ks_p', float('nan'))
        suffix = '  BEST FIT' if is_best else ''
        ax.set_xlabel('Theoretical CDF F(t)')
        ax.set_ylabel('Empirical CDF')
        ax.set_title(f'{name}{suffix}\nAD={ad_val:.3f}  KS p={ks_p:.3f}')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        _apply_theme(ax)

    plt.suptitle(
        'Q-Q Probability Plot — Points on diagonal = perfect fit',
        fontsize=9, color='#c9d1d9', y=1.02)
    plt.tight_layout()
    return fig


# ── MAIN APP ──────────────────────────────────────────────────────────────────

st.markdown('<h1>Engine Failure Analysis</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#8b949e; font-size:0.9rem; margin-top:0; margin-bottom:2rem;">'
    'Upload failure data — the pipeline runs automatically: '
    'IID verification → model selection → distribution fitting → reliability metrics'
    '</p>', unsafe_allow_html=True)

# ── DEMO LOADERS ─────────────────────────────────────────────────────────────
st.markdown('<h3>Try a demo dataset</h3>', unsafe_allow_html=True)
col_d1, col_d2, col_d3 = st.columns([1, 1, 3])
load_pass = col_d1.button("Demo — IID Pass", use_container_width=True)
load_fail = col_d2.button("Demo — IID Fail", use_container_width=True)

# session state: remember which demo (if any) is loaded
if load_pass:
    st.session_state['demo'] = 'pass'
elif load_fail:
    st.session_state['demo'] = 'fail'

uploaded = st.file_uploader(
    "Or upload your own engine failure data (.xlsx or .csv)",
    type=['xlsx', 'csv'],
    label_visibility='visible'
)
# uploading a new file clears the demo selection
if uploaded is not None:
    st.session_state['demo'] = None

# determine data source
demo_choice = st.session_state.get('demo', None)

if uploaded is None and demo_choice is None:
    st.markdown("""
    <div class="note-box">
    Expected columns: <code>Failure No.</code>, <code>Build No.</code>,
    <code>Cumulative Time in Hrs</code>, <code>Failure Time Since Last in Hrs</code>,
    <code>Cumulative MTBF</code>. First row of each engine type is excluded
    from the TBF sequence (no predecessor). Negative TBF values are corrected
    to zero automatically.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
import os
DEMO_DIR = 'demo_data/'

try:
    if uploaded is not None:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        source_label = uploaded.name
    elif demo_choice == 'pass':
        df = pd.read_excel(os.path.join(DEMO_DIR, 'demo_IID_PASS.xlsx'))
        source_label = 'Demo dataset — IID Pass (exponential, no trend, no dependence)'
        st.info(source_label)
    elif demo_choice == 'fail':
        df = pd.read_excel(os.path.join(DEMO_DIR, 'demo_IID_FAIL.xlsx'))
        source_label = 'Demo dataset — IID Fail (deteriorating trend + AR(1) serial dependence)'
        st.info(source_label)
    else:
        st.stop()
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

try:
    tbf, neg_count = extract_tbf(df)
    cum_times = extract_cumtime(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

n_zero = int((tbf == 0).sum())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Failures loaded</div>'
                f'<div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">TBF sequence length</div>'
                f'<div class="metric-value">{len(tbf)}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Simultaneous (TBF=0)</div>'
                f'<div class="metric-value">{n_zero}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Negative TBF corrected</div>'
                f'<div class="metric-value">{neg_count}</div></div>', unsafe_allow_html=True)

# ── STEP 1: IID ───────────────────────────────────────────────────────────────
st.markdown('<h2>Step 1 — IID Verification</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="note-box">Trend is tested before independence. '
    'A trend in the failure process would confound serial correlation results. '
    'Laplace (primary) + Mann-Kendall (distribution-free corroboration) for trend; '
    'Spearman rank correlation at lag-1 and lag-2 for independence.</p>',
    unsafe_allow_html=True)

iid = run_iid(tbf, cum_times)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<h3>Trend Tests</h3>', unsafe_allow_html=True)
    trend_badge = '<span class="pass-badge">PASS</span>' if iid['trend_pass'] else '<span class="fail-badge">FAIL</span>'
    st.markdown(f"""
    <div class="metric-card">
      <div style="margin-bottom:8px">{trend_badge}</div>
      <div class="metric-label">Laplace U</div>
      <div class="metric-value" style="font-size:1.1rem">{iid['lap_U']:+.4f}
        <span class="metric-unit">p = {iid['lap_p']:.4f}</span></div>
      <div style="color:#8b949e;font-size:0.8rem;margin-top:4px">{iid['lap_dec']}</div>
      <div style="margin-top:12px" class="metric-label">Mann-Kendall Z</div>
      <div class="metric-value" style="font-size:1.1rem">{iid['mk_Z']:+.4f}
        <span class="metric-unit">p = {iid['mk_p']:.4f}</span></div>
      <div style="color:#8b949e;font-size:0.8rem;margin-top:4px">{iid['mk_dec']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<h3>Independence Tests</h3>', unsafe_allow_html=True)
    indep_badge = '<span class="pass-badge">PASS</span>' if iid['indep_pass'] else '<span class="fail-badge">FAIL</span>'
    st.markdown(f"""
    <div class="metric-card">
      <div style="margin-bottom:8px">{indep_badge}</div>
      <div class="metric-label">Spearman Lag-1 (primary)</div>
      <div class="metric-value" style="font-size:1.1rem">rs = {iid['rs1']:+.4f}
        <span class="metric-unit">p = {iid['p1']:.4f}</span></div>
      <div style="color:#8b949e;font-size:0.8rem;margin-top:4px">Independence {iid['d1']}</div>
      <div style="margin-top:12px" class="metric-label">Spearman Lag-2 (robustness)</div>
      <div class="metric-value" style="font-size:1.1rem">rs = {iid['rs2']:+.4f}
        <span class="metric-unit">p = {iid['p2']:.4f}</span></div>
      <div style="color:#8b949e;font-size:0.8rem;margin-top:4px">Independence {iid['d2']}</div>
    </div>
    """, unsafe_allow_html=True)

iid_verdict_badge = '<span class="pass-badge">IID SUPPORTED</span>' if iid['iid_pass'] else '<span class="fail-badge">IID NOT SUPPORTED</span>'
if iid['trend_pass'] != iid['indep_pass']:
    iid_verdict_badge = '<span class="warn-badge">BORDERLINE</span>'

st.markdown(f'<div style="margin:12px 0">Overall verdict: {iid_verdict_badge}</div>',
            unsafe_allow_html=True)

st.pyplot(plot_iid(tbf, cum_times, iid), use_container_width=True)

# ── STEP 2: MODEL SELECTION AND FITTING ──────────────────────────────────────
use_nhpp = not iid['trend_pass']

if use_nhpp:
    st.markdown('<h2>Step 2 — NHPP Power Law Model (Crow-AMSAA)</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-banner">
    Trend confirmed → static distribution fitting is not appropriate.
    The failure intensity is changing over time. The Power Law NHPP
    (Crow-AMSAA) is fitted instead. It directly models how the cumulative
    failure count evolves with test time, allowing for increasing or
    decreasing failure intensity.
    </div>
    """, unsafe_allow_html=True)

    nh = fit_nhpp(cum_times)
    beta, lam, T_obs = nh['beta'], nh['lambda'], nh['T']

    trend_direction = 'Improving (burn-in / reliability growth)' if beta < 1 else \
                      ('Deteriorating (wear-out)' if beta > 1 else 'Constant (HPP equivalent)')

    col1, col2, col3, col4 = st.columns(4)
    metrics_nhpp = [
        ('beta (shape)', f'{beta:.4f}', ''),
        ('lambda (scale)', f'{lam:.4f}', '/hr^β'),
        ('MTBF at end of test', f'{nh["mtbf_T"]:.4f}', 'hr'),
        ('Failure intensity ρ(T)', f'{nh["rho_T"]:.4f}', '/hr'),
    ]
    for col, (label, val, unit) in zip([col1, col2, col3, col4], metrics_nhpp):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{val}<span class="metric-unit">{unit}</span>'
                        f'</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="margin-top:8px">
      <div class="metric-label">Trend direction</div>
      <div style="color:#c9d1d9; font-size:0.9rem; margin-top:4px">{trend_direction}</div>
      <div style="margin-top:10px" class="metric-label">Goodness of Fit — Cramer-von Mises</div>
      <div class="metric-value" style="font-size:1rem">C_M = {nh['cvm']:.4f}
        <span class="metric-unit"> → {nh['cvm_dec']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.pyplot(plot_nhpp(cum_times, nh), use_container_width=True)

    # Reliability metrics for NHPP
    st.markdown('<h2>Reliability Metrics — Mission Analysis</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="note-box">For NHPP, the mission reliability R(t) is the probability '
        'of completing the next t hours without a failure, given the current test age T. '
        'It uses the Poisson zero-probability formula: R = exp(-N(T, T+t)), where '
        'N(T, T+t) = lambda*(T+t)^beta - lambda*T^beta is the expected failure count '
        'in the mission window. This correctly accounts for the time-varying failure '
        'intensity — unlike t/MTBF which assumes a constant rate.</p>',
        unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        t_input = st.number_input(
            'Mission duration t (hours)', min_value=0.1,
            max_value=10000.0, value=5.0, step=0.5,
            key='t_nhpp'
        )
    rm = reliability_metrics_nhpp(t_input, beta, lam, T_obs)
    with col2:
        r_color = GREEN if rm['R_mission'] >= 0.9 else (ORANGE if rm['R_mission'] >= 0.5 else RED)
        st.markdown(f"""
        <div class="metric-card">
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
            <div>
              <div class="metric-label">R_mission — P(zero failures in {t_input:.1f} hr)</div>
              <div class="metric-value" style="color:{r_color}">{rm['R_mission']:.4f}
                <span class="metric-unit">({rm['R_mission']*100:.1f}%)</span></div>
            </div>
            <div>
              <div class="metric-label">Expected failures N(T, T+t)</div>
              <div class="metric-value">{rm['N_future']:.4f}
                <span class="metric-unit">≈ {round(rm['N_future'])}</span></div>
            </div>
            <div>
              <div class="metric-label">Failure intensity NOW ρ(T)</div>
              <div class="metric-value" style="font-size:1.1rem">{rm['rho_now']:.4f}
                <span class="metric-unit">/hr</span></div>
            </div>
            <div>
              <div class="metric-label">Instantaneous MTBF NOW</div>
              <div class="metric-value" style="font-size:1.1rem">{rm['mtbf_now']:.4f}
                <span class="metric-unit">hr</span></div>
            </div>
            <div>
              <div class="metric-label">Failure intensity at end ρ(T+t)</div>
              <div class="metric-value" style="font-size:1.1rem">{rm['rho_end']:.4f}
                <span class="metric-unit">/hr</span></div>
            </div>
            <div>
              <div class="metric-label">Instantaneous MTBF at end</div>
              <div class="metric-value" style="font-size:1.1rem">{rm['mtbf_end']:.4f}
                <span class="metric-unit">hr</span></div>
            </div>
          </div>
          <div style="margin-top:10px;color:#8b949e;font-size:0.78rem">
          R = exp(-N(T,T+t)) = exp(-{rm['N_future']:.4f}) = {rm['R_mission']:.4f} &nbsp;|&nbsp;
          Current age T = {T_obs:.2f} hr &nbsp;|&nbsp;
          Mission end T+t = {T_obs+t_input:.2f} hr
          </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown('<h2>Step 2 — Distribution Fitting</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-banner">
    IID confirmed → inter-failure times are treated as independent draws
    from a fixed distribution. Three candidates fitted via MLE:
    Exponential, Weibull, Lognormal. Best fit selected by
    Anderson-Darling statistic (lower = better). KS test reported
    for supplementary confirmation.
    </div>
    """, unsafe_allow_html=True)

    fits, best = fit_distributions(tbf)
    best_f = fits[best]

    col1, col2, col3 = st.columns(3)
    for col, (name, f) in zip([col1, col2, col3], fits.items()):
        border_color = BLUE if name == best else '#21262d'
        best_marker  = ' ✓ Best fit' if name == best else ''
        with col:
            ks_badge = 'not rejected' if f['ks_p'] >= 0.05 else 'rejected'
            ks_color = GREEN if f['ks_p'] >= 0.05 else RED
            st.markdown(f"""
            <div class="metric-card" style="border-color:{border_color}">
              <div class="metric-label">{name}{best_marker}</div>
              <div style="color:#c9d1d9;font-size:0.82rem;margin:6px 0">{f['label']}</div>
              <div class="metric-label" style="margin-top:8px">MTBF</div>
              <div class="metric-value" style="font-size:1.1rem">{f['mtbf']:.4f}
                <span class="metric-unit">hr</span></div>
              <div class="metric-label" style="margin-top:8px">AD statistic</div>
              <div style="font-family:monospace;font-size:0.9rem;color:#f0f6fc">{f['ad']:.4f}</div>
              <div class="metric-label" style="margin-top:8px">KS p-value</div>
              <div style="font-family:monospace;font-size:0.9rem;color:{ks_color}">
                {f['ks_p']:.4f} ({ks_badge})</div>
            </div>
            """, unsafe_allow_html=True)

    if 'b10' in best_f:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:8px">
          <div class="metric-label">Best fit — {best} — B10 Life</div>
          <div class="metric-value">{best_f['b10']:.4f}
            <span class="metric-unit">hr  (10% of failures expected by this time)</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.pyplot(plot_dist_fit(tbf, fits, best), use_container_width=True)

    # Reliability metrics from fitted distribution
    st.markdown('<h2>Reliability Metrics — Mission Analysis</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="note-box">Enter a mission time t to compute reliability metrics '
        'directly from the fitted distribution parameters. R(t) and h(t) use mu/sigma '
        '(Lognormal) or beta/eta (Weibull) — not just the mean. '
        'N(t) = t/MTBF is also shown as a planning approximation.</p>',
        unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        t_input = st.number_input(
            'Mission time t (hours)', min_value=0.001,
            max_value=10000.0, value=2.0, step=0.1,
            key='t_iid'
        )

    # compute metrics using fitted shape parameters
    if best == 'Lognormal':
        rm = reliability_metrics_lognormal(
            t_input, best_f['mu'], best_f['sigma'])
    elif best == 'Weibull':
        rm = reliability_metrics_weibull(
            t_input, best_f['beta'], best_f['eta'])
    else:  # Exponential — same as Weibull with beta=1
        rm = {'R': float(np.exp(-best_f['lambda']*t_input)),
              'F': float(1-np.exp(-best_f['lambda']*t_input)),
              'h': float(best_f['lambda']),
              'mtbf': best_f['mtbf'],
              'N': t_input/best_f['mtbf'],
              'B': {p: float(-np.log(1-p)/best_f['lambda'])
                   for p in [0.10,0.50,0.90]}}

    r_color = GREEN if rm['R'] >= 0.9 else (ORANGE if rm['R'] >= 0.5 else RED)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
            <div>
              <div class="metric-label">R(t) — P(survive beyond {t_input:.2f} hr)</div>
              <div class="metric-value" style="color:{r_color}">{rm['R']:.4f}
                <span class="metric-unit">({rm['R']*100:.1f}%)</span></div>
            </div>
            <div>
              <div class="metric-label">F(t) — P(failure before {t_input:.2f} hr)</div>
              <div class="metric-value">{rm['F']:.4f}
                <span class="metric-unit">({rm['F']*100:.1f}%)</span></div>
            </div>
            <div>
              <div class="metric-label">h(t) — instantaneous failure rate</div>
              <div class="metric-value" style="font-size:1.1rem">{rm['h']:.4f}
                <span class="metric-unit">/hr</span></div>
            </div>
            <div>
              <div class="metric-label">N(t) — expected failures (t/MTBF)</div>
              <div class="metric-value" style="font-size:1.1rem">{rm['N']:.4f}
                <span class="metric-unit">≈ {round(rm['N'])}</span></div>
            </div>
          </div>
          <div style="margin-top:14px">
            <div class="metric-label">B-lives (percentile lives from fitted parameters)</div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:6px">
              <div style="font-family:monospace;font-size:0.85rem;color:#f0f6fc">
                B10 = {rm['B'][0.10]:.4f} hr<br>
                <span style="color:#8b949e;font-size:0.75rem">10% failed by this age</span></div>
              <div style="font-family:monospace;font-size:0.85rem;color:#f0f6fc">
                B50 = {rm['B'][0.50]:.4f} hr<br>
                <span style="color:#8b949e;font-size:0.75rem">median life</span></div>
              <div style="font-family:monospace;font-size:0.85rem;color:#f0f6fc">
                B90 = {rm['B'][0.90]:.4f} hr<br>
                <span style="color:#8b949e;font-size:0.75rem">90% failed by this age</span></div>
            </div>
          </div>
          <div style="margin-top:10px;color:#8b949e;font-size:0.78rem">
          Best fit: {best} &nbsp;|&nbsp; MTBF = {rm['mtbf']:.4f} hr &nbsp;|&nbsp;
          R(t) and h(t) computed from shape parameters, not just the mean
          </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Compare metrics across all fitted distributions"):
        for name, f in fits.items():
            if name == 'Lognormal':
                m = reliability_metrics_lognormal(t_input, f['mu'], f['sigma'])
            elif name == 'Weibull':
                m = reliability_metrics_weibull(t_input, f['beta'], f['eta'])
            else:
                m = {'R': float(np.exp(-f['lambda']*t_input)),
                     'h': float(f['lambda']), 'N': t_input/f['mtbf']}
            best_mark = ' (best fit)' if name == best else ''
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.82rem;'
                f'color:#c9d1d9;padding:5px 0;border-bottom:1px solid #21262d">'
                f'<b>{name}{best_mark}</b> &nbsp; '
                f'R({t_input:.1f}hr)={m["R"]:.4f} &nbsp; '
                f'h({t_input:.1f}hr)={m["h"]:.4f}/hr &nbsp; '
                f'N={m["N"]:.3f}</div>',
                unsafe_allow_html=True)

# ── RAW DATA PREVIEW ──────────────────────────────────────────────────────────
with st.expander("Raw data preview"):
    st.dataframe(df.head(20), use_container_width=True)
