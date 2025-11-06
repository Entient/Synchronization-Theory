#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 1 analysis - FINAL WORKING VERSION
Fixed: basename extraction to match NPY naming exactly
"""
import os, re, json, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, hilbert, butter, filtfilt, correlate
from scipy.stats import linregress
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)
sns.set(style="whitegrid", context="talk")

def clean_cols(df):
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(' ', '_') for c in d.columns]
    return d

def standardize_condition_freq(df):
    d = df.copy()
    if 'condition' not in d.columns and 'group' in d.columns:
        d['condition'] = d['group']
    d['condition'] = d['condition'].astype(str).str.strip().str.title()
    if 'pace_hz' in d.columns:
        d['frequency'] = d['pace_hz']
    elif 'freq_hz' in d.columns:
        d['frequency'] = d['freq_hz']
    d['frequency'] = pd.to_numeric(d['frequency'], errors='coerce')
    if 'slice' in d.columns:
        d['slice'] = d['slice'].astype(str)
    else:
        d['slice'] = d['filename'].apply(lambda s: re.sub(r'[^A-Za-z0-9]+','_', os.path.basename(str(s))))
    return d

def base_stem_from_filename(fn):
    """
    CRITICAL: Must match extraction script exactly
    1. Get basename
    2. Remove extension (.tif, .tiff, etc)
    3. Replace non-alphanumeric with _
    4. Strip trailing _
    """
    b = os.path.basename(str(fn))
    # Remove extension FIRST
    b = re.sub(r'\.(tif+|tiff+|npy+|csv+)$', '', b, flags=re.I)
    # THEN replace non-alphanumeric
    b = re.sub(r'[^A-Za-z0-9]+', '_', b)
    b = b.strip('_')
    return b

def load_timeseries(timeseries_dir, filename):
    base = base_stem_from_filename(filename)
    v_path = os.path.join(timeseries_dir, f"timeseries_{base}_voltage.npy")
    c_path = os.path.join(timeseries_dir, f"timeseries_{base}_calcium.npy")
    if not os.path.exists(v_path) or not os.path.exists(c_path):
        return None, None
    v = np.load(v_path)
    c = np.load(c_path)
    if np.ndim(v) != 1 or np.ndim(c) != 1 or len(v) != len(c):
        return None, None
    return v.astype(float), c.astype(float)

def butter_hp(x, fs, hp=0.5, order=2):
    if fs is None or fs <= 0:
        return x
    nyq = fs/2.0
    wc  = min(hp/nyq, 0.99)
    try:
        b, a = butter(order, wc, btype='high')
        return filtfilt(b, a, x)
    except Exception:
        return x

def compute_plv(sig1, sig2):
    a1, a2 = hilbert(sig1), hilbert(sig2)
    dphi = np.angle(a1) - np.angle(a2)
    return np.abs(np.mean(np.exp(1j*dphi)))

def lag_crosscorr(sig1, sig2, fs, half_window_ms):
    if fs is None or fs <= 0: return np.nan
    hw = int(max(1, (half_window_ms/1000.0)*fs))
    N = len(sig1)
    x = (sig1 - np.mean(sig1))
    y = (sig2 - np.mean(sig2))
    corr = correlate(y, x, mode='full')
    lags = np.arange(-N+1, N)
    mask = (lags >= -hw) & (lags <= hw)
    if not np.any(mask):
        return np.nan
    sub = corr[mask]
    lsub = lags[mask]
    idx = int(np.argmax(sub))
    lag_samples = lsub[idx]
    return (lag_samples/fs)*1000.0

def segment_beats_from_vm(vm, fs, freq_hz, min_cycles=3):
    if fs is None or fs <= 0 or freq_hz is None or freq_hz <= 0:
        return []
    cycle_len = int(round(fs / float(freq_hz)))
    if cycle_len < 10: return []
    prom = np.std(vm) * 0.3
    peaks, _ = find_peaks(vm, distance=int(cycle_len*0.7), prominence=prom)
    if len(peaks) < min_cycles+1:
        peaks_neg, _ = find_peaks(-vm, distance=int(cycle_len*0.7), prominence=prom)
        if len(peaks_neg) > len(peaks):
            peaks = peaks_neg
    if len(peaks) < min_cycles+1:
        return []
    cycles = []
    for i in range(len(peaks)-1):
        s, e = int(peaks[i]), int(peaks[i+1])
        if e - s >= 10:
            cycles.append((s,e))
    return cycles

def per_beat_metrics(vm, ca, fs, freq_hz):
    results = dict(plv=[], lag_ms=[], dV=[], dCa=[],
                   ca_ttp_ms=[], ca_t50_ms=[], ca_t90_ms=[], ca_tau_ms=[],
                   vm_upstroke=[], vm_apd50_ms=[], vm_apd90_ms=[])
    cycles = segment_beats_from_vm(vm, fs, freq_hz)
    if not cycles: return results

    vm_hp = butter_hp(vm, fs, 0.5)
    ca_hp = butter_hp(ca, fs, 0.5)
    T_ms = 1000.0 / float(freq_hz)

    for (s,e) in cycles:
        v_seg = vm_hp[s:e]
        c_seg = ca_hp[s:e]
        v_raw = vm[s:e]
        c_raw = ca[s:e]
        if len(v_seg) < 10 or len(c_seg) < 10:
            continue

        results['plv'].append(compute_plv(v_seg, c_seg))
        results['lag_ms'].append(lag_crosscorr(v_seg, c_seg, fs, half_window_ms=0.5*T_ms))
        results['dV'].append(float(np.max(v_raw) - np.min(v_raw)))
        results['dCa'].append(float(np.max(c_raw) - np.min(c_raw)))

        # Ca kinetics
        peak_idx = int(np.argmax(c_raw))
        ttp_ms = (peak_idx / fs) * 1000.0
        results['ca_ttp_ms'].append(ttp_ms)

        # Decay from peak
        decay = c_raw[peak_idx:]
        if len(decay) < 3:
            results['ca_t50_ms'].append(np.nan)
            results['ca_t90_ms'].append(np.nan)
            results['ca_tau_ms'].append(np.nan)
        else:
            pk = decay[0]
            bs = np.min(decay)
            amp = pk - bs
            if amp > 1e-6:
                t50_idx = np.where(decay <= (pk - 0.5*amp))[0]
                t90_idx = np.where(decay <= (pk - 0.9*amp))[0]
                t50 = (t50_idx[0]/fs)*1000.0 if len(t50_idx)>0 else np.nan
                t90 = (t90_idx[0]/fs)*1000.0 if len(t90_idx)>0 else np.nan
                results['ca_t50_ms'].append(t50)
                results['ca_t90_ms'].append(t90)
                # Exponential tau
                try:
                    from scipy.optimize import curve_fit
                    def exp_decay(t, tau, offset):
                        return amp*np.exp(-t/tau) + offset
                    xdata = np.arange(len(decay))/fs*1000.0
                    ydata = decay - bs
                    popt, _ = curve_fit(exp_decay, xdata, ydata, p0=[50.0, 0.0], maxfev=1000)
                    results['ca_tau_ms'].append(abs(popt[0]))
                except:
                    results['ca_tau_ms'].append(np.nan)
            else:
                results['ca_t50_ms'].append(np.nan)
                results['ca_t90_ms'].append(np.nan)
                results['ca_tau_ms'].append(np.nan)

        # Vm upstroke
        dv = np.diff(v_raw)*fs
        results['vm_upstroke'].append(float(np.max(dv)) if len(dv)>0 else np.nan)

        # APD surrogates
        vpk = np.max(v_raw)
        vbs = np.min(v_raw)
        vamp = vpk - vbs
        if vamp > 1e-6:
            thresh50 = vbs + 0.5*vamp
            thresh90 = vbs + 0.1*vamp
            above50 = (v_raw > thresh50).astype(int)
            above10 = (v_raw > thresh90).astype(int)
            apd50 = (np.sum(above50)/fs)*1000.0
            apd90 = (np.sum(above10)/fs)*1000.0
            results['vm_apd50_ms'].append(apd50)
            results['vm_apd90_ms'].append(apd90)
        else:
            results['vm_apd50_ms'].append(np.nan)
            results['vm_apd90_ms'].append(np.nan)

    return results

def plv_lag_variability(per_beat_dict):
    plv = np.array([x for x in per_beat_dict['plv'] if np.isfinite(x)])
    lag = np.array([x for x in per_beat_dict['lag_ms'] if np.isfinite(x)])
    
    def stats(arr):
        if len(arr) < 2:
            return np.nan, np.nan, np.nan
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        cv = s/abs(m) if abs(m)>1e-9 else np.nan
        return m, s, cv
    
    plv_m, plv_sd, plv_cv = stats(plv)
    lag_m, lag_sd, lag_cv = stats(lag)
    return plv_m, plv_sd, plv_cv, lag_m, lag_sd, lag_cv

def ec_gain_from_beats(per_beat_dict):
    dv = np.array([x for x in per_beat_dict['dV'] if np.isfinite(x)])
    dc = np.array([x for x in per_beat_dict['dCa'] if np.isfinite(x)])
    n  = min(dv.size, dc.size)
    if n < 3:
        return np.nan, np.nan, 0
    slope, intercept, r, p, se = linregress(dv[:n], dc[:n])
    return float(slope), float(r), int(n)

def run_tier1(input_csv, timeseries_dir, output_csv, fig_dir, fs_hint=250.0, hp_for_plv=0.5):
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = clean_cols(df)
    df = standardize_condition_freq(df)
    df['base'] = df['filename'].apply(lambda s: base_stem_from_filename(s))

    newcols = {
        'plv_beat_mean':[], 'plv_beat_sd':[], 'plv_beat_cv':[],
        'lag_beat_mean':[], 'lag_beat_sd':[], 'lag_beat_cv':[],
        'ec_coupling_gain':[], 'ec_gain_r':[], 'ec_gain_n_beats':[],
        'ca_ttp_ms':[], 'ca_t50_ms':[], 'ca_t90_ms':[], 'ca_tau_ms':[],
        'vm_upstroke':[], 'vm_apd50_ms':[], 'vm_apd90_ms':[]
    }

    loaded_count = 0
    for idx, row in df.iterrows():
        v, c = load_timeseries(timeseries_dir, row['filename'])
        if v is None or c is None:
            for k in newcols: newcols[k].append(np.nan)
            continue

        loaded_count += 1
        fs = fs_hint
        freq = row.get('frequency', np.nan)
        try:
            freq = float(freq)
        except:
            freq = np.nan

        pb = per_beat_metrics(v, c, fs, freq)
        plv_m, plv_sd, plv_cv, lag_m, lag_sd, lag_cv = plv_lag_variability(pb)
        gain, r, nbeats = ec_gain_from_beats(pb)

        def med(lst):
            a = np.array([x for x in lst if np.isfinite(x)])
            return float(np.median(a)) if a.size>0 else np.nan

        newcols['plv_beat_mean'].append(plv_m)
        newcols['plv_beat_sd'  ].append(plv_sd)
        newcols['plv_beat_cv'  ].append(plv_cv)
        newcols['lag_beat_mean'].append(lag_m)
        newcols['lag_beat_sd'  ].append(lag_sd)
        newcols['lag_beat_cv'  ].append(lag_cv)
        newcols['ec_coupling_gain'].append(gain)
        newcols['ec_gain_r'].append(r)
        newcols['ec_gain_n_beats'].append(nbeats)
        newcols['ca_ttp_ms' ].append(med(pb['ca_ttp_ms']))
        newcols['ca_t50_ms' ].append(med(pb['ca_t50_ms']))
        newcols['ca_t90_ms' ].append(med(pb['ca_t90_ms']))
        newcols['ca_tau_ms' ].append(med(pb['ca_tau_ms']))
        newcols['vm_upstroke' ].append(med(pb['vm_upstroke']))
        newcols['vm_apd50_ms' ].append(med(pb['vm_apd50_ms']))
        newcols['vm_apd90_ms' ].append(med(pb['vm_apd90_ms']))

    for k,v in newcols.items():
        df[k] = v

    df.to_csv(output_csv, index=False)
    print(f"✓ Loaded {loaded_count}/{len(df)} timeseries files")
    print(f"✓ Saved enriched CSV: {output_csv}")

    # Figures
    fig1 = os.path.join(fig_dir, "Tier1_Fig1_variability_violin.png")
    plt.figure(figsize=(12,6))
    sns.violinplot(data=df, x='frequency', y='plv_beat_cv', hue='condition',
                   inner='box', cut=0, palette='Set2')
    plt.title("PLV Beat-wise CV (Instability) by Frequency and Condition")
    plt.ylabel("PLV CV (per beat)")
    plt.tight_layout(); plt.savefig(fig1, dpi=200); plt.close()
    print(f"Saved: {fig1}")

    fig1b = os.path.join(fig_dir, "Tier1_Fig1b_lagvariability_violin.png")
    plt.figure(figsize=(12,6))
    sns.violinplot(data=df, x='frequency', y='lag_beat_cv', hue='condition',
                   inner='box', cut=0, palette='Set1')
    plt.title("Lag Beat-wise CV (Instability) by Frequency and Condition")
    plt.ylabel("Lag CV (per beat)")
    plt.tight_layout(); plt.savefig(fig1b, dpi=200); plt.close()
    print(f"Saved: {fig1b}")

    fig2 = os.path.join(fig_dir, "Tier1_Fig2_ec_gain_by_group.png")
    plt.figure(figsize=(10,6))
    sns.violinplot(data=df, x='condition', y='ec_coupling_gain', hue='frequency', palette='Set3', inner='box', cut=0)
    plt.title("E–C Coupling Gain (ΔCa vs ΔV slope) by Condition & Frequency")
    plt.ylabel("Gain (ΔCa / ΔV)")
    plt.legend(title='Frequency (Hz)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(fig2, dpi=200, bbox_inches='tight'); plt.close()
    print(f"Saved: {fig2}")

    for metric, title, fname in [
        ('ca_ttp_ms', 'Calcium Time-to-Peak (ms)', 'Tier1_Fig3_ca_ttp.png'),
        ('ca_tau_ms', 'Calcium Decay Tau (ms)', 'Tier1_Fig3_ca_tau.png'),
        ('vm_upstroke', 'Vm Upstroke Slope (a.u./s)', 'Tier1_Fig3_vm_upstroke.png'),
        ('vm_apd50_ms', 'Vm APD50 surrogate (ms)', 'Tier1_Fig3_vm_apd50.png'),
        ('vm_apd90_ms', 'Vm APD90 surrogate (ms)', 'Tier1_Fig3_vm_apd90.png'),
    ]:
        fpath = os.path.join(fig_dir, fname)
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x='frequency', y=metric, hue='condition', palette='Paired',
                    errorbar='sd', err_kws={'linewidth': 1.5}, capsize=0.1)
        plt.title(title)
        plt.tight_layout(); plt.savefig(fpath, dpi=200); plt.close()
        print(f"Saved: {fpath}")

    # Mixed effects
    results_rows = []
    metrics_for_mlm = [
        'plv_beat_mean','plv_beat_cv',
        'lag_beat_mean','lag_beat_cv',
        'ec_coupling_gain',
        'ca_ttp_ms','ca_tau_ms',
        'vm_upstroke','vm_apd50_ms','vm_apd90_ms'
    ]
    dmm = df.copy()
    dmm = dmm[np.isfinite(dmm['frequency'])]
    dmm = dmm.dropna(subset=['slice'])
    dmm['condition'] = dmm['condition'].astype('category')
    if 'Sham' in list(dmm['condition'].cat.categories):
        dmm['condition'] = dmm['condition'].cat.reorder_categories(['Sham'] + [c for c in dmm['condition'].cat.categories if c!='Sham'])

    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    mlm_text = []
    for metric in metrics_for_mlm:
        sub = dmm[['slice','condition','frequency',metric]].dropna()
        if sub.shape[0] < 12:
            continue
        try:
            model = smf.mixedlm(f"{metric} ~ condition * frequency", data=sub, groups=sub["slice"])
            fit = model.fit(reml=True, method='lbfgs', maxiter=500, disp=False)
            mlm_text.append(f"\n=== Mixed Effects for {metric} ===\n{fit.summary()}")
            pvals = fit.pvalues.to_dict()
            params = fit.params.to_dict()
            results_rows.append({
                'metric': metric,
                'n': sub.shape[0],
                'AIC': fit.aic, 'BIC': fit.bic,
                'p_condition': pvals.get('condition[T.Tac]', np.nan),
                'p_frequency': pvals.get('frequency', np.nan),
                'p_interaction': pvals.get('condition[T.Tac]:frequency', np.nan)
            })
        except Exception as e:
            mlm_text.append(f"\n=== Mixed Effects for {metric} ===\nERROR: {e}")

    mix_tab = pd.DataFrame(results_rows)
    mix_csv = os.path.join(fig_dir, "Tier1_mixed_effects_results.csv")
    mix_tab.to_csv(mix_csv, index=False)
    print(f"Saved: {mix_csv}")

    mix_txt = os.path.join(fig_dir, "Tier1_mixed_effects_summary.txt")
    with open(mix_txt, "w") as f:
        f.write("\n".join(mlm_text))
    print(f"Saved: {mix_txt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--timeseries_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--fs_hint", type=float, default=250.0)
    args = ap.parse_args()

    run_tier1(
        input_csv=args.input_csv,
        timeseries_dir=args.timeseries_dir,
        output_csv=args.output_csv,
        fig_dir=args.fig_dir,
        fs_hint=args.fs_hint
    )
