#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 2 analysis - FIXED VERSION
- Band-limited PLV at 2/4/8/16 Hz (±15% BW)
- Signed hysteresis orientation (per beat & per file)
- Alternans screening (amplitude alternans AI for Ca and Vm)
"""

import os, re, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert, find_peaks

warnings.simplefilter("ignore")
sns.set(style="whitegrid", context="talk")

def clean_cols(df):
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(' ','_') for c in d.columns]
    return d

def standardize(df):
    d = df.copy()
    if 'condition' not in d.columns and 'group' in d.columns:
        d['condition'] = d['group']
    d['condition'] = d['condition'].astype(str).str.strip().str.title()
    if 'frequency' not in d.columns:
        if 'pace_hz' in d.columns: d['frequency'] = d['pace_hz']
        elif 'freq_hz' in d.columns: d['frequency'] = d['freq_hz']
    d['frequency'] = pd.to_numeric(d['frequency'], errors='coerce')
    return d

def stem_from_filename(fn):
    """CRITICAL: Must match extraction script exactly"""
    b = os.path.basename(str(fn))
    # Remove extension FIRST
    b = re.sub(r'\.(tif+|tiff+|npy+|csv+)$', '', b, flags=re.I)
    # THEN replace non-alphanumeric
    b = re.sub(r'[^A-Za-z0-9]+', '_', b)
    b = b.strip('_')
    return b

def load_timeseries(timeseries_dir, filename):
    stem = stem_from_filename(filename)
    v = os.path.join(timeseries_dir, f"timeseries_{stem}_voltage.npy")
    c = os.path.join(timeseries_dir, f"timeseries_{stem}_calcium.npy")
    if not (os.path.exists(v) and os.path.exists(c)):
        return None, None
    vm = np.load(v).astype(float)
    ca = np.load(c).astype(float)
    if vm.ndim != 1 or ca.ndim != 1 or len(vm) != len(ca):
        return None, None
    return vm, ca

def bp_filter(x, fs, f0, frac=0.15, order=3):
    """Bandpass ±frac around center f0"""
    if fs<=0 or f0<=0: return x
    nyq = fs/2.0
    lo = max(0.01, f0*(1-frac))
    hi = min(nyq*0.99, f0*(1+frac))
    if lo >= hi: return x
    b,a = butter(order, [lo/nyq, hi/nyq], btype='band')
    try:
        return filtfilt(b,a,x)
    except Exception:
        return x

def compute_plv(sig1, sig2):
    a1,a2 = hilbert(sig1), hilbert(sig2)
    dphi = np.angle(a1) - np.angle(a2)
    return float(np.abs(np.mean(np.exp(1j*dphi))))

def segment_beats(vm, fs, freq_hz, min_cycles=3):
    if fs<=0 or freq_hz<=0: return []
    cycle = int(round(fs/freq_hz))
    if cycle < 10: return []
    prom = np.std(vm)*0.3
    peaks, _ = find_peaks(vm, distance=int(cycle*0.7), prominence=prom)
    if len(peaks) < min_cycles+1:
        peaks2, _ = find_peaks(-vm, distance=int(cycle*0.7), prominence=prom)
        if len(peaks2)>len(peaks): peaks = peaks2
    if len(peaks) < min_cycles+1: return []
    cyc = []
    for i in range(len(peaks)-1):
        s,e = int(peaks[i]), int(peaks[i+1])
        if e-s>=10: cyc.append((s,e))
    return cyc

def signed_hysteresis(V, C):
    """Signed polygon area (no abs); >0 = counter-clockwise; <0 = clockwise."""
    Vn = (V - np.min(V)) / (np.ptp(V)+1e-12)
    Cn = (C - np.min(C)) / (np.ptp(C)+1e-12)
    area = 0.0
    n = len(Vn)
    for i in range(n):
        j = (i+1)%n
        area += Vn[i]*Cn[j] - Cn[i]*Vn[j]
    return 0.5*area

def amplitude_alternans(amplitudes):
    """AI = mean |A_n - A_{n+1}| / mean(A), n>=1; returns (AI, n_pairs)."""
    a = np.array([x for x in amplitudes if np.isfinite(x)])
    if a.size < 3: return np.nan, 0
    diffs = np.abs(a[1:] - a[:-1])
    denom = np.mean(a) + 1e-12
    return float(np.mean(diffs)/denom), int(diffs.size)

def run_tier2(input_csv, timeseries_dir, output_csv, fig_dir,
              fs_hint=250.0, plv_freqs=(2,4,8,16), bw_frac=0.15):
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    df = standardize(clean_cols(df))
    df['base'] = df['filename'].apply(stem_from_filename)

    # New columns
    for f in plv_freqs:
        df[f'plv_band_{f}hz'] = np.nan
    df['orientation_frac_ccw'] = np.nan
    df['orientation_nbeats'] = np.nan
    df['ai_ca'] = np.nan; df['ai_ca_n'] = np.nan
    df['ai_vm'] = np.nan; df['ai_vm_n'] = np.nan

    loaded_count = 0
    for i,row in df.iterrows():
        vm, ca = load_timeseries(timeseries_dir, row['filename'])
        if vm is None or ca is None:
            continue
        
        loaded_count += 1
        fs = fs_hint
        freq = float(row.get('frequency', np.nan)) if pd.notna(row.get('frequency', np.nan)) else np.nan

        # Band-limited PLV
        for f0 in plv_freqs:
            vfb = bp_filter(vm, fs, f0, frac=bw_frac)
            cfb = bp_filter(ca, fs, f0, frac=bw_frac)
            df.loc[i, f'plv_band_{f0}hz'] = compute_plv(vfb, cfb)

        # Signed hysteresis orientation per beat
        cyc = segment_beats(vm, fs, freq if pd.notna(freq) else plv_freqs[0])
        ccw = 0; nbeats = 0
        dV, dC = [], []
        for s,e in cyc:
            v = vm[s:e]; c = ca[s:e]
            if len(v) < 10 or len(c) < 10: continue
            area = signed_hysteresis(v, c)
            if np.isfinite(area):
                nbeats += 1
                if area > 0: ccw += 1
            dV.append(np.max(v)-np.min(v))
            dC.append(np.max(c)-np.min(c))
        if nbeats > 0:
            df.loc[i,'orientation_frac_ccw'] = ccw/nbeats
            df.loc[i,'orientation_nbeats'] = nbeats

        # Alternans indices
        ai_ca, n_ca = amplitude_alternans(dC)
        ai_vm, n_vm = amplitude_alternans(dV)
        df.loc[i,'ai_ca'] = ai_ca; df.loc[i,'ai_ca_n'] = n_ca
        df.loc[i,'ai_vm'] = ai_vm; df.loc[i,'ai_vm_n'] = n_vm

    df.to_csv(output_csv, index=False)
    print(f"✓ Loaded {loaded_count}/{len(df)} timeseries files")
    print(f"✓ Saved Tier-2 CSV: {output_csv}")

    # Figures
    # 1) Band-limited PLV
    plv_cols = [c for c in df.columns if c.startswith("plv_band_")]
    plv_long = (df[['condition','frequency']+plv_cols]
                .melt(id_vars=['condition','frequency'], var_name='band', value_name='plv'))
    plv_long['f_label'] = plv_long['band'].str.extract(r'(\d+)hz')[0].astype(float)
    fig1 = os.path.join(fig_dir, "Tier2_Fig1_bandlimited_plv.png")
    sub = plv_long.dropna(subset=['plv','f_label','condition'])
    if not sub.empty:
        plt.figure(figsize=(11,6))
        sns.boxplot(data=sub, x='f_label', y='plv', hue='condition', palette='Set2')
        plt.title("Band-limited PLV by Center Frequency and Condition")
        plt.xlabel("Band center (Hz)"); plt.ylabel("PLV (band-limited)")
        plt.tight_layout(); plt.savefig(fig1, dpi=200); plt.close()
        print(f"Saved: {fig1}")

    # 2) Orientation fraction
    fig2 = os.path.join(fig_dir, "Tier2_Fig2_orientation_fraction.png")
    sub2 = df.dropna(subset=['orientation_frac_ccw','condition','frequency'])
    if not sub2.empty:
        plt.figure(figsize=(10,6))
        sns.boxplot(data=sub2, x='frequency', y='orientation_frac_ccw', hue='condition', palette='Pastel2')
        plt.title("Fraction of CCW V–Ca loops (orientation) per file")
        plt.ylabel("CCW fraction"); plt.xlabel("Pacing Frequency (Hz)")
        plt.tight_layout(); plt.savefig(fig2, dpi=200); plt.close()
        print(f"Saved: {fig2}")

    # 3) Alternans incidence
    fig3 = os.path.join(fig_dir, "Tier2_Fig3_alternans_incidence.png")
    sub3 = df.dropna(subset=['ai_ca','condition','frequency']).copy()
    if not sub3.empty:
        thr = 0.10
        sub3['alt_pos'] = (sub3['ai_ca'] > thr).astype(int)
        inc = (sub3.groupby(['condition','frequency'])['alt_pos']
               .agg(['mean','count']).reset_index())
        inc['percent'] = 100.0*inc['mean']
        plt.figure(figsize=(10,6))
        sns.barplot(data=inc, x='frequency', y='percent', hue='condition', palette='Set1', errorbar=None)
        plt.title(f"Calcium Alternans Incidence (AI > {thr:.2f})")
        plt.ylabel("Files with alternans (%)"); plt.xlabel("Pacing Frequency (Hz)")
        plt.tight_layout(); plt.savefig(fig3, dpi=200); plt.close()
        print(f"Saved: {fig3}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--timeseries_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--fs_hint", type=float, default=250.0)
    args = ap.parse_args()
    run_tier2(
        input_csv=args.input_csv,
        timeseries_dir=args.timeseries_dir,
        output_csv=args.output_csv,
        fig_dir=args.fig_dir,
        fs_hint=args.fs_hint
    )
