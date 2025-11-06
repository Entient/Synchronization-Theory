#!/usr/bin/env python3
"""
FINAL_CORRECTED_plv_analysis.py
Includes:
1. Proper detrending to remove photobleaching
2. Frame time offset correction
3. Cross-correlation for lag
"""
import numpy as np
from tifffile import TiffFile
from scipy.signal import butter, filtfilt, hilbert, correlate, detrend
import sys
import re

def analyze_final(filepath):
    """Ultimate corrected analysis"""
    
    print(f"\nAnalyzing: {filepath}")
    print("="*80)
    
    # Load
    with TiffFile(filepath) as tf:
        arr = tf.asarray()
    
    # SWAPPED assignment (based on your findings)
    vm_stack = arr[1::2].astype(np.float32)  # ODD frames = Vm
    ca_stack = arr[0::2].astype(np.float32)  # EVEN frames = Ca
    
    # Camera parameters
    fs_camera = 500.0  # Total camera rate (Hz)
    fs_channel = fs_camera / 2  # Per-channel rate (Hz)
    frame_time = 1000.0 / fs_camera  # Time between frames (ms)
    
    print(f"Camera rate: {fs_camera} Hz")
    print(f"Channel rate: {fs_channel} Hz per channel")
    print(f"Frame time: {frame_time} ms")
    print(f"Frames per channel: {vm_stack.shape[0]}")
    print(f"Duration: {vm_stack.shape[0]/fs_channel:.2f} s")
    
    # ROI detection
    varmap = np.var(vm_stack, axis=0)
    y, x = np.unravel_index(np.argmax(varmap), varmap.shape)
    y1, y2 = max(0, y-32), min(vm_stack.shape[1], y+32)
    x1, x2 = max(0, x-32), min(vm_stack.shape[2], x+32)
    
    print(f"ROI: y=[{y1}:{y2}], x=[{x1}:{x2}]")
    
    # Extract traces
    vm_trace_raw = vm_stack[:, y1:y2, x1:x2].mean(axis=(1,2))
    ca_trace_raw = ca_stack[:, y1:y2, x1:x2].mean(axis=(1,2))
    
    # FIX #1: Remove photobleaching/drift using detrend
    print("\nApplying detrending to remove photobleaching...")
    vm_trace = detrend(vm_trace_raw, type='linear')
    ca_trace = detrend(ca_trace_raw, type='linear')
    
    # Check improvement
    vm_drift_removed = np.std(vm_trace_raw) - np.std(vm_trace)
    ca_drift_removed = np.std(ca_trace_raw) - np.std(ca_trace)
    print(f"  Vm drift removed: {vm_drift_removed:.1f} a.u.")
    print(f"  Ca drift removed: {ca_drift_removed:.1f} a.u.")
    
    # Bandpass filter
    def bandpass(x, fs):
        nyq = fs / 2
        b, a = butter(3, [0.5/nyq, min(100/nyq, 0.99)], 'bandpass')
        return filtfilt(b, a, x - np.mean(x))
    
    vm_filt = bandpass(vm_trace, fs_channel)
    ca_filt = bandpass(ca_trace, fs_channel)
    
    # PLV calculation
    phi1 = np.angle(hilbert(vm_filt))
    phi2 = np.angle(hilbert(ca_filt))
    dphi = phi1 - phi2
    
    plv = np.abs(np.mean(np.exp(1j * dphi)))
    
    # Stability
    dphi_unwrap = np.unwrap(dphi)
    mu = np.mean(dphi_unwrap)
    sigma = np.std(dphi_unwrap)
    sigmamu = sigma / abs(mu) if abs(mu) > 1e-10 else np.inf
    
    # Cross-correlation for lag (uncorrected)
    corr = correlate(ca_filt, vm_filt, mode='same')
    center = len(corr) // 2
    lag_samples = np.argmax(corr) - center
    lag_ms_uncorrected = (lag_samples / fs_channel) * 1000
    
    # FIX #2: Correct for frame time offset
    # Ca frames (0,2,4...) are sampled BEFORE Vm frames (1,3,5...)
    # So if we measure "Ca after Vm by X ms", the true lag is X + frame_time
    lag_ms_corrected = lag_ms_uncorrected + frame_time
    
    print(f"\nLag calculation:")
    print(f"  Raw cross-correlation: {lag_ms_uncorrected:+.2f} ms")
    print(f"  Frame time offset: +{frame_time:.2f} ms")
    print(f"  Corrected Ca lag: {lag_ms_corrected:+.2f} ms")
    
    # SNR calculation (now with detrended signals)
    def calc_snr(signal):
        signal_power = np.var(signal)
        b, a = butter(3, 0.3, btype='high')
        noise = filtfilt(b, a, signal)
        noise_power = np.var(noise)
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    
    vm_snr = calc_snr(vm_trace)  # Using detrended
    ca_snr = calc_snr(ca_trace)
    
    # Extract pacing frequency
    match = re.search(r'(\d+)\s*Hz', filepath)
    pace_hz = int(match.group(1)) if match else None
    
    # Print results
    print("\n" + "="*80)
    print("FINAL CORRECTED RESULTS")
    print("="*80)
    print(f"Pacing frequency: {pace_hz} Hz")
    print(f"PLV: {plv:.4f}")
    print(f"σ/μ: {sigmamu:.4f}")
    print(f"Ca lag (corrected): {lag_ms_corrected:+.2f} ms")
    print(f"Vm SNR (after detrend): {vm_snr:.1f} dB")
    print(f"Ca SNR (after detrend): {ca_snr:.1f} dB")
    
    # Quality assessment
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT")
    print("="*80)
    
    # Lag check
    if lag_ms_corrected < 0:
        print("Ca lag      : ✗ NEGATIVE - channels still wrong!")
    elif lag_ms_corrected < 10:
        print(f"Ca lag      : ⚠️  Very fast ({lag_ms_corrected:.1f} ms)")
    elif 10 <= lag_ms_corrected < 60:
        print(f"Ca lag      : ✓ Normal range ({lag_ms_corrected:.1f} ms)")
    elif 60 <= lag_ms_corrected < 100:
        print(f"Ca lag      : ⚠️  Prolonged ({lag_ms_corrected:.1f} ms)")
    else:
        print(f"Ca lag      : ⚠️  Unusually long ({lag_ms_corrected:.1f} ms)")
    
    # PLV check
    if plv > 0.85:
        print(f"PLV         : ✓ Excellent ({plv:.3f})")
    elif plv > 0.70:
        print(f"PLV         : ✓ Good ({plv:.3f})")
    elif plv > 0.50:
        print(f"PLV         : ○ Moderate ({plv:.3f})")
    else:
        print(f"PLV         : ⚠️  Low ({plv:.3f})")
    
    # Stability
    if sigmamu < 0.5:
        print(f"σ/μ         : ✓ Stable ({sigmamu:.3f})")
    elif sigmamu < 0.8:
        print(f"σ/μ         : ○ Moderate ({sigmamu:.3f})")
    else:
        print(f"σ/μ         : ⚠️  Variable ({sigmamu:.3f})")
    
    # SNR
    if vm_snr > 20 and ca_snr > 20:
        print(f"SNR         : ✓ High quality (Vm:{vm_snr:.0f}, Ca:{ca_snr:.0f} dB)")
    elif vm_snr > 15 and ca_snr > 15:
        print(f"SNR         : ○ Acceptable (Vm:{vm_snr:.0f}, Ca:{ca_snr:.0f} dB)")
    else:
        print(f"SNR         : ⚠️  Low quality (Vm:{vm_snr:.0f}, Ca:{ca_snr:.0f} dB)")
    
    # χ = 0.437 test
    if pace_hz and 5 < lag_ms_corrected < 100:
        cycle_ms = 1000.0 / pace_hz
        lag_ratio = lag_ms_corrected / cycle_ms
        
        print("\n" + "="*80)
        print("ENTIENT χ = 0.437 TEST")
        print("="*80)
        print(f"Pacing frequency: {pace_hz} Hz")
        print(f"Cycle period: {cycle_ms:.1f} ms")
        print(f"Ca lag (corrected): {lag_ms_corrected:.2f} ms")
        print(f"Lag / Cycle ratio: {lag_ratio:.4f}")
        print(f"χ prediction: 0.437")
        print(f"Difference: {abs(lag_ratio - 0.437):.4f}")
        
        if abs(lag_ratio - 0.437) < 0.05:
            print("\n✓✓✓ EXCELLENT MATCH to χ = 0.437!")
        elif abs(lag_ratio - 0.437) < 0.1:
            print("\n✓✓ GOOD - Close to χ = 0.437")
        elif abs(lag_ratio - 0.437) < 0.2:
            print("\n✓ Within range of χ")
        else:
            print(f"\n○ Ratio {lag_ratio:.3f} differs from χ")
            print(f"   (May match at different frequency)")
    
    print("\n" + "="*80)
    
    return {
        'plv': plv,
        'sigmamu': sigmamu,
        'lag_ms_uncorrected': lag_ms_uncorrected,
        'lag_ms_corrected': lag_ms_corrected,
        'vm_snr': vm_snr,
        'ca_snr': ca_snr,
        'pace_hz': pace_hz
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FINAL_CORRECTED_plv_analysis.py <tif_file>")
        print("\nFixes applied:")
        print("  1. Detrending to remove photobleaching")
        print("  2. Frame time offset correction (+2ms at 250Hz)")
        print("  3. Cross-correlation for accurate lag")
        sys.exit(1)
    
    filepath = sys.argv[1]
    results = analyze_final(filepath)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Configuration: SWAPPED channels, 500 Hz camera, 250 Hz per channel")
    print(f"PLV: {results['plv']:.4f}")
    print(f"Ca lag (with corrections): {results['lag_ms_corrected']:.2f} ms")
    print(f"This is your FINAL corrected result!")
    print("="*80)
