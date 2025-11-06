#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRECTED extraction - channels properly assigned
Channel 0 = Calcium (sharp transient)
Channel 1 = Voltage (plateau)
"""

import os, re
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm import tqdm

def clean_basename(filepath):
    base = os.path.basename(str(filepath))
    base = re.sub(r'\.(tif+|tiff+)$', '', base, flags=re.I)
    base = re.sub(r'[^A-Za-z0-9]+', '_', base)
    base = base.strip('_')
    return base

def extract_roi_timeseries(tif_stack, roi):
    y1, y2, x1, x2 = roi
    
    if tif_stack.ndim == 4:
        # CORRECTED: Channel 0 is Calcium, Channel 1 is Voltage
        calcium = tif_stack[:, 0, y1:y2, x1:x2].mean(axis=(1, 2))
        voltage = tif_stack[:, 1, y1:y2, x1:x2].mean(axis=(1, 2))
    elif tif_stack.ndim == 3:
        n_frames = tif_stack.shape[0]
        if n_frames % 2 == 0:
            # Assume alternating frames
            calcium = tif_stack[0::2, y1:y2, x1:x2].mean(axis=(1, 2))
            voltage = tif_stack[1::2, y1:y2, x1:x2].mean(axis=(1, 2))
        else:
            mid = n_frames // 2
            calcium = tif_stack[:mid, y1:y2, x1:x2].mean(axis=(1, 2))
            voltage = tif_stack[mid:, y1:y2, x1:x2].mean(axis=(1, 2))
    else:
        raise ValueError(f"Unexpected TIF dimensions: {tif_stack.shape}")
    
    return voltage, calcium

def process_csv_and_extract(input_csv, output_dir='./'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    if 'filename' not in df.columns or 'roi' not in df.columns:
        raise ValueError("CSV must have 'filename' and 'roi' columns")
    
    print(f"Processing {len(df)} files...")
    
    successful = 0
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            filepath = row['filename']
            roi_str = row['roi']
            
            roi_str = str(roi_str).replace('np.int64', '').replace('(', '').replace(')', '')
            roi_parts = [int(x.strip()) for x in roi_str.split(',')]
            if len(roi_parts) != 4:
                raise ValueError(f"ROI must have 4 values, got {len(roi_parts)}")
            roi = tuple(roi_parts)
            
            if not os.path.exists(filepath):
                alt_path = filepath.lstrip('.\\').lstrip('./')
                if os.path.exists(alt_path):
                    filepath = alt_path
                else:
                    raise FileNotFoundError(f"Cannot find {filepath}")
            
            tif_stack = imread(filepath)
            voltage, calcium = extract_roi_timeseries(tif_stack, roi)
            
            basename = clean_basename(filepath)
            voltage_file = os.path.join(output_dir, f"timeseries_{basename}_voltage.npy")
            calcium_file = os.path.join(output_dir, f"timeseries_{basename}_calcium.npy")
            
            np.save(voltage_file, voltage)
            np.save(calcium_file, calcium)
            
            successful += 1
            
        except Exception as e:
            failed.append((row['filename'], str(e)))
            print(f"\nError processing {row['filename']}: {e}")
    
    print(f"\n✓ Successfully extracted {successful}/{len(df)} files")
    
    if failed:
        print(f"\n✗ Failed to process {len(failed)} files:")
        for fname, error in failed[:10]:
            print(f"  {fname}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")
    
    return successful, failed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="./")
    args = parser.parse_args()
    
    successful, failed = process_csv_and_extract(args.input_csv, args.output_dir)
    print(f"\nDone! Corrected timeseries files saved to: {args.output_dir}")
