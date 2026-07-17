#!/usr/bin/env python3
"""Aggregate yellow blink diagnostic reports into CSV/JSON and figures.

Usage:
    python scripts/aggregate_results.py --outputs outputs/analytical --report-dir reports
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

KEYS = [
    'result','frequency_hz','confidence','reason','fps','sample_count',
    'rising edges','falling edges','edge frequency','dominant frequency',
    'ratio slow 0.6-1.25Hz','ratio fast 1.35-2.25Hz','energy slow','energy fast',
    'track_quality_score','track_quality_flags'
]

def parse_report(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    row = {'dataset': path.parents[1].name, 'signal': path.parent.name}
    for key in KEYS:
        m = re.search(r'^' + re.escape(key) + r':\s*(.*)$', text, re.M)
        if m:
            row[key.replace(' ', '_').replace('.', '').replace('-', '_')] = m.group(1).strip()
    m = re.search(r'frames:\s*(\d+)\.\.(\d+)', text)
    if m:
        row['frame_start'], row['frame_end'] = int(m.group(1)), int(m.group(2))
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs/analytical')
    ap.add_argument('--report-dir', default='reports')
    args = ap.parse_args()
    outputs = Path(args.outputs)
    report_dir = Path(args.report_dir)
    fig_dir = report_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = [parse_report(p) for p in outputs.glob('*/*/*_diagnostic_report.txt')]
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ('dataset','signal','result','reason','track_quality_flags'):
            df[c] = pd.to_numeric(df[c], errors='ignore')
    if 'result' in df.columns:
        df['is_blinking'] = df['result'].fillna('').str.contains('blinking')
        df['blink_speed'] = df['result'].map({
            'blinking_yellow_slow':'slow',
            'blinking_yellow_fast':'fast',
            'steady_yellow':'not blinking',
        }).fillna('unknown')
    df.to_csv(report_dir / 'summary_table.csv', index=False)
    df.to_json(report_dir / 'summary_table.json', orient='records', indent=2, force_ascii=False)

    if not df.empty:
        plt.figure(figsize=(8,4.2))
        df.groupby(['dataset','result']).size().unstack(fill_value=0).plot(kind='bar', ax=plt.gca())
        plt.xlabel('dataset')
        plt.ylabel('number of evaluated signals')
        plt.title('Blinking classification counts')
        plt.tight_layout()
        plt.savefig(fig_dir / 'classification_counts.png', dpi=160)
        plt.close()

if __name__ == '__main__':
    main()
