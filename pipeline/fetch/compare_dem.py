#!/usr/bin/env python3
"""Compare a fetched grid with the quantized DEMS in an existing model (read-only)."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys


def compare(dem_path, model_path):
    dem = json.loads(Path(dem_path).read_text(encoding='utf-8'))
    html = Path(model_path).read_text(encoding='utf-8')
    encoded = re.search(r'const\s+DEMS\s*=\s*"([\d,\s-]+)"', html)
    frame = re.search(r'const\s+DN\s*=\s*(\d+)\s*,\s*DC\s*=\s*([\d.]+)', html)
    if not encoded or not frame or not re.search(r'v\s*=>\s*v\s*/\s*10\s*\+\s*380', html):
        raise ValueError('Unrecognized model DEM encoding/frame; expected DEMS, DN/DC and v/10+380')
    reference = [float(v) / 10 + 380 for v in encoded[1].split(',')]
    if (dem['rows'], dem['cols']) != (int(frame[1]), int(frame[1])) or abs(dem['cell_size_ft'] - float(frame[2])) > 1e-6:
        raise ValueError('Grid dimensions or cell spacing differ from model')
    if dem['row_order'] != 'north_to_south' or dem['column_order'] != 'west_to_east':
        raise ValueError('Grid axis order differs from model')
    values = dem['elevations_ft']
    if len(reference) != len(values) or len(values) != dem['rows'] * dem['cols']:
        raise ValueError('Grid lengths differ')
    errors = [abs(a - b) for a, b in zip(values, reference)]
    return {'cells': len(errors), 'median_absolute_difference_ft': statistics.median(errors),
            'max_absolute_difference_ft': max(errors),
            'mean_absolute_difference_ft': statistics.mean(errors),
            'within_1_ft': max(errors) <= 1,
            'note': 'Assumes the model and fetched grid share the same geographic center.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dem', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = compare(args.dem, args.model)
    text = json.dumps(result, indent=2) + '\n'
    print(text, end='')
    if args.report:
        args.report.write_text(text, encoding='utf-8')
    return 0 if result['within_1_ft'] else 1


if __name__ == '__main__':
    sys.exit(main())
