from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

PY = Path('.venv/bin/python')
SCRIPT = Path('scripts/report_trend_variants.py')
OUTDIR = Path('reports/trend_based/ablation/variants')
OUTDIR.mkdir(parents=True, exist_ok=True)

# Keep p3-end fixed as requested
P3_END = '2026-02-13'

TOGGLES = [
    ('ema_sep', '--no-ema-sep'),
    ('nochop', '--no-nochop'),
    ('base_slope', '--no-base-slope'),
    ('corr', '--no-corr'),
]


def main() -> None:
    rows = []
    for mask in itertools.product([0, 1], repeat=len(TOGGLES)):
        disabled = [TOGGLES[i][0] for i, bit in enumerate(mask) if bit]
        flags = [TOGGLES[i][1] for i, bit in enumerate(mask) if bit]

        name = 'best_trend'
        if disabled:
            name += '__no-' + '__no-'.join(disabled)
        else:
            name += '__all_on'
        out_name = f'{name}.html'

        cmd = [str(PY), str(SCRIPT), '--out-dir', str(OUTDIR), '--out-name', out_name, '--p3-end', P3_END] + flags
        print('RUN', out_name, '::', ' '.join(flags) if flags else '(none)')
        subprocess.run(cmd, check=True)
        rows.append((out_name, disabled))

    print(f"Done. Wrote {len(rows)} variants to {OUTDIR}")


if __name__ == '__main__':
    main()
