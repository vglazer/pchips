"""Time constructing Pchips against the original Octave implementation.

Only construction is timed. Evaluation is done by Octave's ppval and SciPy's PPoly code,
not by either implementation, so it says nothing about the port.

The original Octave implementation is pchint, in interpolation/hermite at
https://github.com/vglazer/USRA. Pchips' 'cubic' and 'quartic' derivative approximations are
its appord = 32 and 44 (Newton form). Both sides
build the interpolant from the same data, read from the same files, and the times are the
best of several runs, measured inside each process so that start-up is not counted. The data
are those of benchmark_scipy.py. As a check, the script also reports how far apart the two
sets of derivatives are, and fails if they differ by more than MAX_DIFFERENCE.

--quick runs only small cases, once each, to check that the script works.

Usage: uv run python scripts/benchmark_octave.py [--original PATH] [--repeats N] [--quick]
"""

import argparse
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

from pchips import PchipInterpolator

from benchmark_scipy import CASES, QUICK_CASES, best_of, dataset

ROOT = Path(__file__).resolve().parent.parent
# The largest relative difference in the derivatives that still counts as agreement
MAX_DIFFERENCE = 1e-8
# Where the original lives in a clone of its repository, next to this one
DEFAULT_ORIGINAL = ROOT.parent / 'USRA' / 'interpolation' / 'hermite'

# (label, Pchips options, MATLAB appord, MATLAB constraint)
SETTINGS = [
    ('default', {'approx_order': 'cubic', 'mono_constraint': 'M3'}, 32, 'M3'),
    ('quartic, M4', {'approx_order': 'quartic', 'mono_constraint': 'M4'}, 44, 'M4'),
]

OCTAVE = r"""
addpath('{original}/cubic'); addpath('{original}/Lib');
function y = rd(p, m, n)
  % NumPy writes row-major, so an (n, m) array reads in as an m-by-n matrix: one data set
  % per row, the layout pchint expects
  fid = fopen(p, 'r'); y = reshape(fread(fid, Inf, 'double'), m, n); fclose(fid);
end
x = rd('{dir}/x.bin', 1, {n}); y = rd('{dir}/y.bin', {m}, {n});
best = Inf;
for r = 1:{repeats}
  t = tic; pp = pchint(x, y, {appord}, 1, '{constr}'); best = min(best, toc(t));
end
fid = fopen('{dir}/d.bin', 'w'); fwrite(fid, pp.coefs(:, 3), 'double'); fclose(fid);
printf('%.9g\n', best);
"""


def octave_build(original, workdir, n, m, appord, constr, repeats):
    script = OCTAVE.format(original=original, dir=workdir, n=n, m=m, appord=appord,
                           constr=constr, repeats=repeats)
    out = subprocess.run(['octave', '--no-gui', '--quiet', '--eval', script],
                         capture_output=True, text=True, check=True).stdout
    best = float(out.strip().splitlines()[-1])
    # pp.coefs has one row per (interval, column), interval-major; column 3 holds the
    # derivatives at the left end of each interval
    d = np.fromfile(workdir / 'd.bin').reshape(n - 1, m)
    return best, d


def ms(t):
    return f'{t * 1e3:.2g}' if t < 1e-2 else f'{t * 1e3:,.0f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--original', type=Path, default=DEFAULT_ORIGINAL,
                        help="path to the original's interpolation/hermite directory")
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--quick', action='store_true', help='only small cases, run once each')
    args = parser.parse_args()
    cases, budget = (QUICK_CASES, 0) if args.quick else (CASES, 0.3)
    if args.quick:
        args.repeats = 1
    original = args.original.resolve()
    version = subprocess.run(['octave', '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    print(version)

    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for label, n, m in cases:
            x, y = dataset(n, m)
            x.tofile(workdir / 'x.bin')
            y.tofile(workdir / 'y.bin')
            for name, opts, appord, constr in SETTINGS:
                ob, od = octave_build(original, workdir, n, m, appord, constr, args.repeats)
                pb = best_of(partial(PchipInterpolator, x, y, **opts), args.repeats, budget)
                pd = PchipInterpolator(x, y, **opts).c[2].reshape(od.shape)
                diff = np.max(np.abs(pd - od)) / np.max(np.abs(od))
                print(f'{label:>26} {name:>12}: Octave {ms(ob)} ms, Pchips {ms(pb)} ms '
                      f'({ob / pb:,.0f}x faster), max relative difference in the '
                      f'derivatives {diff:.1e}', flush=True)
                worst = max(worst, diff)
    if worst > MAX_DIFFERENCE:
        sys.exit(f'The derivatives differ by up to {worst:.1e}, more than {MAX_DIFFERENCE:g}')


if __name__ == '__main__':
    main()
