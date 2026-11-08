"""Time constructing Pchips against the original MATLAB implementation, run under GNU Octave.

Only construction is timed. Evaluation is done by Octave's ppval and SciPy's PPoly code,
not by either implementation, so it says nothing about the port.

The MATLAB original is pchint in vglazer/USRA (interpolation/hermite). Pchips' 'cubic' and
'quartic' derivative approximations are its appord = 32 and 44 (Newton form). Both sides
build the interpolant from the same data, read from the same files, and the times are the
best of several runs, measured inside each process so that start-up is not counted. The data
are those of benchmark_scipy.py. As a check, the script also reports how far apart the two
sets of derivatives are.

Usage: uv run python scripts/benchmark_usra.py [--usra PATH] [--repeats N]
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pchips import PchipInterpolator

from benchmark_scipy import CASES, best_of, dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USRA = ROOT.parent / 'USRA' / 'interpolation' / 'hermite'

# (label, Pchips options, MATLAB appord, MATLAB constraint)
SETTINGS = [
    ('default', dict(approx_order='cubic', mono_constraint='M3'), 32, 'M3'),
    ('quartic, M4', dict(approx_order='quartic', mono_constraint='M4'), 44, 'M4'),
]

OCTAVE = r"""
addpath('{usra}/cubic'); addpath('{usra}/Lib');
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


def octave_build(usra, workdir, n, m, appord, constr, repeats):
    script = OCTAVE.format(usra=usra, dir=workdir, n=n, m=m, appord=appord,
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
    parser.add_argument('--usra', type=Path, default=DEFAULT_USRA,
                        help='path to USRA/interpolation/hermite')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    usra = args.usra.resolve()
    version = subprocess.run(['octave', '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    print(version)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for label, n, m in CASES:
            x, y = dataset(n, m)
            x.tofile(workdir / 'x.bin')
            y.tofile(workdir / 'y.bin')
            for name, opts, appord, constr in SETTINGS:
                ob, od = octave_build(usra, workdir, n, m, appord, constr, args.repeats)
                pb = best_of(lambda: PchipInterpolator(x, y, **opts), args.repeats)
                pd = PchipInterpolator(x, y, **opts).c[2].reshape(od.shape)
                diff = np.max(np.abs(pd - od)) / np.max(np.abs(od))
                print(f'{label:>26} {name:>12}: Octave {ms(ob)} ms, Pchips {ms(pb)} ms '
                      f'({ob / pb:,.0f}x faster), max relative difference in the '
                      f'derivatives {diff:.1e}', flush=True)


if __name__ == '__main__':
    main()
