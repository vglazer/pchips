"""Time Pchips against SciPy's PchipInterpolator, for the Performance section of the README.

Construction is timed for each case, with Pchips' default settings and with the quartic
approximation and the M4 constraint. Evaluation is timed separately, at 100,000 points: both
interpolants are CubicHermiteSplines, so it should take the same time. Each time is the best
of several runs.

Usage: uv run python scripts/benchmark_scipy.py [--repeats N]
"""

import argparse
import platform
import timeit

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator as ScipyPchip

from pchips import PchipInterpolator

# (label, number of points, number of columns)
CASES = [
    ('20 points', 20, 1),
    ('1,000 points', 1_000, 1),
    ('100,000 points', 100_000, 1),
    ('50 points × 1,000 columns', 50, 1_000),
]
EVAL_POINTS = 100_000


def best_of(f, repeats):
    return min(timeit.repeat(f, number=1, repeat=repeats))


def ms(t):
    return f'{t * 1e3:.2g}'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    print(f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, best of {args.repeats} runs')

    rng = np.random.default_rng(0)
    print('\nConstruction:')
    for label, n, m in CASES:
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x[:, None] + rng.uniform(0, 2 * np.pi, m))
        if m == 1:
            y = y[:, 0]
        ts = best_of(lambda: ScipyPchip(x, y), args.repeats)
        td = best_of(lambda: PchipInterpolator(x, y), args.repeats)
        tq = best_of(lambda: PchipInterpolator(x, y, approx_order='quartic',
                                               mono_constraint='M4'), args.repeats)
        print(f'{label:>26}: SciPy {ms(ts)} ms, Pchips default {ms(td)} ms '
              f'({td / ts:.1f}x slower), quartic and M4 {ms(tq)} ms ({tq / ts:.1f}x slower)')

    x = np.linspace(0, 2 * np.pi, 1_000)
    y = np.sin(x)
    u = np.sort(rng.uniform(0, 2 * np.pi, EVAL_POINTS))
    s, p = ScipyPchip(x, y), PchipInterpolator(x, y)
    print(f'\nEvaluation at {EVAL_POINTS:,} points of a 1,000-point interpolant:')
    for label, f in [('values', lambda i: i(u)), ('derivatives', lambda i: i(u, nu=1))]:
        print(f'{label:>26}: SciPy {ms(best_of(lambda: f(s), args.repeats))} ms, '
              f'Pchips {ms(best_of(lambda: f(p), args.repeats))} ms')


if __name__ == '__main__':
    main()
