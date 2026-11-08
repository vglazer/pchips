"""Time Pchips against SciPy's PchipInterpolator, for the Performance section of the README.

Construction is timed for each case, with Pchips' default settings and with the quartic
approximation and the M4 constraint, on samples of sin(x) on [0, 10] and, for the multi-column
case, standard normal values on [0, 1]. Evaluation is timed separately, at 100,000 points:
both interpolants are CubicHermiteSplines, so it should take the same time. Each time is the
best of several runs, each of which repeats the call enough times to take about 0.3 s.

--quick runs only small cases, once each, to check that the script works.

Usage: uv run python scripts/benchmark_scipy.py [--repeats N] [--quick]
"""

import argparse
import platform
import timeit
from functools import partial

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
# Small cases for --quick
QUICK_CASES = [
    ('20 points', 20, 1),
    ('50 points × 10 columns', 50, 10),
]
EVAL_POINTS = 100_000


def dataset(n, m):
    """Samples of sin(x) on [0, 10] for a single column, standard normal values on [0, 1] for
    several, as in the README's Performance section."""
    if m == 1:
        x = np.linspace(0, 10, n)
        return x, np.sin(x)
    return np.linspace(0, 1, n), np.random.default_rng(0).normal(size=(n, m))


def best_of(f, repeats, budget=0.3):
    number = max(1, int(budget / max(timeit.timeit(f, number=1), 1e-6)))
    return min(timeit.repeat(f, number=number, repeat=repeats)) / number


def ms(t):
    return f'{t * 1e3:.2g}'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--quick', action='store_true', help='only small cases, run once each')
    args = parser.parse_args()
    cases, budget = (QUICK_CASES, 0) if args.quick else (CASES, 0.3)
    if args.quick:
        args.repeats = 1
    print(f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, best of {args.repeats} runs')

    print('\nConstruction:')
    for label, n, m in cases:
        x, y = dataset(n, m)
        ts = best_of(partial(ScipyPchip, x, y), args.repeats, budget)
        td = best_of(partial(PchipInterpolator, x, y), args.repeats, budget)
        tq = best_of(partial(PchipInterpolator, x, y, approx_order='quartic',
                             mono_constraint='M4'), args.repeats, budget)
        print(f'{label:>26}: SciPy {ms(ts)} ms, Pchips default {ms(td)} ms '
              f'({td / ts:.1f}x slower), quartic and M4 {ms(tq)} ms ({tq / ts:.1f}x slower)')

    x, y = dataset(1_000, 1)
    u = np.linspace(0, 10, EVAL_POINTS)
    s, p = ScipyPchip(x, y), PchipInterpolator(x, y)
    print(f'\nEvaluation at {EVAL_POINTS:,} points of a 1,000-point interpolant:')
    for label, f in [('values', lambda i: i(u)), ('derivatives', lambda i: i(u, nu=1))]:
        print(f'{label:>26}: SciPy {ms(best_of(partial(f, s), args.repeats, budget))} ms, '
              f'Pchips {ms(best_of(partial(f, p), args.repeats, budget))} ms')


if __name__ == '__main__':
    main()
