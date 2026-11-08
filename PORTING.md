# Porting notes

Pchips is a Python port of the accurate monotone cubic interpolant
[originally implemented in MATLAB](https://github.com/vglazer/USRA/tree/master/interpolation). These
notes cover how the port was produced and how it is tested.

## How the port was produced

The first version was produced using [Gemini CLI](https://github.com/google-gemini/gemini-cli) in an
afternoon. It was then refined using [Claude Code](https://claude.com/claude-code):

- **Checked against Huynh's paper**, which turned up and fixed two bugs in the M4 constraint
  inherited from the MATLAB original
- **Turned into a drop-in replacement** for SciPy's
  [PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
- **Vectorized**
- **Given a more thorough test suite**

## Where things are

All the logic is in [interpolate.py](src/pchips/interpolate.py). The [tests](tests) compare it to
SciPy, check the guarantees from Huynh's paper and generate comparison plots.
[test_scipy_suite.py](tests/test_scipy_suite.py) runs SciPy's own `PchipInterpolator` tests against
it.

## Performance versus the original

The original's `pchint` computes the derivatives one sample at a time in loops, while Pchips
computes them all at once with NumPy. Pchips' `'cubic'` and `'quartic'` approximations are the
original's `appord = 32` and `44`.

**Only construction is timed**, that is, computing the derivatives and building the interpolant.
Evaluation isn't timed: on both sides it is done by the host library's own piecewise polynomial
code (`ppval` in Octave, `CubicHermiteSpline` in SciPy) rather than by ported code. The table
shows construction times for the same data as the README's Performance section: samples of
$\sin(x)$ on $[0, 10]$, and standard normal values on $[0, 1]$ for the multi-column case.
Each time is the best of 7 runs on an Apple M3 Max, with GNU Octave 11.3 for the original and
Python 3.13, NumPy 2.3 and SciPy 1.16 for Pchips. The speedup is in brackets:

| Data | Original (`32`, M3) | Pchips (default) | Original (`44`, M4) | Pchips (`approx_order='quartic'`, `mono_constraint='M4'`) |
| --- | --- | --- | --- | --- |
| 20&nbsp;points | 2.5&nbsp;ms | 0.08&nbsp;ms&nbsp;(32×) | 4.0&nbsp;ms | 0.11&nbsp;ms&nbsp;(38×) |
| 1,000&nbsp;points | 128&nbsp;ms | 0.10&nbsp;ms&nbsp;(1,284×) | 216&nbsp;ms | 0.14&nbsp;ms&nbsp;(1,490×) |
| 100,000&nbsp;points | 12.0&nbsp;s | 2.4&nbsp;ms&nbsp;(4,967×) | 20.8&nbsp;s | 4.0&nbsp;ms&nbsp;(5,252×) |
| 50&nbsp;points&nbsp;×&nbsp;1,000&nbsp;columns | 6.0&nbsp;s | 1.1&nbsp;ms&nbsp;(5,653×) | 10.2&nbsp;s | 1.7&nbsp;ms&nbsp;(6,053×) |

In every case the two sets of derivatives agree to within $3 \times 10^{-11}$, relative to the
largest derivative. The M4 bug fixes don't come into play on these data.

Octave runs loops in its interpreter. MATLAB compiles them just in time, so the original would
probably run faster there. These numbers measure it under Octave only.
[benchmark_usra.py](scripts/benchmark_usra.py) produces these numbers: run
`uv run python scripts/benchmark_usra.py`, with Octave on your `PATH` and the
[USRA repository on GitHub](https://github.com/vglazer/USRA/tree/master/interpolation), which holds
the original, cloned next to this one (or pass `--usra` with the path to its
`interpolation/hermite` directory).
