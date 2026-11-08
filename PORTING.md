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
