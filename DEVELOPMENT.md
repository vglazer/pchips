# Development

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Then:

```sh
git clone https://github.com/vglazer/pchips
cd pchips
uv sync                 # create the environment, with the dev dependencies
uv run pytest           # run the tests
uv run ruff check .     # lint
```

The tests compare Pchips with SciPy, check the guarantees from Huynh's paper and run SciPy's own
`PchipInterpolator` tests against it, writing comparison plots to the `plots` directory along the
way. They also run the [quickstart notebook](examples/quickstart.ipynb) and smoke-test the scripts
below.

## Scripts

```sh
uv run python examples/make_quickstart.py  # rebuild and run the quickstart notebook, and redraw the README's top plot
uv run python scripts/benchmark_scipy.py   # time against SciPy (the README's Performance section)
uv run python scripts/benchmark_octave.py  # time against the original (PORTING.md)
```

- **The quickstart notebook** is generated from `examples/make_quickstart.py`, so edit that and
  rerun it rather than editing the notebook. A test fails if the notebook is out of date.
- **The benchmarks** take `--quick` to run only small cases, once each, which is what the tests do.
- **The Octave benchmark** needs [GNU Octave](https://octave.org) and a clone of the original's
  repository next to this one; see [PORTING.md](PORTING.md). Its test is skipped without them.

## Releasing

1. Bump `version` in [pyproject.toml](pyproject.toml), then commit and push.
2. [Create a release](https://github.com/vglazer/pchips/releases/new) on GitHub with a new tag `v`
   followed by that version, e.g. `v1.0.8`.
3. The [publish workflow](.github/workflows/publish.yml) then runs the tests, builds the package and
   publishes it to PyPI.

To rehearse a release, run the publish workflow by hand (Actions > Publish to PyPI > Run workflow).
It publishes a development release such as `1.0.8.dev7` to
[TestPyPI](https://test.pypi.org/project/pchips/) instead.
