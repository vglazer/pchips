# Pchips

## Overview

This is a Python port, produced with the help of [Gemini CLI](https://github.com/google-gemini/gemini-cli), of the monotonicity-preserving interpolant originally implemented [in MATLAB here](https://github.com/vglazer/USRA/tree/master/interpolation). It's meant to be used as a drop-in replacement for SciPy's [PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

The algorithm is due to [H. T. Huynh](https://scholar.google.com/citations?user=ZXhGCtwAAAAJ&hl=en) and is described in this [NASA Tech Memo](https://ntrs.nasa.gov/citations/19910011517) (here is the [SIAM Journal on Numerical Analysis version](https://epubs.siam.org/doi/10.1137/0730004)).

## Quickstart

- Make sure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/)
- Clone repo
- cd into repo dir
- `uv sync`
- `uv run pytest`
- Check out the plots this will generate in the `plots` subdirectory
