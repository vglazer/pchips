# Pchips

![Moler data comparison](assets/Moler_data_pchip_vs_pchips.png)
![Hyman data: Pchip vs True](assets/Hyman_scipy_pchip_vs_true.png)
![Hyman data: Pchips vs True](assets/Hyman_pchips_vs_true.png)

## Overview

This is a Python port, produced with the assistance of [Gemini CLI](https://github.com/google-gemini/gemini-cli) in an afternoon, of the monotonicity-preserving interpolant [originally implemented in MATLAB here](https://github.com/vglazer/USRA/tree/master/interpolation). It's meant to serve as a drop-in replacement for SciPy's [PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html). 

All the logic is in [interpolate.py](src/pchips/interpolate.py). The main point of the [tests](tests) is to generate plots.

The algorithm, due to [H. T. Huynh](https://scholar.google.com/citations?user=ZXhGCtwAAAAJ&hl=en), is described in this [NASA Technical Memo](https://ntrs.nasa.gov/citations/19910011517), which is self-contained and quite readable. A nearly identical version was later published in the [SIAM Journal on Numerical Analysis](https://epubs.siam.org/doi/10.1137/0730004).

## Quickstart

- Make sure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/)
- Clone repo
- cd into repo dir
- `uv sync`
- `uv run pytest`
- Check out the plots this will generate in the `plots` subdirectory

## Mathematical background

- Say that you have a bunch of discrete samples $\{(x_i, y_i)\}_{i = 0}^{n}$ which you want to interpolate on (i.e. match exactly) using some nice function $f: \mathbb{R} \rightarrow \mathbb{R}$ such that $f(x_i) = y_i, 0 \leq i \leq n$
- [Cubic splines](https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation) are 4th order accurate (i.e. the error term is $O(h^4)$ ) and quite smooth:
  - Their first derivative $f'$ is not only continuous, but also differentiable (i.e. $f \in C^2$)
  - However, cubic splines may "wiggle" by "overshooting" and "undershooting" the data. This is not "visually pleasing" and may be problematic for some applications
- Another approach is to construct a [Hermite spline](https://en.wikipedia.org/wiki/Cubic_Hermite_spline), which will match not only the data but also its first derivative:
  - That is, we also have $f'(x_i) = \hat{f'}(x_i), 0 \leq i \leq n$, where $\hat{f'}$ is the derivative of $x_i$, approximated using [Newton interpolation](https://en.wikipedia.org/wiki/Polynomial_interpolation#Newton_Interpolation), say
  - If $\hat{f'}$ is 3rd-order accurate or higher then $f$ is 4th-order accurate, as with cubic splines. You give up some smoothness, though: while $f'$ is still continuous, it is no longer differentiable (i.e. $f \in C^1$)
  - Moreover, **Hermite interpolants are not guaranteed to preserve monotonicity in the sense that they may be increasing in a section where "the data is decreasing" (suitably defined) and vice versa**
- A number approaches have been proposed for dealing with this, but they generally trade away accuracy in order to preserve monotonicity. For example, [PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) uses the method suggested by [Fritsch and Butland](https://epubs.siam.org/doi/10.1137/0905021), which is only 2nd order accurate in general:
  - **Intuitively, the issue is that the interpolant _imposes_ monotonicty on the data rather than simply _preserving_ it**
  - This results in "slicing" near local maxima and minima, causing accuracy to degrade to $O(h^2)$ which is no better than linear interpolation
  - The "Monotone Convex" spline proposed by [Hagan and West](https://www.deriscope.com/docs/Hagan_West_curves_AMF.pdf) in the context of forward curve construction also suffers from this problem
  - The "modified Akima" or ["Makima" interpolant](https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/) which is implemented by SciPy's [Akima1DInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html#scipy.interpolate.Akima1DInterpolator) produces a spline which "appears natural" and avoids "slicing" for non-monotone data. However, as far as I can tell, it does not guarantee $O(h^4)$ accuracy. Here is Akima's [original paper](https://dl.acm.org/doi/pdf/10.1145/321607.321609).
- [H.T. Huynh's interpolant](https://ntrs.nasa.gov/api/citations/19910011517/downloads/19910011517.pdf), which is implemented here, effectively relaxes the monotonicty constraint near strict local extrema:
  - If you use a high enough-order approximation of $\hat{f'}$, this results in a uniformly $O(h^4)$ accurate interpolant - the best you can do with a cubic - which preserves monotonicity without imposing it
  - It's quite nice to look at as well
