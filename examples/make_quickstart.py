"""Builds and executes examples/quickstart.ipynb, and saves the first plot of its comparison
with SciPy as the image at the top of README.md (run: uv run python examples/make_quickstart.py).

The notebook compares Pchips with SciPy's PchipInterpolator, and its options with each other.
The classic data sets from data/ are written into it, so that it runs on its own, with nothing
but pchips, SciPy and matplotlib installed.
"""

import base64
import csv
from pathlib import Path

import nbformat
import nbformat.sign
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
README = "https://github.com/vglazer/pchips#"
DATASETS = {"hyman": "Hyman", "moler": "Moler", "akima": "Akima", "rpn14": "RPN 14",
            "titanium": "Titanium"}


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


def data_cell():
    lines = ["DATA = {"]
    for name in DATASETS:
        with open(ROOT / "data" / f"{name}_data.csv") as f:
            rows = list(csv.DictReader(f))
        xs = ", ".join(r["x"] for r in rows)
        ys = ", ".join(r["y"] for r in rows)
        lines.append(f'    "{name}": (np.array([{xs}]),\n{" " * 14}np.array([{ys}])),')
    lines.append("}")
    return code("\n".join(lines))


cells = [
    md("""
# Pchips quickstart

[Pchips](https://github.com/vglazer/pchips) is a drop-in replacement for SciPy's
[PchipInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
that _preserves_ monotonicity rather than _imposing_ it, which keeps it fourth-order accurate at
peaks and troughs. This notebook shows how to use it, then compares it with SciPy side by side, and
its options with each other: the cubic and quartic derivative approximations, and the M3 and M4
monotonicity constraints.

It needs `pchips`, `scipy` and `matplotlib`. Every plot is drawn from scratch, so change the data or
the options and run the cells again.
"""),
    code("""
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as ScipyPchip

from pchips import PchipInterpolator

print("pchips", version("pchips"), "| scipy", version("scipy"))
"""),
    md("""
## Using it

`PchipInterpolator` takes the same arguments as SciPy's, plus two options: `approx_order`
(`"cubic"`, the default, or `"quartic"`) and `mono_constraint` (`"M3"`, the default, or `"M4"`).
The result is a SciPy `CubicHermiteSpline`, so it evaluates, differentiates, integrates and finds
roots the same way.
"""),
    code("""
x = np.linspace(0, 2*np.pi, 11)
interp = PchipInterpolator(x, np.sin(x))

print("value at pi/2:        ", interp(np.pi / 2))
print("derivative at 0:      ", interp(0, nu=1))
print("integral over [0, pi]:", interp.integrate(0, np.pi))
print("roots in [0, 2 pi]:   ", interp.roots(extrapolate=False))

quartic = PchipInterpolator(x, np.sin(x), approx_order="quartic", mono_constraint="M4")
print("quartic, M4 at pi/2:  ", quartic(np.pi / 2))
"""),
    md("""
## Comparing with SciPy

A few plotting helpers, and a consistent look. SciPy is always blue and Pchips' default always orange.
"""),
    code("""
SCIPY, PCHIPS, OTHER = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 1.8, "font.size": 10,
})


def samples(ax, x, y):
    ax.plot(x, y, "o", color=INK, markersize=5.5, zorder=5, label="Samples")


def true_function(ax, u, v, label="True function"):
    ax.plot(u, v, "--", color=INK_SECONDARY, linewidth=1.2, label=label)


def max_errors(make, f, a, b, sizes):
    \"\"\"Maximum error of make(x, f(x)) on [a, b], for each number of evenly spaced samples.\"\"\"
    u = np.linspace(a, b, 20001)
    return np.array([np.max(np.abs(make(x, f(x))(u) - f(u)))
                     for x in (np.linspace(a, b, n) for n in sizes)])
"""),
    md("""
### Preserving versus imposing monotonicity

#### Peaks and troughs

The peak of $\\sin(x)$ falls exactly between two samples with equal values, so the data are flat there.
SciPy's interpolant _imposes_ monotonicity by setting the derivative to zero at both samples, which cuts
the peak off flat at 0.951. Pchips only _preserves_ monotonicity where the data are monotone, so it
follows the curve up to 1.002, close to the true peak of 1. The maximum error is 0.049 for SciPy and
0.0055 for Pchips.
"""),
    code("""
x = np.linspace(0, 2*np.pi, 11)
y = np.sin(x)
u = np.linspace(0, 2*np.pi, 4001)
scipy_values = ScipyPchip(x, y)(u)
pchips_values = PchipInterpolator(x, y)(u)

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
for ax in (left, right):
    true_function(ax, u, np.sin(u))
    ax.plot(u, scipy_values, color=SCIPY, label="SciPy")
    ax.plot(u, pchips_values, color=PCHIPS, label="Pchips")
    samples(ax, x, y)
    ax.set_xlabel("x")
left.set_title("sin(x) from 11 samples")
left.legend(loc="lower left")
right.set_xlim(0.9, 2.25)
right.set_ylim(0.85, 1.03)
right.set_title("Zoomed in on the peak")
fig.tight_layout()
plt.show()

print(f"peak: SciPy {scipy_values.max():.3f}, Pchips {pchips_values.max():.3f}")
print(f"max error: SciPy {np.max(np.abs(scipy_values - np.sin(u))):.4f}, "
      f"Pchips {np.max(np.abs(pchips_values - np.sin(u))):.4f}")
"""),
    md("""
#### Accuracy as the samples get denser

Flattening every peak and trough limits SciPy's interpolant to $O(h^2)$ accuracy, while Pchips is
$O(h^4)$: doubling the number of samples cuts SciPy's error by a factor of 4, and Pchips' by 16. With
641 samples, the maximum error is 3.6e-6 for SciPy and 3.6e-10 for Pchips.
"""),
    code("""
sizes = np.array([11, 21, 41, 81, 161, 321, 641])
scipy_errors = max_errors(ScipyPchip, np.sin, 0, 2*np.pi, sizes)
pchips_errors = max_errors(PchipInterpolator, np.sin, 0, 2*np.pi, sizes)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(sizes, scipy_errors, "o-", color=SCIPY, markersize=6, label="SciPy")
ax.loglog(sizes, pchips_errors, "o-", color=PCHIPS, markersize=6, label="Pchips")
ax.annotate("error ∝ h²", (sizes[3], scipy_errors[3]), textcoords="offset points",
            xytext=(0, 12), ha="center", color=INK_SECONDARY)
ax.annotate("error ∝ h⁴", (sizes[3], pchips_errors[3]), textcoords="offset points",
            xytext=(14, -4), ha="left", color=INK_SECONDARY)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Maximum error")
ax.set_title("Interpolating sin(x) on [0, 2π]")
ax.legend()
plt.show()

print("error ratio per doubling: SciPy", np.round(scipy_errors[:-1] / scipy_errors[1:], 1))
print("                         Pchips", np.round(pchips_errors[:-1] / pchips_errors[1:], 1))
print(f"at {sizes[-1]} samples: SciPy {scipy_errors[-1]:.1e}, Pchips {pchips_errors[-1]:.1e}")
"""),
    md("""
The classic data sets used below, from the `data` directory of the repository:
"""),
    data_cell(),
    md("""
#### Monotone data stay monotone

On monotone data, both PCHIPs give a monotone interpolant with no overshoot, unlike a cubic spline,
which dips to 4.74 on data which never go below 10 and bulges above 60 between the samples at 50 and 60.
"""),
    code("""
x, y = DATA["akima"]
u = np.linspace(x.min(), x.max(), 4001)
curves = [("Cubic spline", OTHER, CubicSpline(x, y)(u)),
          ("SciPy PCHIP", SCIPY, ScipyPchip(x, y)(u)),
          ("Pchips", PCHIPS, PchipInterpolator(x, y)(u))]

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color, v in curves:
    ax.plot(u, v, color=color, label=label)
samples(ax, x, y)
ax.set_xlabel("x")
ax.set_title("Akima's monotone data")
ax.legend(loc="upper left")
plt.show()

for label, _, v in curves:
    print(f"{label}: monotone {bool(np.all(np.diff(v) >= -1e-12))}, range [{v.min():.2f}, {v.max():.2f}]")
"""),
    md(f"""
#### The price: overshoot on rough data

Relaxing the constraint near peaks and troughs means Pchips can overshoot the data there. On smooth
data the overshoot is small and usually closer to the truth, as for $\\sin(x)$ above. On rough data it
can be large: on these 20 random values in [0.08, 0.93], from SciPy's own `test_overshoot`, Pchips
reaches 1.43, while SciPy stays within the range of the data. See
[Reasons not to switch]({README}reasons-not-to-switch) in the README.
"""),
    code("""
rng = np.random.RandomState(1234)   # the data from SciPy's test_overshoot
x = np.sort(rng.random(20))
y = rng.random(20)
u = np.linspace(x.min(), x.max(), 4001)
scipy_values = ScipyPchip(x, y)(u)
pchips_values = PchipInterpolator(x, y)(u)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(u, scipy_values, color=SCIPY, label="SciPy")
ax.plot(u, pchips_values, color=PCHIPS, label="Pchips")
samples(ax, x, y)
ax.axhspan(y.min(), y.max(), color=GRID, alpha=0.5, zorder=0, label="Range of the data")
ax.set_xlabel("x")
ax.set_title("20 random values (SciPy's test_overshoot data)")
ax.legend(loc="upper right", ncols=2)
plt.show()

print(f"data [{y.min():.2f}, {y.max():.2f}], SciPy [{scipy_values.min():.2f}, {scipy_values.max():.2f}], "
      f"Pchips [{pchips_values.min():.2f}, {pchips_values.max():.2f}]")
"""),
    md("""
### Cubic versus quartic

Both derivative approximations give an $O(h^4)$ interpolant, but the quartic one has a smaller error
constant: with 641 samples of $e^{-x^2}$, the maximum error is 3.6e-9 with cubic and 2.5e-10 with
quartic (and 6.5e-6 with SciPy). With very few samples, however, quartic can be worse, since it relies
on the data being smooth over five neighbouring samples rather than four: with 11 samples, the error is
3.6e-2 with cubic and 5.2e-2 with quartic. Use quartic once the samples resolve the shape of the data.
"""),
    code("""
f = lambda x: np.exp(-x**2)
sizes = np.array([7, 11, 21, 41, 81, 161, 321, 641])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for label, color, make in [
    ("SciPy", SCIPY, ScipyPchip),
    ("Pchips, cubic", PCHIPS, PchipInterpolator),
    ("Pchips, quartic", OTHER, lambda x, y: PchipInterpolator(x, y, approx_order="quartic")),
]:
    errors = max_errors(make, f, -3, 3, sizes)
    ax.loglog(sizes, errors, "o-", color=color, markersize=6, label=label)
    print(f"{label:>15}: " + ", ".join(f"{n}: {e:.1e}" for n, e in zip(sizes, errors)))
ax.set_xlabel("Number of samples")
ax.set_ylabel("Maximum error")
ax.set_title("Interpolating exp(-x²) on [-3, 3]")
ax.legend()
plt.show()
"""),
    md("""
### M3 versus M4

**Left**: on smooth, well-sampled data, M3 and M4 give the same interpolant except near a flat
inflection point, where the slope of a monotone function drops to zero between samples, as for
$x - \\sin(x)$ at $x = 0$. There, M3 clips the derivatives to keep the interpolant monotone and loses
accuracy: with 41 samples, the maximum error is 1.8e-5 with M3 and 6.7e-6 with M4. SciPy, which imposes
monotonicity everywhere, is shown for comparison.

**Right**: the price of M4. To stay fourth-order accurate, M4 allows derivatives which can make the
interpolant non-monotone on rough monotone data. On this data set, which increases at every sample, the
M4 interpolant decreases by 0.11 between $x = 3.5$ and $x = 4.5$, while the M3 interpolant and SciPy's
stay monotone.
"""),
    code("""
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

# M4 is more accurate at a flat inflection point. Plot the maximum error in each interval between
# samples; M3 is drawn on top since the two coincide away from x = 0
f = lambda x: x - np.sin(x)
x = np.linspace(-2, 3.1, 41)
intervals = [np.linspace(x[i], x[i+1], 201) for i in range(len(x) - 1)]
for label, color, interp in [("SciPy", SCIPY, ScipyPchip(x, f(x))),
                             ("M4", OTHER, PchipInterpolator(x, f(x), mono_constraint="M4")),
                             ("M3", PCHIPS, PchipInterpolator(x, f(x), mono_constraint="M3"))]:
    errors = [np.max(np.abs(interp(v) - f(v))) for v in intervals]
    left.stairs(errors, x, color=color, linewidth=1.8, label=label, baseline=None)
    print(f"x - sin(x): {label} max error {max(errors):.1e}")
left.set_yscale("log")
left.annotate("flat inflection point", (0, 1.8e-5), textcoords="offset points",
              xytext=(16, 8), color=INK_SECONDARY)
left.set_xlabel("x")
left.set_ylabel("Maximum error between samples")
left.set_title("x - sin(x) from 41 samples")
handles, labels = left.get_legend_handles_labels()
left.legend(handles[::-1], labels[::-1], loc="lower left")

# M4 can break monotonicity on rough monotone data
y = np.array([0.534, 1.144, 1.229, 3.16, 3.205, 3.261, 4.597, 4.803, 6.408, 6.74])
x = np.arange(len(y), dtype=float)
u = np.linspace(x.min(), x.max(), 4001)
for label, color, interp in [("SciPy", SCIPY, ScipyPchip(x, y)),
                             ("M3", PCHIPS, PchipInterpolator(x, y, mono_constraint="M3")),
                             ("M4", OTHER, PchipInterpolator(x, y, mono_constraint="M4"))]:
    v = interp(u)
    right.plot(u, v, color=color, label=label)
    print(f"monotone data: {label} largest decrease {np.max(np.maximum.accumulate(v) - v):.3f}")
samples(right, x, y)
right.set_xlim(2.6, 5.4)
right.set_ylim(2.9, 3.5)
right.set_xlabel("x")
right.set_title("Monotone data, zoomed in")
right.legend(loc="upper left")
fig.tight_layout()
plt.show()
"""),
    md("""
### Classic data sets

The data sets loaded above, which the tests also use. Each plot compares SciPy with Pchips' default
settings (and, for Akima's data, the quartic approximation too).

- **Hyman**: nine samples of $e^{-x^2}$, with the peak between the samples at $\\pm 0.4$. SciPy cuts the
  peak off flat at 0.852, an error of 0.148, while Pchips reaches 0.988, an error of 0.012. Away from the
  peak, nine samples are too few for either interpolant to follow the curve closely, and Pchips is off by
  up to 0.128 at $x = 1.06$.
- **Moler**: rough, non-monotone data with plateaus and sharp turns. SciPy stays within the data, flat
  across the plateau at 1.2. Pchips treats the two samples at 1.2 as either side of a peak and rounds it
  off up to 1.57, then dips to -0.118 between the samples at 0.1 and 0. If the plateau and the zero floor
  are real features of the data, SciPy is the better choice here.
- **Akima**: monotone data with a sharp rise, one of the test cases in Huynh's paper. All three
  interpolants are monotone and stay within the data. SciPy and Pchips' default differ by at most 1.8, on
  the final rise from 60 to 85. The quartic approximation, which draws on more neighbouring samples,
  stands out between the samples at 50 and 60: it levels off close to 60 (59.4 at $x = 13.5$, against
  57.0 and 56.8), then rises more steeply to 85.
- **RPN 14**: monotone data which rise steeply and then level off, from Fritsch and Carlson's paper on
  monotone interpolation, and the other test case in Huynh's paper. SciPy and Pchips are both monotone
  and differ by at most 0.036, mostly near the bottom of the rise.
- **Titanium**: a sharp peak which happens to fall on a sample, so neither interpolant overshoots it, and
  both stay within the data. They differ by at most 0.12, at the foot of the peak. Pchips is flat for a
  moment at the sample at $x = 855$: its derivative estimate there came out with the wrong sign for the
  rising data, so the monotonicity constraint set it to zero.
"""),
    code(f"""
TITLES = {DATASETS!r}

for name, title in TITLES.items():
    x, y = DATA[name]
    u = np.linspace(x.min(), x.max(), 4001)
    curves = [("SciPy", SCIPY, ScipyPchip(x, y)(u)), ("Pchips", PCHIPS, PchipInterpolator(x, y)(u))]
    if name == "akima":
        curves.append(("Pchips, quartic", OTHER, PchipInterpolator(x, y, approx_order="quartic")(u)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if name == "hyman":
        true_function(ax, u, np.exp(-u**2), label="True function exp(-x²)")
    for label, color, v in curves:
        ax.plot(u, v, color=color, label=label)
    samples(ax, x, y)
    ax.set_xlabel("x")
    ax.set_title(f"{{title}} data")
    ax.legend(loc="best")
    plt.show()

    ranges = ", ".join(f"{{label}} [{{v.min():.3g}}, {{v.max():.3g}}]" for label, _, v in curves)
    print(f"{{title}}: data [{{y.min():.3g}}, {{y.max():.3g}}], {{ranges}}, "
          f"max |Pchips - SciPy| {{np.max(np.abs(curves[1][2] - curves[0][2])):.2g}}")
    if name == "hyman":
        for label, _, v in curves:
            print(f"  {{label}} max error {{np.max(np.abs(v - np.exp(-u**2))):.3f}}")
"""),
]


def main():
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}})
    NotebookClient(nb, timeout=600, kernel_name="python3").execute()
    # Sign the notebook, as `jupyter trust` would, so that its saved outputs display on this machine
    nbformat.sign.NotebookNotary().sign(nb)
    out = Path(__file__).with_name("quickstart.ipynb")
    nbformat.write(nb, out)
    print(out, f"{out.stat().st_size // 1024} KB")

    # The first plot, sin(x) from 11 samples, is also the image at the top of the README
    first = next(o for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs
                 if "image/png" in o.get("data", {}))
    image = ROOT / "assets" / "preserving_vs_imposing.png"
    image.write_bytes(base64.b64decode(first["data"]["image/png"]))
    print(image)


if __name__ == "__main__":
    main()
