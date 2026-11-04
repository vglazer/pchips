"""Generates the plots in the Gallery section of README.md: uv run python scripts/make_gallery.py"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from pathlib import Path

import pchips

OUT = Path(__file__).parent.joinpath("../assets").resolve()
DATA = Path(__file__).parent.joinpath("../data").resolve()

# Categorical slots in fixed order, plus neutral ink for data and the true function
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SERIES_3 = "#1baf7a"  # aqua
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "font.size": 10,
})


def samples(ax, x, y):
    ax.plot(x, y, "o", color=INK, markersize=5.5, zorder=5, label="Samples")


def true_function(ax, u, v, label="True function"):
    ax.plot(u, v, "--", color=INK_SECONDARY, linewidth=1.2, label=label)


def max_errors(make, f, a, b, sizes):
    u = np.linspace(a, b, 20001)
    errors = []
    for n in sizes:
        x = np.linspace(a, b, n)
        errors.append(np.max(np.abs(make(x, f(x))(u) - f(u))))
    return np.array(errors)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT.joinpath(name), dpi=110)
    plt.close(fig)
    print(f"wrote {name}")


def preserving_vs_imposing():
    x = np.linspace(0, 2*np.pi, 11)
    y = np.sin(x)
    u = np.linspace(0, 2*np.pi, 4001)
    scipy_values = interpolate.PchipInterpolator(x, y)(u)
    pchips_values = pchips.PchipInterpolator(x, y)(u)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for ax in (left, right):
        true_function(ax, u, np.sin(u))
        ax.plot(u, scipy_values, color=SERIES_1, label="SciPy")
        ax.plot(u, pchips_values, color=SERIES_2, label="Pchips")
        samples(ax, x, y)
        ax.set_xlabel("x")
    left.set_title("sin(x) from 11 samples")
    left.legend(loc="lower left")
    right.set_xlim(0.9, 2.25)
    right.set_ylim(0.85, 1.03)
    right.set_title("Zoomed in on the peak")
    print(f"  peak: SciPy {scipy_values.max():.3f}, Pchips {pchips_values.max():.3f}")
    print(f"  max error: SciPy {np.max(np.abs(scipy_values - np.sin(u))):.4f}, "
          f"Pchips {np.max(np.abs(pchips_values - np.sin(u))):.4f}")
    save(fig, "preserving_vs_imposing.png")


def convergence():
    sizes = np.array([11, 21, 41, 81, 161, 321, 641])
    scipy_errors = max_errors(interpolate.PchipInterpolator, np.sin, 0, 2*np.pi, sizes)
    pchips_errors = max_errors(pchips.PchipInterpolator, np.sin, 0, 2*np.pi, sizes)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(sizes, scipy_errors, "o-", color=SERIES_1, markersize=6, label="SciPy")
    ax.loglog(sizes, pchips_errors, "o-", color=SERIES_2, markersize=6, label="Pchips")
    ax.annotate("error ∝ h²", (sizes[3], scipy_errors[3]), textcoords="offset points",
                xytext=(0, 12), ha="center", color=INK_SECONDARY)
    ax.annotate("error ∝ h⁴", (sizes[3], pchips_errors[3]), textcoords="offset points",
                xytext=(14, -4), ha="left", color=INK_SECONDARY)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Maximum error")
    ax.set_title("Interpolating sin(x) on [0, 2π]")
    ax.legend()
    print(f"  at {sizes[-1]} samples: SciPy {scipy_errors[-1]:.1e}, Pchips {pchips_errors[-1]:.1e}")
    save(fig, "convergence.png")


def monotone_data():
    data = np.genfromtxt(DATA.joinpath("akima_data.csv"), delimiter=",", names=True)
    x, y = data["x"], data["y"]
    u = np.linspace(x.min(), x.max(), 4001)
    spline = interpolate.CubicSpline(x, y)(u)
    scipy_values = interpolate.PchipInterpolator(x, y)(u)
    pchips_values = pchips.PchipInterpolator(x, y)(u)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(u, spline, color=SERIES_3, label="Cubic spline")
    ax.plot(u, scipy_values, color=SERIES_1, label="SciPy PCHIP")
    ax.plot(u, pchips_values, color=SERIES_2, label="Pchips")
    samples(ax, x, y)
    ax.set_xlabel("x")
    ax.set_title("Akima's monotone data")
    ax.legend(loc="upper left")
    for name, v in [("cubic spline", spline), ("SciPy", scipy_values), ("Pchips", pchips_values)]:
        print(f"  {name}: monotone {bool(np.all(np.diff(v) >= -1e-12))}, range [{v.min():.2f}, {v.max():.2f}]")
    save(fig, "monotone_data.png")


def overshoot_on_rough_data():
    # The data from SciPy's test_overshoot
    rng = np.random.RandomState(1234)
    x = np.sort(rng.random(20))
    y = rng.random(20)
    u = np.linspace(x.min(), x.max(), 4001)
    scipy_values = interpolate.PchipInterpolator(x, y)(u)
    pchips_values = pchips.PchipInterpolator(x, y)(u)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(u, scipy_values, color=SERIES_1, label="SciPy")
    ax.plot(u, pchips_values, color=SERIES_2, label="Pchips")
    samples(ax, x, y)
    ax.axhspan(y.min(), y.max(), color=GRID, alpha=0.5, zorder=0, label="Range of the data")
    ax.set_xlabel("x")
    ax.set_title("20 random values (SciPy's test_overshoot data)")
    ax.legend(loc="upper right", ncols=2)
    print(f"  data range [{y.min():.2f}, {y.max():.2f}], "
          f"SciPy [{scipy_values.min():.2f}, {scipy_values.max():.2f}], "
          f"Pchips [{pchips_values.min():.2f}, {pchips_values.max():.2f}]")
    save(fig, "overshoot_on_rough_data.png")


def cubic_vs_quartic():
    f = lambda x: np.exp(-x**2)
    sizes = np.array([7, 11, 21, 41, 81, 161, 321, 641])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, color, make in [
        ("SciPy", SERIES_1, interpolate.PchipInterpolator),
        ("Pchips, cubic", SERIES_2, pchips.PchipInterpolator),
        ("Pchips, quartic", SERIES_3, lambda x, y: pchips.PchipInterpolator(x, y, approx_order="quartic")),
    ]:
        errors = max_errors(make, f, -3, 3, sizes)
        ax.loglog(sizes, errors, "o-", color=color, markersize=6, label=label)
        print(f"  {label}: " + ", ".join(f"{n}: {e:.1e}" for n, e in zip(sizes, errors)))
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Maximum error")
    ax.set_title("Interpolating exp(-x²) on [-3, 3]")
    ax.legend()
    save(fig, "cubic_vs_quartic.png")


def m3_vs_m4():
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    # M4 is more accurate at a flat inflection point
    f = lambda x: x - np.sin(x)
    x = np.linspace(-2, 3.1, 41)
    u = np.linspace(-2, 3.1, 8001)
    # Maximum error in each interval between samples; M3 is drawn on top since the two
    # coincide away from the flat inflection point at x = 0
    for label, color in [("M4", SERIES_3), ("M3", SERIES_2)]:
        interp = pchips.PchipInterpolator(x, f(x), mono_constraint=label)
        errors = [np.max(np.abs(interp(v) - f(v))) for v in
                  (np.linspace(x[i], x[i+1], 201) for i in range(len(x) - 1))]
        left.stairs(errors, x, color=color, linewidth=1.8, label=label, baseline=None)
        print(f"  x - sin(x): {label} max error {max(errors):.1e}")
    left.set_yscale("log")
    left.set_ylim(1e-7, 1e-4)
    left.annotate("flat inflection point", (0, 1.8e-5), textcoords="offset points",
                  xytext=(16, 8), color=INK_SECONDARY)
    left.annotate("identical elsewhere", (2.2, 6e-6), textcoords="offset points",
                  xytext=(0, 12), ha="center", color=INK_SECONDARY)
    left.set_xlabel("x")
    left.set_ylabel("Maximum error between samples")
    left.set_title("x - sin(x) from 41 samples")
    handles, labels = left.get_legend_handles_labels()
    left.legend(handles[::-1], labels[::-1], loc="upper left")

    # M4 can break monotonicity on rough monotone data
    y = np.array([0.534, 1.144, 1.229, 3.16, 3.205, 3.261, 4.597, 4.803, 6.408, 6.74])
    x = np.arange(len(y), dtype=float)
    u = np.linspace(x.min(), x.max(), 4001)
    for label, color in [("M3", SERIES_2), ("M4", SERIES_3)]:
        v = pchips.PchipInterpolator(x, y, mono_constraint=label)(u)
        right.plot(u, v, color=color, label=label)
        drop = np.max(np.maximum.accumulate(v) - v)
        print(f"  monotone data: {label} largest decrease {drop:.3f}")
    samples(right, x, y)
    right.set_xlim(2.6, 5.4)
    right.set_ylim(2.9, 3.5)
    right.set_xlabel("x")
    right.set_title("Monotone data, zoomed in")
    right.legend(loc="upper left")
    save(fig, "m3_vs_m4.png")


DATASETS = {
    "akima": "Akima",
    "hyman": "Hyman",
    "moler": "Moler",
    "rpn14": "RPN 14",
    "titanium": "Titanium",
}


def datasets():
    for name, title in DATASETS.items():
        data = np.genfromtxt(DATA.joinpath(f"{name}_data.csv"), delimiter=",", names=True)
        x, y = data["x"], data["y"]
        u = np.linspace(x.min(), x.max(), 4001)
        series = [
            ("SciPy", SERIES_1, interpolate.PchipInterpolator(x, y)(u)),
            ("Pchips", SERIES_2, pchips.PchipInterpolator(x, y)(u)),
        ]
        if name == "akima":
            series.append(("Pchips, quartic", SERIES_3,
                           pchips.PchipInterpolator(x, y, approx_order="quartic")(u)))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if name == "hyman":
            true_function(ax, u, np.exp(-u**2), label="True function exp(-x²)")
        for label, color, values in series:
            ax.plot(u, values, color=color, label=label)
        samples(ax, x, y)
        ax.set_xlabel("x")
        ax.set_title(f"{title} data")
        ax.legend(loc="best")

        summary = ", ".join(f"{label} [{values.min():.3g}, {values.max():.3g}]" for label, _, values in series)
        difference = np.max(np.abs(series[1][2] - series[0][2]))
        print(f"  {title}: data [{y.min():.3g}, {y.max():.3g}], {summary}, max |Pchips - SciPy| {difference:.2g}")
        if name == "hyman":
            for label, _, values in series:
                print(f"    {label} max error {np.max(np.abs(values - np.exp(-u**2))):.3f}")
        save(fig, f"dataset_{name}.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    preserving_vs_imposing()
    convergence()
    monotone_data()
    overshoot_on_rough_data()
    cubic_vs_quartic()
    m3_vs_m4()
    datasets()
