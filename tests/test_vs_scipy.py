import numpy as np
import pytest
import matplotlib.pyplot as plt
import pchips
import scipy.interpolate as interpolate
from pathlib import Path

CONFIGURATIONS = [
    {'approx_order': 'cubic', 'mono_constraint': 'M4'},
    {'approx_order': 'cubic', 'mono_constraint': 'M3'},
    {'approx_order': 'quartic', 'mono_constraint': 'M4'},
    {'approx_order': 'quartic', 'mono_constraint': 'M3'},
]

M3_CONFIGURATIONS = [c for c in CONFIGURATIONS if c['mono_constraint'] == 'M3']
M4_CONFIGURATIONS = [c for c in CONFIGURATIONS if c['mono_constraint'] == 'M4']

SMOOTH_FUNCTIONS = {
    'sin': (np.sin, 0, 2*np.pi),
    'gaussian': (lambda x: np.exp(-x**2), -3, 3),
    'tanh': (np.tanh, -3, 3),
}

GRID_SIZES = [11, 21, 41, 81, 161, 321]

def random_knots(rng, n):
    # Non-uniform knots with spacing bounded away from zero
    return np.cumsum(rng.uniform(0.1, 1.0, n))

def random_datasets(seed, count=50):
    rng = np.random.default_rng(seed)
    return [(random_knots(rng, n), rng) for n in rng.integers(5, 30, count)]

def monotone_datasets(seed, count=50):
    datasets = []
    for x, rng in random_datasets(seed, count):
        y = np.cumsum(rng.exponential(1.0, len(x)))
        datasets.append((x, y))
    return datasets

def is_nondecreasing(v, tol=1e-12):
    return np.all(np.diff(v) >= -tol * max(1.0, np.max(np.abs(v))))

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_interpolates_data(config):
    for x, rng in random_datasets(seed=0):
        y = rng.normal(size=len(x))
        interp = pchips.PchipInterpolator(x, y, **config)
        scipy_interp = interpolate.PchipInterpolator(x, y)
        np.testing.assert_allclose(interp(x), y, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scipy_interp(x), y, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("config", M3_CONFIGURATIONS)
def test_monotone_data_preserved(config):
    for x, y in monotone_datasets(seed=1):
        eval_points = np.linspace(x.min(), x.max(), 2001)
        interp = pchips.PchipInterpolator(x, y, **config)
        scipy_interp = interpolate.PchipInterpolator(x, y)
        assert is_nondecreasing(scipy_interp(eval_points))
        assert is_nondecreasing(interp(eval_points))

def minmod(a, b):
    return (np.sign(a) + np.sign(b)) / 2 * np.minimum(np.abs(a), np.abs(b))

def is_q_monotone(x, y, i):
    # Huynh (4.15): the slopes (4.8a,b) of the cubic on [x_i, x_i+1] lie in I[0, 3s]
    s = np.diff(y) / np.diff(x)
    d = np.diff(s) / (x[2:] - x[:-2])   # d[j] is centered at x[j+1]
    e3 = np.diff(d) / (x[3:] - x[:-3])  # e3[j] is centered between x[j+1] and x[j+2]
    e = minmod(e3[:-1], e3[1:])         # e[j] is centered at x[j+2]
    h = x[i+1] - x[i]
    left = s[i] - h*minmod(d[i-1] + e[i-2]*(x[i] - x[i-1]), d[i] + e[i-1]*(x[i] - x[i+2]))
    right = s[i] + h*minmod(d[i-1] + e[i-2]*(x[i+1] - x[i-1]), d[i] + e[i-1]*(x[i+1] - x[i+2]))
    return all(0 <= q / s[i] <= 3 for q in (left, right))

def is_monotone_in(interp, x, y, i):
    v = np.sign(y[i+1] - y[i]) * interp(np.linspace(x[i], x[i+1], 201))
    return is_nondecreasing(v)

@pytest.mark.parametrize("config", M4_CONFIGURATIONS)
def test_q_monotone_data_preserved_m4(config):
    # M4 reproduces cubics, so it cannot preserve monotonicity in general.
    # Huynh's Theorem 3 guarantees it in interior intervals where the data are q-monotone.
    checked = 0
    for x, rng in random_datasets(seed=4, count=200):
        if len(x) < 8:
            continue
        y = rng.normal(size=len(x))
        interp = pchips.PchipInterpolator(x, y, **config)
        for i in range(3, len(x) - 4):
            if is_q_monotone(x, y, i):
                checked += 1
                assert is_monotone_in(interp, x, y, i)
    assert checked > 100

@pytest.mark.parametrize("config", M4_CONFIGURATIONS)
def test_monotone_data_preserved_near_boundary_m4(config):
    # The MP constraint is enforced next to the endpoints (Huynh, section 5)
    for x, y in monotone_datasets(seed=1):
        interp = pchips.PchipInterpolator(x, y, **config)
        for i in (0, 1, len(x) - 3, len(x) - 2):
            assert is_monotone_in(interp, x, y, i)

def test_m4_ttilde_exact_for_cubics():
    # Both one-sided cubic slopes at x[k] reproduce f'(x[k]) when the data are on a cubic
    x = random_knots(np.random.default_rng(5), 10)
    y = x**3 - 4*x**2 + x
    interp = pchips.PchipInterpolator(x, y, mono_constraint='M4')
    s, ds, e, _ = interp._vecdiffs()
    emin = minmod(e[:-1], e[1:])
    h = np.diff(x)
    for k in range(3, len(x) - 3):
        np.testing.assert_allclose(interp._ttilde(k, s, ds, emin, h), 3*x[k]**2 - 8*x[k] + 1, rtol=1e-10)

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_reproduces_linear(config):
    x = random_knots(np.random.default_rng(2), 12)
    y = 3*x - 1
    eval_points = np.linspace(x.min(), x.max(), 1001)
    interp = pchips.PchipInterpolator(x, y, **config)
    scipy_interp = interpolate.PchipInterpolator(x, y)
    np.testing.assert_allclose(interp(eval_points), 3*eval_points - 1, atol=1e-12)
    np.testing.assert_allclose(scipy_interp(eval_points), 3*eval_points - 1, atol=1e-12)

@pytest.mark.parametrize("config", CONFIGURATIONS)
@pytest.mark.parametrize("degree", [2, 3])
def test_reproduces_monotone_polynomials(config, degree):
    # Pchips reproduces monotone quadratics and cubics exactly, SciPy does not
    x = random_knots(np.random.default_rng(3), 12)
    y = x**degree
    eval_points = np.linspace(x.min(), x.max(), 1001)
    true_results = eval_points**degree
    interp = pchips.PchipInterpolator(x, y, **config)
    scipy_interp = interpolate.PchipInterpolator(x, y)
    np.testing.assert_allclose(interp(eval_points), true_results, rtol=1e-12)
    assert np.max(np.abs(scipy_interp(eval_points) - true_results)) > 1e-3

def max_errors(make_interp, f, a, b):
    eval_points = np.linspace(a, b, 20001)
    errors = []
    for n in GRID_SIZES:
        x = np.linspace(a, b, n)
        errors.append(np.max(np.abs(make_interp(x, f(x))(eval_points) - f(eval_points))))
    return np.array(errors)

def observed_orders(errors):
    return np.log2(errors[:-1] / errors[1:])

@pytest.mark.parametrize("config", CONFIGURATIONS)
@pytest.mark.parametrize("name", SMOOTH_FUNCTIONS)
def test_convergence_vs_scipy(config, name):
    f, a, b = SMOOTH_FUNCTIONS[name]
    errors = max_errors(lambda x, y: pchips.PchipInterpolator(x, y, **config), f, a, b)
    scipy_errors = max_errors(interpolate.PchipInterpolator, f, a, b)

    # Pchips is uniformly fourth order accurate on smooth data,
    # SciPy degrades to second order near extrema
    assert np.all(observed_orders(errors)[-3:] > 3.5)
    if name != 'tanh':
        assert np.all(observed_orders(scipy_errors)[-3:] < 2.5)
    assert np.all(errors[-3:] < scipy_errors[-3:])

    # Plotting
    h = (b - a) / (np.array(GRID_SIZES) - 1)
    plt.figure(figsize=(10, 6))
    plt.loglog(h, errors, 'o-', label=f'Pchips {config}')
    plt.loglog(h, scipy_errors, 's--', label='SciPy (Pchip)')
    plt.title(f'Max error vs knot spacing ({name})')
    plt.xlabel('h')
    plt.ylabel('max |error|')
    plt.legend()
    plt.grid(True, which='both')
    plots_dir = Path(__file__).parent.joinpath("../plots").resolve()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_file = f'Convergence_{name}_{config["approx_order"]}_{config["mono_constraint"]}.pdf'
    plot_path = plots_dir.joinpath(plot_file)
    plt.savefig(plot_path)
    plt.close()

@pytest.mark.parametrize("u", [1.5, [1.5, 2.5], np.array(1.5), np.array([[1.5, 2.5], [3.5, 4.5]])],
                         ids=['scalar', 'list', '0d', '2d'])
def test_output_shape_matches_scipy(u):
    x = np.arange(8.0)
    y = np.sin(x)
    interp = pchips.PchipInterpolator(x, y)
    scipy_interp = interpolate.PchipInterpolator(x, y)
    assert np.shape(interp(u)) == np.shape(scipy_interp(u))
