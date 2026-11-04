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
    s, ds, e, _ = interp._vecdiffs(y[:, None])
    emin = minmod(e[:-1], e[1:])
    h = np.diff(x)[:, None]
    k = np.arange(3, len(x) - 3)
    np.testing.assert_allclose(interp._ttilde(3, len(x) - 3, s, ds, emin, h)[:, 0], 3*x[k]**2 - 8*x[k] + 1, rtol=1e-10)

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

@pytest.fixture(scope="module")
def cubic_data():
    x = random_knots(np.random.default_rng(6), 12)
    return x, x**3

def test_is_cubic_hermite_spline_like_scipy(cubic_data):
    x, y = cubic_data
    interp = pchips.PchipInterpolator(x, y)
    assert isinstance(interp, interpolate.CubicHermiteSpline)
    assert isinstance(interpolate.PchipInterpolator(x, y), interpolate.CubicHermiteSpline)
    assert interp.c.shape == interpolate.PchipInterpolator(x, y).c.shape

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_derivative(cubic_data, config):
    x, y = cubic_data
    eval_points = np.linspace(x.min(), x.max(), 1001)
    interp = pchips.PchipInterpolator(x, y, **config)
    exact = [3*eval_points**2, 6*eval_points, np.full_like(eval_points, 6.0)]
    for nu in (1, 2, 3):
        np.testing.assert_allclose(interp.derivative(nu)(eval_points), exact[nu-1], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(interp(eval_points, nu=nu), interp.derivative(nu)(eval_points))

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_derivative_continuous_at_knots(config):
    # Both interpolants are C1
    x, rng = random_datasets(seed=7, count=1)[0]
    y = rng.normal(size=len(x))
    h = np.diff(x)[:-1]
    for interp in (pchips.PchipInterpolator(x, y, **config), interpolate.PchipInterpolator(x, y)):
        # Slope at x[i] of the polynomial on the left vs the one on the right
        c = interp.c
        left = c[2, :-1] + 2*c[1, :-1]*h + 3*c[0, :-1]*h**2
        right = c[2, 1:]
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_antiderivative_and_integrate(cubic_data, config):
    x, y = cubic_data
    a, b = x[0], x[-1]
    eval_points = np.linspace(a, b, 1001)
    for interp in (pchips.PchipInterpolator(x, y, **config), interpolate.PchipInterpolator(x, y)):
        antiderivative = interp.antiderivative()
        assert antiderivative(a) == 0
        np.testing.assert_allclose(antiderivative.derivative()(eval_points), interp(eval_points), atol=1e-9)
        np.testing.assert_allclose(interp.antiderivative(2).derivative(2)(eval_points), interp(eval_points), atol=1e-9)
        np.testing.assert_allclose(interp.integrate(a, b), antiderivative(b) - antiderivative(a))
    # Pchips reproduces the cubic, so its integral is exact
    interp = pchips.PchipInterpolator(x, y, **config)
    np.testing.assert_allclose(interp.integrate(a, b), (b**4 - a**4) / 4, rtol=1e-12)

def test_extrapolate(cubic_data):
    x, y = cubic_data
    outside = np.array([x[0] - 1, x[0], x[-1], x[-1] + 1])
    for cls in (pchips.PchipInterpolator, interpolate.PchipInterpolator):
        assert cls(x, y).extrapolate
        assert np.all(np.isfinite(cls(x, y)(outside)))
        no_extrapolation = cls(x, y, extrapolate=False)(outside)
        np.testing.assert_array_equal(np.isnan(no_extrapolation), [True, False, False, True])
        np.testing.assert_array_equal(np.isnan(cls(x, y)(outside, extrapolate=False)), [True, False, False, True])
    # Pchips reproduces the cubic, so it extrapolates it exactly too
    np.testing.assert_allclose(pchips.PchipInterpolator(x, y)(outside), outside**3, rtol=1e-12)

@pytest.mark.parametrize("config", CONFIGURATIONS)
@pytest.mark.parametrize("shape,axis", [((12, 3), 0), ((3, 12), 1), ((2, 12, 3), 1), ((2, 3, 12), -1)])
def test_axis(config, shape, axis):
    rng = np.random.default_rng(8)
    x = random_knots(rng, 12)
    y = rng.normal(size=shape)
    eval_points = np.linspace(x.min(), x.max(), 101)
    interp = pchips.PchipInterpolator(x, y, axis=axis, **config)
    results = interp(eval_points)
    assert results.shape == interpolate.PchipInterpolator(x, y, axis=axis)(eval_points).shape

    # Each column matches the corresponding 1D interpolant
    columns = np.moveaxis(y, axis, -1).reshape(-1, len(x))
    column_results = np.moveaxis(results, axis, -1).reshape(-1, len(eval_points))
    for column, column_result in zip(columns, column_results):
        np.testing.assert_allclose(column_result, pchips.PchipInterpolator(x, column, **config)(eval_points))

def test_array_like_inputs():
    x = np.arange(8.0)
    y = np.sin(x)
    eval_points = np.linspace(0, 7, 51)
    expected = pchips.PchipInterpolator(x, y)(eval_points)
    np.testing.assert_array_equal(pchips.PchipInterpolator(list(x), list(y))(eval_points), expected)
    np.testing.assert_array_equal(pchips.PchipInterpolator(np.arange(8), y)(eval_points), expected)

def test_unsorted_x_raises_like_scipy():
    x = np.arange(8.0)
    y = np.sin(x)
    idx = np.random.default_rng(9).permutation(len(x))
    for cls in (pchips.PchipInterpolator, interpolate.PchipInterpolator):
        with pytest.raises(ValueError, match="strictly increasing"):
            cls(x[idx], y[idx])

@pytest.mark.parametrize("config", M3_CONFIGURATIONS)
def test_solve_inverts_monotone_data(config):
    # Midpoints avoid knots where the derivative is clamped to zero,
    # since PPoly.solve can miss roots where the curve only touches the target value
    for x, y in monotone_datasets(seed=10, count=10):
        interp = pchips.PchipInterpolator(x, y, **config)
        for u in (x[:-1] + x[1:]) / 2:
            roots = interp.solve(interp(u), extrapolate=False)
            assert len(roots) == 1
            np.testing.assert_allclose(interp(roots[0]), interp(u), rtol=1e-12)

@pytest.mark.parametrize("x,y", [
    (np.arange(8.0), np.arange(7.0)),
    (np.arange(8.0).reshape(2, 4), np.arange(8.0)),
    (np.arange(1.0), np.arange(1.0)),
    (np.array([0.0, 1.0, np.inf, 3.0, 4.0]), np.arange(5.0)),
    (np.arange(5.0), np.array([0.0, 1.0, np.nan, 3.0, 4.0])),
], ids=['length mismatch', '2d x', 'one point', 'infinite x', 'nan y'])
def test_invalid_inputs_raise_like_scipy(x, y):
    for cls in (pchips.PchipInterpolator, interpolate.PchipInterpolator):
        with pytest.raises(ValueError):
            cls(x, y)

def test_complex_y_raises():
    # Recent SciPy versions raise too; older ones silently accept complex data
    with pytest.raises(ValueError, match="real values"):
        pchips.PchipInterpolator(np.arange(8.0), np.arange(8.0) + 1j)

@pytest.mark.parametrize("option", [{'approx_order': 'quintic'}, {'mono_constraint': 'M5'}])
def test_invalid_options(option):
    with pytest.raises(ValueError, match="Unsupported"):
        pchips.PchipInterpolator(np.arange(8.0), np.arange(8.0), **option)

def test_signature_matches_scipy():
    import inspect
    ours = inspect.signature(pchips.PchipInterpolator).parameters
    theirs = inspect.signature(interpolate.PchipInterpolator).parameters
    positional = [name for name, p in ours.items() if p.kind == p.POSITIONAL_OR_KEYWORD]
    assert positional == list(theirs)
    for name, p in theirs.items():
        assert ours[name].default == p.default
    assert ours['approx_order'].kind == ours['mono_constraint'].kind == inspect.Parameter.KEYWORD_ONLY
    assert pchips.pchip is pchips.PchipInterpolator
    assert list(inspect.signature(pchips.pchip_interpolate).parameters)[:5] == \
        list(inspect.signature(interpolate.pchip_interpolate).parameters)

def test_positional_axis_and_extrapolate():
    x = np.arange(8.0)
    y = np.vstack([np.sin(x), np.cos(x)])
    outside = np.array([-1.0, 3.5])
    for cls in (pchips.PchipInterpolator, interpolate.PchipInterpolator):
        values = cls(x, y, 1, False)(outside)
        assert values.shape == (2, 2)
        assert np.all(np.isnan(values[:, 0])) and np.all(np.isfinite(values[:, 1]))

@pytest.mark.parametrize("config", CONFIGURATIONS)
@pytest.mark.parametrize("n", [2, 3, 4])
def test_few_points_match_scipy(config, n):
    x = random_knots(np.random.default_rng(11), n)
    y = np.random.default_rng(12).normal(size=(n, 3))
    eval_points = np.linspace(x.min() - 1, x.max() + 1, 101)
    np.testing.assert_allclose(pchips.PchipInterpolator(x, y, **config)(eval_points),
                               interpolate.PchipInterpolator(x, y)(eval_points), rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_pchip_interpolate(config):
    x = random_knots(np.random.default_rng(13), 10)
    y = np.random.default_rng(14).normal(size=(3, 10))
    eval_points = np.linspace(x.min(), x.max(), 101)
    interp = pchips.PchipInterpolator(x, y, axis=1, **config)
    np.testing.assert_array_equal(pchips.pchip_interpolate(x, y, eval_points, axis=1, **config), interp(eval_points))
    np.testing.assert_array_equal(pchips.pchip_interpolate(x, y, eval_points, 2, 1, **config),
                                  interp.derivative(2)(eval_points))
    values, slopes = pchips.pchip_interpolate(x, y, eval_points, der=[0, 1], axis=1, **config)
    np.testing.assert_array_equal(values, interp(eval_points))
    np.testing.assert_array_equal(slopes, interp.derivative()(eval_points))

@pytest.mark.parametrize("approx_order", ['cubic', 'quartic'])
def test_m4_fourth_order_at_flat_inflection(approx_order):
    # x - sin(x) is monotone, and its slope vanishes at 0, which is not a knot.
    # M3 clips the derivatives there and loses accuracy, M4 stays fourth order.
    f, a, b = (lambda x: x - np.sin(x)), -2, 3.1
    eval_points = np.linspace(a, b, 40001)
    errors = {}
    for constraint in ('M3', 'M4'):
        errors[constraint] = []
        for n in [41, 81, 161, 321, 641]:
            x = np.linspace(a, b, n)
            interp = pchips.PchipInterpolator(x, f(x), approx_order=approx_order, mono_constraint=constraint)
            errors[constraint].append(np.max(np.abs(interp(eval_points) - f(eval_points))))
    assert np.all(observed_orders(np.array(errors['M4'])) > 3.5)
    assert errors['M3'][-1] > 3 * errors['M4'][-1]
