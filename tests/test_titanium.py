
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pchips import PchipInterpolator
from scipy.interpolate import PchipInterpolator as ScipyPchipInterpolator
from pathlib import Path

CONFIGURATIONS = [
    {'approx_order': 'cubic', 'mono_constraint': 'M4'},
    {'approx_order': 'cubic', 'mono_constraint': 'M3'},
    {'approx_order': 'quartic', 'mono_constraint': 'M4'},
    {'approx_order': 'quartic', 'mono_constraint': 'M3'},
]

@pytest.fixture(scope="module")
def titanium_data():
    data = np.genfromtxt('../data/titanium_data.csv', delimiter=',', names=True)
    return data['x'], data['y']

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_titanium_vs_scipy(titanium_data, config):
    x, y = titanium_data
    eval_points = np.linspace(x.min(), x.max(), 1001)

    interp = PchipInterpolator(x, y, **config)
    ported_results = interp(eval_points)

    scipy_interp = ScipyPchipInterpolator(x, y)
    scipy_results = scipy_interp(eval_points)

    np.testing.assert_allclose(ported_results, scipy_results, rtol=1e-1, atol=0.2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(eval_points, ported_results, '-', label=f'PCHIPs {config}')
    plt.plot(eval_points, scipy_results, '--', label='SciPy')
    plt.title('Titanium Dataset Comparison')
    plt.legend()
    plt.grid(True)
    Path('../plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../plots/Titanium_vs_scipy_{config["approx_order"]}_{config["mono_constraint"]}.pdf')
    plt.close()
