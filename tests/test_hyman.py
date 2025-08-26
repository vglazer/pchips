
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
def hyman_data():
    data_path = Path(__file__).parent.joinpath("../data").resolve().joinpath("hyman_data.csv")
    data = np.genfromtxt(data_path, delimiter=',', names=True)
    return data['x'], data['y']

def true_function(x):
    return np.exp(-x**2)

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_hyman_vs_scipy(hyman_data, config):
    x, y = hyman_data
    eval_points = np.linspace(x.min(), x.max(), 1001)

    # Pchips implementation
    interp = PchipInterpolator(x, y, **config)
    ported_results = interp(eval_points)

    # SciPy (Pchip) implementation
    scipy_interp = ScipyPchipInterpolator(x, y)
    scipy_results = scipy_interp(eval_points)

    # Comparison
    np.testing.assert_allclose(ported_results, scipy_results, rtol=1e-1, atol=0.2)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(eval_points, ported_results, '-', label=f'Pchips {config}')
    plt.plot(eval_points, scipy_results, '--', label='SciPy (Pchip)')
    plt.title('Hyman Dataset Comparison vs SciPy')
    plt.legend()
    plt.grid(True)
    plots_dir = Path(__file__).parent.joinpath("../plots").resolve()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_file = f'Hyman_vs_scipy_{config["approx_order"]}_{config["mono_constraint"]}.pdf'
    plot_path = plots_dir.joinpath(plot_file)
    plt.savefig(plot_path)
    plt.close()

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_hyman_vs_true(hyman_data, config):
    x, y = hyman_data
    eval_points = np.linspace(x.min(), x.max(), 1001)
    true_results = true_function(eval_points)

    # Pchips implementation
    interp = PchipInterpolator(x, y, **config)
    ported_results = interp(eval_points)

    # Comparison
    np.testing.assert_allclose(ported_results, true_results, rtol=1e-1, atol=0.1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(eval_points, ported_results, '-', label=f'Pchips {config}')
    plt.plot(eval_points, true_results, '--', label='True Function')
    plt.title('Hyman Dataset Pchips vs True Function')
    plt.legend()
    plt.grid(True)
    plots_dir = Path(__file__).parent.joinpath("../plots").resolve()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_file = f'Hyman_pchips_vs_true_{config["approx_order"]}_{config["mono_constraint"]}.pdf'
    plot_path = plots_dir.joinpath(plot_file)
    plt.savefig(plot_path)
    plt.close()

def test_scipy_vs_true(hyman_data):
    x, y = hyman_data
    eval_points = np.linspace(x.min(), x.max(), 1001)
    true_results = true_function(eval_points)

    # SciPy (Pchip) implementation
    scipy_interp = ScipyPchipInterpolator(x, y)
    scipy_results = scipy_interp(eval_points)

    # Comparison
    np.testing.assert_allclose(scipy_results, true_results, rtol=1e-1, atol=0.2)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(eval_points, scipy_results, '-', label='SciPy (Pchip)')
    plt.plot(eval_points, true_results, '--', label='True Function')
    plt.title('Hyman Dataset SciPy (Pchip) vs True Function')
    plt.legend()
    plt.grid(True)
    plots_dir = Path(__file__).parent.joinpath("../plots").resolve()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_file = 'Hyman_scipy_pchip_vs_true.pdf'
    plot_path = plots_dir.joinpath(plot_file)
    plt.savefig(plot_path)
    plt.close()
