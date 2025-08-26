import numpy as np
import pytest
import matplotlib.pyplot as plt
#from pchips import PchipInterpolator
import pchips
import scipy.interpolate as interpolate
from pathlib import Path

CONFIGURATIONS = [
    {'approx_order': 'cubic', 'mono_constraint': 'M4'},
    {'approx_order': 'cubic', 'mono_constraint': 'M3'},
    {'approx_order': 'quartic', 'mono_constraint': 'M4'},
    {'approx_order': 'quartic', 'mono_constraint': 'M3'},
]

@pytest.fixture(scope="module")
def moler_data():
    data_path = Path(__file__).parent.joinpath("../data").resolve().joinpath("moler_data.csv")
    data = np.genfromtxt(data_path, delimiter=',', names=True)
    return data['x'], data['y']

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_moler_vs_scipy(moler_data, config):
    x, y = moler_data
    eval_points = np.linspace(x.min(), x.max(), 1001)

    # Pchips implementation
    interp = pchips.PchipInterpolator(x, y, **config)
    ported_results = interp(eval_points)

    # SciPy (Pchip) implementation
    scipy_interp = interpolate.PchipInterpolator(x, y)
    scipy_results = scipy_interp(eval_points)

    # Comparison
    np.testing.assert_allclose(ported_results, scipy_results, rtol=1e-1, atol=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(eval_points, ported_results, '-', label=f'Pchips {config}')
    plt.plot(eval_points, scipy_results, '--', label='SciPy (Pchip)')
    plt.title('Moler Dataset Comparison')
    plt.legend()
    plt.grid(True)

    plots_dir = Path(__file__).parent.joinpath("../plots").resolve()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_file = f'Moler_vs_scipy_{config["approx_order"]}_{config["mono_constraint"]}.pdf'
    plot_path = plots_dir.joinpath(plot_file)
    plt.savefig(plot_path)
    
    plt.close()
