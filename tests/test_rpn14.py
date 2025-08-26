
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
def rpn14_data():
    data_path = Path(__file__).parent.joinpath("../data").resolve().joinpath("rpn14_data.csv")
    data = np.genfromtxt(data_path, delimiter=',', names=True)
    return data['x'], data['y']

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_rpn14_vs_scipy(rpn14_data, config):
    x, y = rpn14_data
    eval_points = np.linspace(x.min(), x.max(), 1001)

    interp = PchipInterpolator(x, y, **config)
    ported_results = interp(eval_points)

    scipy_interp = ScipyPchipInterpolator(x, y)
    scipy_results = scipy_interp(eval_points)

    np.testing.assert_allclose(ported_results, scipy_results, rtol=1e-1, atol=0.1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(eval_points, ported_results, '-', label=f'Pchips {config}')
    plt.plot(eval_points, scipy_results, '--', label='SciPy (Pchip)')
    plt.title('RPN14 Dataset Comparison')
    plt.legend()
    plt.grid(True)
    plots_dir = Path(__file__).parent.joinpath("../plots").resolve()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_file = f'RPN14_vs_scipy_{config["approx_order"]}_{config["mono_constraint"]}.pdf'
    plot_path = plots_dir.joinpath(plot_file)
    plt.savefig(plot_path)
    plt.close()
