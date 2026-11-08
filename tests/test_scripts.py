"""Smoke tests for the scripts in scripts/ and examples/."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'USRA' / 'interpolation' / 'hermite'


def run(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True,
                            text=True, encoding='utf-8', check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_benchmark_scipy_runs():
    out = run('scripts/benchmark_scipy.py', '--quick')
    assert '20 points' in out and 'Evaluation' in out


@pytest.mark.skipif(shutil.which('octave') is None or not ORIGINAL.is_dir(),
                    reason='needs Octave and a clone of the original next to this repository')
def test_benchmark_octave_runs_and_agrees():
    # The script fails if Pchips' derivatives differ from the original's
    out = run('scripts/benchmark_octave.py', '--quick')
    assert '20 points' in out


def test_quickstart_notebook_is_up_to_date():
    """The checked-in notebook has the cells make_quickstart.py builds: rerun it after editing."""
    nbformat = pytest.importorskip('nbformat')
    pytest.importorskip('nbclient')
    spec = importlib.util.spec_from_file_location('make_quickstart',
                                                  ROOT / 'examples' / 'make_quickstart.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)  # builds the cells; running them is left to main()
    notebook = nbformat.read(ROOT / 'examples' / 'quickstart.ipynb', as_version=4)
    assert [c.source for c in notebook.cells] == [c.source for c in builder.cells]
