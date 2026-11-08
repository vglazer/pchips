"""Runs examples/quickstart.ipynb, to catch changes in pchips, SciPy or matplotlib that break it.

Only checks that every cell runs without error: the outputs vary with platform and library
versions. Skipped unless nbformat, nbclient, ipykernel and matplotlib are installed.
"""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "quickstart.ipynb"


def test_quickstart_runs(monkeypatch):
    # Draw plots without a display
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3").execute()
