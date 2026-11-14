"""Repository-wide pytest configuration.

Some tests check the training data itself (the row manifest, the held-out prompt sets). That data is not in
this repository (see DATA.md), so those tests are skipped here with a reason, and run again once the data
has been rebuilt locally. A test is skipped only when the file it could not open is one of the files this
repository is known not to ship; any other missing file is still a failure.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
NOT_SHIPPED_DIRS = {"data", "models", "results", "logs"}
NOT_SHIPPED_FILES = {
    "coherence_pilot/hh_helpful_test.json",
    "coherence_pilot/prompts.json",
    "sw_program/prompts.json",
    "s15_susceptibility/splits.json",
}


def _skip_if_not_shipped(err: FileNotFoundError) -> None:
    if not err.filename:
        return
    try:
        rel = Path(err.filename).resolve().relative_to(ROOT)
    except ValueError:
        return
    if rel.parts[0] in NOT_SHIPPED_DIRS or rel.as_posix() in NOT_SHIPPED_FILES:
        pytest.skip(f"needs {rel.as_posix()}, which is not in the repository (see DATA.md)")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except FileNotFoundError as err:
        _skip_if_not_shipped(err)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except FileNotFoundError as err:
        _skip_if_not_shipped(err)
        raise
