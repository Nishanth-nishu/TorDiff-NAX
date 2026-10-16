"""Shared pytest setup. Tests marked ``cluster`` need the TD environment (torch_geometric, the torus tables) or data
that live only on gnode118; they are skipped elsewhere and run there with ``pytest -m cluster``."""
import importlib.util

import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'cluster: needs the gnode118 TD environment or data (skipped locally)')


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec('torch_geometric') is not None:
        return
    skip = pytest.mark.skip(reason='cluster-only: torch_geometric / TD data not available here')
    for item in items:
        if 'cluster' in item.keywords:
            item.add_marker(skip)
