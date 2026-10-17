"""The gateway imports baseline code without side effects (plan_P1 §4.1 test_td_gateway)."""
import ast
import hashlib
import os
import sys
from pathlib import Path

import pytest

from flexitors import td
from flexitors.td import RESERVED_NAMES, TD_ROOT, import_baseline

PKG = Path(__file__).resolve().parent.parent


def test_no_module_shadows_td_packages():
    names = {p.stem for p in PKG.rglob('*.py')} | {p.name for p in PKG.rglob('*') if p.is_dir()}
    assert not names & set(RESERVED_NAMES)


def test_only_td_py_touches_sys_path_or_cwd():
    for p in PKG.rglob('*.py'):
        if p.name == 'td.py' or 'tests' in p.parts:
            continue
        src = p.read_text()
        assert 'sys.path' not in src and 'os.chdir' not in src, p


def test_tools_import_keeps_cwd():
    cwd = os.getcwd()
    msh = import_baseline('make_split_half')
    assert os.getcwd() == cwd
    assert msh.split_indices(5) == ([0, 2, 4], [1, 3])


def test_every_baseline_import_runs_in_the_td_root(tmp_path, monkeypatch):
    """review P1 B1 / P2 S1: indirect imports of diffusion.torus must also see the TD root as cwd"""
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'probe_cwd_r5.py').write_text('import os\nCWD = os.getcwd()\n')
    monkeypatch.setattr(td, 'TOOLS_ROOT', tools)
    monkeypatch.chdir(tmp_path)
    probe = td.import_baseline('probe_cwd_r5')
    assert Path(probe.CWD).resolve() == TD_ROOT.resolve() and Path(os.getcwd()) == tmp_path


@pytest.mark.cluster
@pytest.mark.parametrize('name', ['diffusion.torus', 'utils.training', 'utils.utils', 'diffusion.sampling'])
def test_torus_tables_never_written_outside_td_root(tmp_path, monkeypatch, name):
    """fails if importing a TD module (directly or through another module) writes torus tables anywhere but the TD
    root, or changes the TD tables (pinned SHA-256 and mtimes)"""
    before = {t: (TD_ROOT / t).stat().st_mtime for t in td.TORUS_TABLE_PINS}
    monkeypatch.chdir(tmp_path)
    mod = import_baseline(name)
    assert os.getcwd() == str(tmp_path)
    assert not list(tmp_path.rglob('*.npy'))
    torus = sys.modules['diffusion.torus']
    assert Path(torus.__file__).resolve().parent.parent == TD_ROOT.resolve() and mod is not None
    assert {t: (TD_ROOT / t).stat().st_mtime for t in td.TORUS_TABLE_PINS} == before


@pytest.mark.cluster
def test_torus_table_pins():
    for t, pin in td.TORUS_TABLE_PINS.items():
        h = hashlib.sha256()
        with open(TD_ROOT / t, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        assert h.hexdigest() == pin, t


def test_gateway_module_imports_no_heavy_dependency():
    tree = ast.parse((PKG / 'td.py').read_text())
    mods = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not mods & {'torch', 'rdkit', 'numpy', 'torch_geometric'}
