"""The gateway imports baseline code without side effects (plan_P1 §4.1 test_td_gateway)."""
import ast
import os
from pathlib import Path

import pytest

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


@pytest.mark.cluster
def test_torus_import_uses_td_tables_and_keeps_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torus = import_baseline('diffusion.torus')
    assert os.getcwd() == str(tmp_path)
    assert not (tmp_path / '.p.npy').exists()  # tables were read from the TD root, not recomputed here
    assert Path(torus.__file__).resolve().parent.parent == TD_ROOT.resolve()


def test_gateway_module_imports_no_heavy_dependency():
    tree = ast.parse((PKG / 'td.py').read_text())
    mods = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not mods & {'torch', 'rdkit', 'numpy', 'torch_geometric'}
