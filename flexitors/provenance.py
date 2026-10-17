"""Provenance record written next to every flexitors output (plan_P2 V-X6; round2/FIXES.md X4).

On gnode118 the code arrives as an archive without git metadata, so the sync script exports the commit in
``FLEXITORS_COMMIT``; locally the commit and dirty flag come from git.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from flexitors.td import REPO_ROOT, changed_baseline_files


def _git(*args: str) -> str:
    try:
        return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def provenance() -> dict:
    """Commit, dirty flag, library versions, host, command line and baseline integrity of the current process."""
    import numpy
    import rdkit
    commit = os.environ.get('FLEXITORS_COMMIT') or _git('rev-parse', 'HEAD') or 'unknown'
    dirty = 'unknown' if 'FLEXITORS_COMMIT' in os.environ else bool(
        _git('status', '--porcelain', '--', 'flexitors', 'tools', 'experiments', 'round5/data'))
    return dict(commit=commit, dirty=dirty, python=sys.version.split()[0], rdkit=rdkit.__version__,
                numpy=numpy.__version__, host=platform.node(), argv=sys.argv,
                baseline_changed_files=changed_baseline_files())


def write_provenance(out_dir: Path) -> Path:
    """Write ``provenance.json`` into ``out_dir`` and return its path."""
    p = Path(out_dir) / 'provenance.json'
    p.write_text(json.dumps(provenance(), indent=1) + '\n')
    return p
