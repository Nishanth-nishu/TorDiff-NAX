"""The single gateway from flexitors to the frozen baseline (``torsional-diffusion/``) and to ``tools/``.

round5/DECISION.md graft 1 (plan_P1 §1.2): this is the only module that touches ``sys.path`` or the working directory
for baseline imports, and it pins the baseline files by SHA-256 so that any change to the frozen code is detected
(``flexitors/tests/test_baseline_frozen.py``; the pins of the evaluator, generator and sampler are the round-3 pins of
``tools/tests/test_leakfree.py``).

Two baseline quirks are handled here (plan_P1 §1.2 item 2):
* ``diffusion/torus.py:26-34`` loads ``.p.npy`` / ``.score.npy`` from the *current directory* and otherwise recomputes
  and writes them there. TD modules import it indirectly (``utils/utils.py:6`` -> ``diffusion/score_model.py:11``;
  ``utils/training.py:5``; ``train.py:5-9``; ``utils/boltzmann.py:1``), so EVERY baseline import, TD or tools, runs
  with the cwd set to the TD root, and imports of TD modules first assert that the tables exist (review P1 B1 /
  P2 S1). Their SHA-256 are pinned in ``TORUS_TABLE_PINS`` (checked on gnode118 by ``test_td_gateway.py``);
* TD's top-level packages are called ``diffusion`` and ``utils`` (and ``train`` is a script), so no flexitors module
  may use those names (checked by ``test_td_gateway.py``). After the first call ``TD_ROOT`` and ``TOOLS_ROOT`` stay on
  ``sys.path`` for the rest of the process, so a bare ``import utils`` then resolves to TD.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
TD_ROOT = Path(os.environ.get('FLEXITORS_TD_ROOT', REPO_ROOT / 'torsional-diffusion'))
TOOLS_ROOT = Path(os.environ.get('FLEXITORS_TOOLS_ROOT', REPO_ROOT / 'tools'))
RESERVED_NAMES = ('diffusion', 'utils', 'train')

# LF-normalised SHA-256 of every tracked TD python file at the start of Stage A (main @ 40a461f).
BASELINE_PINS = {
    'diffusion/likelihood.py': '7ca3b3375b6fb82fd0a39b1ba207ddd0fa1f4ad4ecdc0a9baa15a989a32e294c',
    'diffusion/sampling.py': '412bff7efb168b3f7c4c107d009dc0630b07e7be93b2c80774de2c1dfc302006',
    'diffusion/score_model.py': '6345a7c40e8887358b912251fc658289ecb6a4851f8a8436a019bdd745697d81',
    'diffusion/torus.py': '6ccd52b468af6df5718da4620f0c4fe43126d742a97d817cbb8b55968695f16d',
    'evaluate_confs.py': '67052cadccc365041c44c054add60865d8501d712b831548e9c06698ef09a3f4',
    'generate_confs.py': '706977cdd6fb5d7bc5da256ff070c5d7dc0a1b643331957ac79afa644e4df451',
    'optimize_confs.py': 'be8a43980030013e51b85281936020177822c600f1c6fdece23dd80212542c76',
    'standardize_confs.py': '9732b1d2aade38f17ad782ee5878846d7c200cc9344f3d2e062ef1a1364ff8c7',
    'test_boltzmann.py': '4579e5aa2b0fef1648f2c94f3d069b23bdcfc3f9cd839b117ca09b062dcce5f1',
    'train.py': '2b251556626695c1d570d6977951f9a0591576317d21972bd24b32928efcd519',
    'utils/boltzmann.py': 'a3bd98094799c87344b8611f0f69fb7cb880af7004356cf0c29e9a3edf1555ed',
    'utils/dataset.py': 'd8b943d61905316615a49829682cc393408a8c89e249dfac9cefce258abef6cb',
    'utils/featurization.py': '177c0fcddd93dc37418b347af70c34c622b457b0be7ad69f7e65fef042598bc1',
    'utils/parsing.py': '84b6d6630f910cfff1e326dcda05b6c8dd8b01878fa078cea9260d838434a0fb',
    'utils/standardization.py': '6069eae7c595a8d5b8b2506831cfbcd9429dde88cb2f3a2d4f0d7dff3e84d6b0',
    'utils/torsion.py': '47bfdc2cf25ef0e0fcceed2e0afd25b9e7ee2961ce46789fd91a211404e68c0f',
    'utils/training.py': '78333c36642ed831168705d7d4bf75c6d541033f65e3628eacb509530a239046',
    'utils/utils.py': '043ecc768ec92da7153b03123eead4de03a1a32444d018ef8f5acb1dc8f5b0cb',
    'utils/visualise.py': '670ce9d30c0ebb24c2ab9870235eabed0e30c2fcd88c55345b61e0a8af5ff6c2',
    'utils/xtb.py': '002fa88735ca0a4145afe9a01256a479636445fa89233af3f3b3afa4e05a1757',
}

# SHA-256 of the torus tables diffusion/torus.py reads from the TD root (gnode118 copies, 2026-10-01)
TORUS_TABLE_PINS = {'.p.npy': 'dbc17c9ed911e0e489ae36210f27b03e39dd60ed5dca3341ef1a65d15c3d4bea',
                    '.score.npy': 'a0ca1caa14fe16107f28325a7170a3442ff3e19f04ee5c238463c18e2bacce6c'}


def lf_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF (the repo stores LF; Windows checkouts may not)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def changed_baseline_files() -> list[str]:
    """Pinned baseline files whose content differs from the pin, or that are missing (empty list = intact)."""
    out = []
    for rel, pin in BASELINE_PINS.items():
        p = TD_ROOT / rel
        if not p.exists() or lf_sha256(p) != pin:
            out.append(rel)
    return out


def _is_td_module(name: str) -> bool:
    """True if the top-level name of ``name`` resolves inside the TD root (package directory or script)."""
    top = name.split('.')[0]
    return (TD_ROOT / top).is_dir() or (TD_ROOT / f'{top}.py').exists()


def import_baseline(name: str) -> ModuleType:
    """Import a module of the frozen baseline (e.g. ``'utils.training'``) or of ``tools/`` (e.g.
    ``'make_split_half'``) without modifying either.

    The import always runs with the cwd set to the TD root, so a direct or indirect import of ``diffusion.torus``
    reads the precomputed tables there instead of recomputing and writing them in the caller's directory.
    """
    for p in (str(TOOLS_ROOT), str(TD_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if _is_td_module(name):
        for table in TORUS_TABLE_PINS:
            if not (TD_ROOT / table).exists():
                raise FileNotFoundError(f'{TD_ROOT / table} missing: diffusion/torus.py would recompute it')
    old = os.getcwd()
    os.chdir(TD_ROOT)
    try:
        return importlib.import_module(name)
    finally:
        os.chdir(old)
