"""The frozen baseline is unchanged (round5/DECISION.md graft 1; plan_P1 §4.1 test_baseline_frozen)."""
import subprocess

from flexitors.td import BASELINE_PINS, REPO_ROOT, TD_ROOT, changed_baseline_files

# round-3 pins of tools/tests/test_leakfree.py (evaluator, generator, sampler): the gateway must agree with them
ROUND3_PINS = {'evaluate_confs.py': '67052cadccc365041c44c054add60865d8501d712b831548e9c06698ef09a3f4',
               'generate_confs.py': '706977cdd6fb5d7bc5da256ff070c5d7dc0a1b643331957ac79afa644e4df451',
               'diffusion/sampling.py': '412bff7efb168b3f7c4c107d009dc0630b07e7be93b2c80774de2c1dfc302006'}


def test_pins_agree_with_round3():
    for rel, pin in ROUND3_PINS.items():
        assert BASELINE_PINS[rel] == pin


def test_baseline_unchanged():
    assert changed_baseline_files() == []


def test_every_tracked_td_python_file_is_pinned():
    try:
        out = subprocess.run(['git', 'ls-files', 'torsional-diffusion'], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):  # cluster copy has no git metadata: compare the directory
        out = ['torsional-diffusion/' + p.relative_to(TD_ROOT).as_posix()
               for p in TD_ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts]
    tracked = {p[len('torsional-diffusion/'):] for p in out if p.endswith('.py')}
    assert tracked == set(BASELINE_PINS)
