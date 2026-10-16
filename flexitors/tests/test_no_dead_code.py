"""Dead-code audit (round5/PLAN.md "no dead code"; plan_P1 §4.3, plan_P2 §6), AST-based, no extra dependency.

1. Every top-level function and class of a library module is referenced outside its own definition. Names whose
   only references are tests are listed in TEST_ONLY_API with the milestone that adds their first non-test caller;
   the list must match exactly, so it cannot go stale silently.
2. Every imported name is used in its module.
3. Every argparse flag is read (``args.<dest>``) in its module.
4. No TODO / FIXME / XXX markers.
"""
import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
REPO = PKG.parent

# library API whose first non-test caller arrives with a later milestone (round5/IMPLEMENTATION.md, milestone 1)
TEST_ONLY_API = {
    'validate_conformer': 'Gate-0 / Exp 3 evaluation CLIs (milestone 2)',
}


def library_modules():
    return [p for p in PKG.rglob('*.py') if 'tests' not in p.parts]


def referenced_names(paths):
    names = set()
    for p in paths:
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.ImportFrom):
                names.update(a.name for a in n.names)
    return names


def definitions(path):
    tree = ast.parse(path.read_text())
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


def test_every_definition_is_referenced():
    lib = library_modules()
    experiments = list((REPO / 'experiments').rglob('*.py')) if (REPO / 'experiments').exists() else []
    used_by_tests = referenced_names(list((PKG / 'tests').rglob('*.py')))
    test_only, unused = {}, []
    for p in lib:
        used_elsewhere = referenced_names([q for q in lib + experiments if q != p])
        own = p.read_text()
        for name in definitions(p):
            if name == 'main' and "__name__ == '__main__'" in own:
                continue
            if name in used_elsewhere or len(re.findall(rf'\b{name}\b', own)) > 1:
                continue
            if name in used_by_tests:
                test_only[name] = p.name
            else:
                unused.append(f'{p.name}:{name}')
    assert unused == []
    assert set(test_only) == set(TEST_ONLY_API), test_only


def test_imports_are_used():
    for p in library_modules():
        if p.name == '__init__.py':
            continue
        tree = ast.parse(p.read_text())
        imported = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported += [(a.asname or a.name).split('.')[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module != '__future__':
                imported += [a.asname or a.name for a in n.names]
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not [i for i in imported if i not in used], p


def test_cli_flags_are_read():
    pattern = re.compile(r"add_argument\(\s*'--(\w+)'")
    for p in library_modules():
        src = p.read_text()
        for flag in pattern.findall(src):
            assert re.search(rf'\.{flag}\b', src), f'{p.name}: --{flag} is never read'


def test_no_todo_markers():
    for p in library_modules():
        assert not re.search(r'\b(TODO|FIXME|XXX)\b', p.read_text()), p
