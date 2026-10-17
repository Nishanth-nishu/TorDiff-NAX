"""Dead-code audit (round5/PLAN.md "no dead code"; plan_P1 §4.3, plan_P2 §6), AST-based, no extra dependency.

1. Every top-level function and class of a library module is referenced by code (an AST ``Name``, an imported name,
   or ``module.name``) somewhere other than its own definition. Mentions in comments and docstrings do not count
   (review P1 S7). Methods and properties are out of scope. Names whose only references are tests are listed in
   TEST_ONLY_API with the milestone that adds their first non-test caller; the list must match exactly.
2. Every imported name is used in its module.
3. Every argparse flag is read (``args.<dest>``) in its module.
4. No TODO / FIXME / XXX markers.
"""
import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
REPO = PKG.parent

# library API whose first non-test caller arrives with a later milestone (round5/IMPLEMENTATION.md)
TEST_ONLY_API = {}


def library_modules():
    return [p for p in PKG.rglob('*.py') if 'tests' not in p.parts]


def references(tree, skip=None):
    """names referenced by code in ``tree``; ``skip`` = a definition node whose own body is not searched for its
    name (recursion does not keep a function alive)"""
    out = set()
    for n in ast.walk(tree):
        if n is skip:
            continue
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            out.add(n.attr)  # module.name or object.attr of a local name
        elif isinstance(n, ast.ImportFrom):
            out.update(a.name for a in n.names)
    return out


def referenced_names(paths):
    names = set()
    for p in paths:
        names |= references(ast.parse(p.read_text(encoding='utf-8')))
    return names


def test_every_definition_is_referenced():
    lib = library_modules()
    experiments = list((REPO / 'experiments').rglob('*.py')) if (REPO / 'experiments').exists() else []
    used_by_tests = referenced_names(list((PKG / 'tests').rglob('*.py')))
    test_only, unused = {}, []
    for p in lib:
        used_elsewhere = referenced_names([q for q in lib + experiments if q != p])
        tree = ast.parse(p.read_text(encoding='utf-8'))
        src = p.read_text(encoding='utf-8')
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            name = node.name
            if name == 'main' and "__name__ == '__main__'" in src:
                continue
            own = set()
            for other in tree.body:
                if other is not node:
                    own |= references(other)
            if name in used_elsewhere or name in own:
                continue
            if name in used_by_tests:
                test_only[name] = p.name
            else:
                unused.append(f'{p.name}:{name}')
    assert unused == []
    assert set(test_only) == set(TEST_ONLY_API), test_only


def test_comment_mentions_do_not_count():
    tree = ast.parse('def f():\n    """g is mentioned here"""\n    # and g here\n    return 1\n\ndef g():\n    pass\n')
    assert 'g' not in references(tree.body[0])


def test_imports_are_used():
    for p in library_modules():
        if p.name == '__init__.py':
            continue
        tree = ast.parse(p.read_text(encoding='utf-8'))
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
        src = p.read_text(encoding='utf-8')
        for flag in pattern.findall(src):
            assert re.search(rf'\.{flag}\b', src), f'{p.name}: --{flag} is never read'


def test_no_todo_markers():
    for p in library_modules():
        assert not re.search(r'\b(TODO|FIXME|XXX)\b', p.read_text(encoding='utf-8')), p
