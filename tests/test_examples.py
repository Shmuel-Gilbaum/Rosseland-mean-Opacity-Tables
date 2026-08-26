"""Every code block in the shipped documents runs, and prints what it claims.

The documents carry plain copyable snippets rather than an imitated interactive
session, so nothing else runs them. This does: each file's blocks execute in
order in one namespace, and every line printed is checked against the inline
comment beside its `print`.

A comment ending in `...` is a prefix, for digits that vary by platform.
"""
import io
import re
import contextlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["README.md", "USAGE.md"]

_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
_EXPECT = re.compile(r"^.*#\s*->\s*(.*?)\s*$")


def _blocks(text):
    return _BLOCK.findall(text)


def _expectations(code):
    """Each `# ->` comment, in order, as one expected line of output.

    The marker separates an expected line from an ordinary comment, so a block
    can explain itself and still be checked. A comment beside a `print` covers
    that call; one on its own line covers the next line printed, which is how a
    loop's output is written.
    """
    return [m.group(1) for m in
            (_EXPECT.match(line) for line in code.splitlines()) if m]


def _matches(printed, claimed):
    """`...` in a comment stands for any run of characters, as in doctest.

    Trailing digits are not stable across platforms, and the last digit of an
    array element moved between two runs on this machine.
    """
    if "..." not in claimed:
        return printed == claimed
    parts = claimed.split("...")
    if not printed.startswith(parts[0]):
        return False
    at = len(parts[0])
    for part in parts[1:-1]:
        found = printed.find(part, at)
        if found < 0:
            return False
        at = found + len(part)
    return printed.endswith(parts[-1]) and len(printed) - len(parts[-1]) >= at


@pytest.mark.parametrize("name", DOCS)
def test_every_example_runs_and_prints_what_it_claims(name):
    path = ROOT / name
    text = path.read_text()
    blocks = _blocks(text)
    assert blocks, f"{name} has no python blocks"
    ns = {}
    claimed = []
    skipped = []
    buf = io.StringIO()
    for i, code in enumerate(blocks):
        want = _expectations(code)
        with contextlib.redirect_stdout(buf):
            try:
                exec(compile(code, f"{name} block {i + 1}", "exec"), ns)
            except ImportError as exc:
                # A block may show a companion package this one does not
                # depend on. Skip it rather than making it a test dependency.
                skipped.append((i + 1, str(exc)))
                continue
            except Exception as exc:            # noqa: BLE001
                pytest.fail(f"{name} block {i + 1} raised {exc!r}\n{code}")
        claimed += want
    for n_block, why in skipped:
        print(f"{name} block {n_block} skipped: {why}")
    printed = buf.getvalue().splitlines()
    assert len(printed) == len(claimed), (
        f"{name}: {len(printed)} lines printed, {len(claimed)} claimed in "
        f"comments.\nprinted: {printed}\nclaimed: {claimed}")
    for got, want in zip(printed, claimed):
        assert _matches(got, want), f"{name}: printed {got!r}, comment says {want!r}"


@pytest.mark.parametrize("name", DOCS)
def test_no_block_carries_an_interactive_prompt(name):
    """The documents ship copyable code. A leading prompt breaks a paste."""
    for i, code in enumerate(_blocks((ROOT / name).read_text())):
        assert ">>>" not in code, f"{name} block {i + 1} carries a prompt"
