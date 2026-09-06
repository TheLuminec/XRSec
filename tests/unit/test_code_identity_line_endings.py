"""The code identity must not depend on how a checkout wrote its line endings."""
import pytest

from results_log import digest_tree


pytestmark = pytest.mark.unit

LF = "def f():\n    return 1\n"


def test_lf_and_crlf_checkouts_of_the_same_code_share_one_identity(tmp_path):
    """
    Measured 2026-09-06: the same code read 100bd18472 on LF files and 72b8053ec2 under
    autocrlf. Three machines with different settings would record three identities for one
    tree, and a resume keyed on it would re-run everything instead of matching.
    """
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for root in (lf, crlf):
        (root / "pkg").mkdir(parents=True)
    (lf / "pkg" / "a.py").write_bytes(LF.encode())
    (crlf / "pkg" / "a.py").write_bytes(LF.replace("\n", "\r\n").encode())
    assert digest_tree(lf) == digest_tree(crlf)


def test_a_real_change_still_changes_the_identity(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    before = digest_tree(tmp_path)
    (tmp_path / "a.py").write_bytes(b"x = 2\n")
    assert digest_tree(tmp_path) != before


def test_the_relative_path_is_part_of_the_identity(tmp_path):
    """Moving a file is a change too; equal bytes alone are not enough."""
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    before = digest_tree(tmp_path)
    (tmp_path / "a.py").rename(tmp_path / "b.py")
    assert digest_tree(tmp_path) != before
