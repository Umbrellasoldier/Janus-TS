from __future__ import annotations

from janus_ts.native_stack import _glibc_versions, _numeric_version


def test_glibc_symbol_versions_are_numeric_and_deduplicated():
    symbols = "(GLIBC_2.14) GLIBC_2.2.5 GLIBCXX_3.4.21 GLIBC_2.14 GLIBC_2.4"
    assert _glibc_versions(symbols) == ("2.2.5", "2.4", "2.14")
    assert _numeric_version("2.14") < _numeric_version("2.28")
