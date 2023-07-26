"""Test pyordstat."""

import pyordstat


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(pyordstat.__name__, str)
