"""Tests for the ds-covid package."""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    try:
        import features
        import models
        import explorationdata
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import package modules: {e}")


def test_version():
    """Test that version is accessible."""
    from src import __version__
    assert __version__ == "0.1.0"