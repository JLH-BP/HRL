# Verifies the META_HRL package exposes its declared version.
from hrl import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
