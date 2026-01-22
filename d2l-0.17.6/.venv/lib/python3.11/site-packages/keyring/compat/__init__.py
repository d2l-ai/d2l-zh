__all__ = ['properties']


try:
    from jaraco.classes import properties
except ImportError:  # pragma: no cover
    from . import properties  # type: ignore[no-redef]
