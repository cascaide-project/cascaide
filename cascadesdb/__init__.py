"""High-level client for interacting with CascadesDB metadata."""

from importlib import import_module

__all__ = ["CascadesDBClient", "Record", "CascadesDBError", "RecordNotFound", "DownloadError"]


def __getattr__(name):
    if name in __all__:
        module = import_module(".client", __name__)
        return getattr(module, name)
    raise AttributeError(name)
