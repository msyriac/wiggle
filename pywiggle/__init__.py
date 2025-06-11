
from importlib import import_module as _import_module
import sys as _sys


from . import _wiggle
from .core import *



try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version(__name__)
except Exception:          # pragma: no cover
    __version__ = "0.0.0"
