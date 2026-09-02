"""
Feature extractor implementations.

Every module in this package is imported on discovery, so a new extractor file is
picked up with no edits here. See ``model/feature_extractor.py`` for the contract.
"""

import importlib
import pkgutil


def _import_all() -> None:
    for module in pkgutil.iter_modules(__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module.name}")


_import_all()
