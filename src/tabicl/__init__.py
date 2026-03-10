"""Top-level package exports for `tabicl`.

This module intentionally avoids importing submodules that in turn import
`tabicl` at module import time (for example `sklearn.classifier`). Importing
those submodules here causes circular imports when users do ``from tabicl
import ...``. To prevent that, we export lightweight symbols directly and
provide lazy accessors for heavier sklearn-based classes.
"""

# Export core model classes and config eagerly -- these are pure model modules
# and do not import `tabicl` back.
from .model.inference_config import InferenceConfig
from .model.tabicl import TabICL


__all__ = [
	"InferenceConfig",
	"TabICL",
	"TabICLClassifier",
	"MantisICLClassifier",
	"MantisICLClassifierV2",
]


def __getattr__(name: str):
	"""Lazy-import heavier sklearn wrapper classes on attribute access.

	This avoids circular import errors when importing the package. Example:
		from tabicl import TabICLClassifier

	will import the sklearn wrapper only when the attribute is referenced.
	"""
	if name == "TabICLClassifier":
		from .sklearn.classifier import TabICLClassifier

		return TabICLClassifier
	if name == "MantisICLClassifier":
		from .sklearn.classifier import MantisICLClassifier

		return MantisICLClassifier
	if name == "MantisICLClassifierV2":
		from .sklearn.classifier import MantisICLClassifierV2

		return MantisICLClassifierV2
	raise AttributeError(f"module 'tabicl' has no attribute '{name}'")


def __dir__():
    return __all__