from .general import CONSTRAINT_TYPE_MAP as ctm_general
from .wefe import CONSTRAINT_TYPE_MAP as ctm_wefe
from ._version import __version__ as version

CONSTRAINT_TYPE_MAP = ctm_general | ctm_wefe
