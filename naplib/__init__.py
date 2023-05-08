import logging
from typing import Union

# Set up logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)
logger.addHandler(_console)

def set_logging(level: Union[int, str]):
    '''
    Sets the log level at the module level. All functions within this module
    by default use this log level, except when a submodule has its own separate
    log level.

    Parameters
    ----------
    level : string or int
        Any log level that is recognized by python's built-in ``logging`` module
    '''
    if not isinstance(level, (int, str)):
        raise ValueError('Level must be of type int or str')
    
    logger.setLevel(level)
    _console.setLevel(level)


import naplib.features
import naplib.segmentation
import naplib.stats
import naplib.visualization
import naplib.encoding
import naplib.io
import naplib.preprocessing
import naplib.array_ops
import naplib.model_selection
import naplib.utils
from .data import Data, join_fields, concat
import naplib.naplab

__version__ = "0.5.0"

