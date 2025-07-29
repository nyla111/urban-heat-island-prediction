# Package version
__version__ = "1.0.0"

# Import key components to make them available at package level
from .config import *
from .dependencies import *

__all__ = ['data', 'results', 'train', 'config', 'dependencies']