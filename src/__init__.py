"""MLOps project package initialization."""

import logging
import sys
from pathlib import Path

# Setup project root path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging for the package
logger = logging.getLogger(__name__)
logger.info(f"Initialized project with root: {PROJECT_ROOT}")

__version__ = "0.1.0"
