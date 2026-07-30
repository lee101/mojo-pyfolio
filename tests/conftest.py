import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

# empyrical-reloaded 0.5.9 still references aliases removed by NumPy 2.
np.NINF = -np.inf
np.PINF = np.inf
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message='Module "zipline.assets" not found')
