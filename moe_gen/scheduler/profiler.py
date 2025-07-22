import gc
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class Profiler:
    """
    By default homogeneous node. Will do the profiling on the device 0.
    """

    def __init__(self):
        pass
