# Generic imports
import os
import sys
import math
import shutil
import numpy as np

from matplotlib import pyplot          as plt
from types      import SimpleNamespace as pms

# Custom imports
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.pca       import *
from hummingbird.src.plot.plot       import *

###############################################
### Base app
class base_app():
    def __init__(self, name):

        # Create paths for results and open repositories
        self.results_path  = 'results'
        os.makedirs(self.results_path, exist_ok=True)
        self.results_path += '/'+name
        os.makedirs(self.results_path, exist_ok=True)
