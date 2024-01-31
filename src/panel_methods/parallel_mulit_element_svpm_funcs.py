import numpy as np
from .spm_funcs import point_in_polygon
from . import geometric_integrals as gi
from ..code_collections import data_collections as dc
from ..multi_element_airfoil.airfoil_setup import create_clean_panelized_geometry
import numba as nb
import scipy as sp
import pathlib
import tqdm
from itertools import product
