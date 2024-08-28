from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator
from sklearn.datasets import load_diabetes

# https://github.com/davide-burba/code-collection/blob/main/code/quantile-matching/quantile_matching.ipynb