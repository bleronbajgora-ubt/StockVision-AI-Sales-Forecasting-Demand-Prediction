import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".SparseEfficiencyWarning.")


----------------------------
Configuration / constants
----------------------------
DATA_DEFAULT_PATH = "data/amazon.csv"
RANDOM_STATE = 42

TEXT_COLS = ["review_title", "review_content"]
NUMERIC_COLS = ["discounted_price", "actual_price", "rating_count"]
LABEL_COL = "rating"  # original rating (1-5)

Ratings rule for sentiment target
remove rating == 3
rating >= 4 => positive (1)
rating <= 2 => negative (0)
----------------------------
Utility parsing functions
----------------------------
_money_re = re.compile(r"[^\d.]")  # remove currency symbols and commas etc.


def parse_money(x) -> float:
    """Parse values like '₹32,999' or '32,999' or 32999 into float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = _money_re.sub("", s)  # keep digits and dot
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan
