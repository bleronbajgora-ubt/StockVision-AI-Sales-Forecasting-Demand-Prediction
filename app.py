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

def parse_count(x) -> float:
    """Parse values like '1,07,687' or '7,298' or 7298 into float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = re.sub(r"[^\d]", "", s)
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def safe_text(x) -> str:
    """Ensure text is string; missing -> empty."""
    if pd.isna(x):
        return ""
    return str(x)


def build_text_series(df: pd.DataFrame) -> pd.Series:
    """Concatenate title + content; improves signal vs either alone."""
    title = df.get("review_title", pd.Series([""] * len(df))).map(safe_text)
    content = df.get("review_content", pd.Series([""] * len(df))).map(safe_text)
    # Feature engineering insight:
    # - Titles often contain a dense sentiment summary ("Excellent", "Bad", "Value for money")
    # - Content provides context and nuance; concatenating helps linear models and TF-IDF capture both.
    return (title + " " + content).str.strip()


----------------------------
Data preprocessing
----------------------------
@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    # Using low_memory=False for stable dtype inference (production-safe on medium CSVs).
    return pd.read_csv(path, low_memory=False)
def preprocess(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    1) Remove rating==3
    2) Binary target: rating>=4 -> 1, rating<=2 -> 0
    3) Handle missing values (text->"", numeric->median at pipeline stage)
    4) Normalize numeric features (done in pipeline)
    """
    df = df_raw.copy()

    # Ensure required columns exist
    missing_cols = [c for c in (TEXT_COLS + NUMERIC_COLS + [LABEL_COL, "category"]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Coerce rating to numeric
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL])

    # Remove neutral class (rating==3)
    df = df[df[LABEL_COL] != 3].copy()

    # Create binary sentiment label
    y = (df[LABEL_COL] >= 4).astype(int)
    # Drop any invalid ratings outside 1-5 if present
    valid_mask = df[LABEL_COL].between(1, 5)
    df = df[valid_mask].copy()
    y = y.loc[df.index]

    # Parse numeric columns
    df["discounted_price"] = df["discounted_price"].map(parse_money)
    df["actual_price"] = df["actual_price"].map(parse_money)
    df["rating_count"] = df["rating_count"].map(parse_count)

    # Text columns: fill missing
    for c in TEXT_COLS:
        df[c] = df[c].map(safe_text)

    # Category: fill missing (not used in models here, but useful for UI summary)
    df["category"] = df["category"].map(safe_text)

    # NOTE: Numeric missing values are handled in pipeline via FunctionTransformer+np.nan_to_num after median fill.
    return df, y


def median_impute_numpy(X: np.ndarray) -> np.ndarray:
    """Median impute each column; robust for heavy-tailed price/count features."""
    X = X.astype(float)
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        col = X_out[:, j]
        med = np.nanmedian(col)
        if np.isnan(med):
            med = 0.0
        col = np.where(np.isnan(col), med, col)
        X_out[:, j] = col
    return X_out


Feature pipelines
----------------------------
@dataclass(frozen=True)
class FeaturizationConfig:
    max_features: int
    ngram_range: Tuple[int, int]


def make_text_vectorizer(cfg: FeaturizationConfig) -> TfidfVectorizer:
    # Production-ish defaults:
    # - sublinear_tf helps dampen overly frequent words
    # - min_df=2 reduces noise from one-off tokens
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )


def build_features(
    df: pd.DataFrame,
    feature_type: str,
    tfidf_cfg: FeaturizationConfig,
) -> Tuple[csr_matrix, Optional[np.ndarray], TfidfVectorizer]:
    """
    Returns:
      X_text: sparse TF-IDF matrix
      X_num_scaled (dense) or None
      fitted vectorizer
    """
    text = build_text_series(df)
    vectorizer = make_text_vectorizer(tfidf_cfg)
    X_text = vectorizer.fit_transform(text)

    if feature_type == "Text only":
        return X_text.tocsr(), None, vectorizer

    # Hybrid: numeric features + TF-IDF stacking
    X_num = df[NUMERIC_COLS].to_numpy(dtype=float)
    X_num = median_impute_numpy(X_num)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Hybrid feature engineering insight:
    # - TF-IDF captures linguistic sentiment cues.
    # - Numeric signals can correlate with sentiment bias:
    #   e.g., very low discounted_price / high rating_count might correlate with more "value-for-money" positivity.
    X_hybrid = hstack([X_text, csr_matrix(X_num_scaled)], format="csr")
    return X_hybrid, X_num_scaled, vectorizer
def transform_features_for_inference(
    df_one: pd.DataFrame,
    feature_type: str,
    vectorizer: TfidfVectorizer,
    scaler: Optional[StandardScaler],
) -> csr_matrix:
    text = build_text_series(df_one)
    X_text = vectorizer.transform(text)

    if feature_type == "Text only":
        return X_text.tocsr()

    X_num = df_one[NUMERIC_COLS].to_numpy(dtype=float)
    X_num = median_impute_numpy(X_num)
    assert scaler is not None
    X_num_scaled = scaler.transform(X_num)
    return hstack([X_text, csr_matrix(X_num_scaled)], format="csr")
