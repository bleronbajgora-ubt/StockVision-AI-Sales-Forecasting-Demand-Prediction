"""
=============================================================================
 INTELLIGENT AMAZON REVIEW ANALYZER
=============================================================================
 A production-grade Streamlit application for binary sentiment classification
 of Amazon product reviews using multiple ML algorithms, hyperparameter tuning,
 clustering analysis, and real-time prediction.

 Courses  : Machine Learning (AI)  |  Laboratoric Course
 Dataset  : Amazon Product Reviews CSV
 Target   : Positive (rating >= 4) vs Negative (rating <= 2) — neutral removed
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*SparseEfficiencyWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DEFAULT_PATH: str = "data/amazon.csv"
RANDOM_STATE: int = 42

# Column schema
TEXT_COLS: List[str] = ["review_title", "review_content"]
NUMERIC_COLS: List[str] = ["discounted_price", "actual_price", "rating_count"]
LABEL_COL: str = "rating"

# Sentiment labeling rule:
#   rating == 3  →  removed (neutral / ambiguous)
#   rating >= 4  →  Positive  (label = 1)
#   rating <= 2  →  Negative  (label = 0)

PALETTE: Dict[str, str] = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#64748b",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeaturizationConfig:
    """
    Immutable TF-IDF configuration passed throughout the pipeline.
    Freezing prevents accidental mutation between train and inference stages.
    """
    max_features: int
    ngram_range: Tuple[int, int]
    sublinear_tf: bool = True
    min_df: int = 2

    def __str__(self) -> str:
        return (
            f"TF-IDF(max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, sublinear_tf={self.sublinear_tf})"
        )


@dataclass
class ModelResult:
    """Container for a single trained model's complete evaluation output."""
    name: str
    feature_type: str
    tfidf_cfg: FeaturizationConfig
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    train_time_s: float
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]   # probabilities for ROC (None for LinearSVC)
    best_params: Optional[Dict] = None
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "Model":          self.name,
            "Feature Type":   self.feature_type,
            "Accuracy":       round(self.accuracy,   4),
            "Precision":      round(self.precision,  4),
            "Recall":         round(self.recall,     4),
            "F1 Score":       round(self.f1,         4),
            "ROC-AUC":        round(self.roc_auc, 4) if self.roc_auc is not None else "N/A",
            "Train Time (s)": round(self.train_time_s, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PARSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
_MONEY_RE = re.compile(r"[^\d\.]")   # strip currency symbols, commas


def parse_money(x) -> float:
    """
    Parse monetary strings such as '₹32,999', '$1,499.99', or bare numerics.
    Returns np.nan on failure to enable downstream median imputation.
    """
    if pd.isna(x):
        return np.nan
    cleaned = _MONEY_RE.sub("", str(x).strip())
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan


def parse_count(x) -> float:
    """
    Parse Indian-style comma-separated counts like '1,07,687'.
    Returns np.nan on failure.
    """
    if pd.isna(x):
        return np.nan
    cleaned = re.sub(r"[^\d]", "", str(x).strip())
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan


def safe_text(x) -> str:
    """Coerce to string; NaN → empty string."""
    return "" if pd.isna(x) else str(x)


def build_text_series(df: pd.DataFrame) -> pd.Series:
    """
    Concatenate review_title and review_content into a single text feature.

    Design rationale:
    - Titles carry dense, high-signal sentiment terms ("Excellent", "Waste of money").
    - Content provides richer context and nuance.
    - Concatenating both lets TF-IDF exploit complementary cues in one feature space.
    """
    title   = df.get("review_title",   pd.Series([""] * len(df), index=df.index)).map(safe_text)
    content = df.get("review_content", pd.Series([""] * len(df), index=df.index)).map(safe_text)
    return (title + " " + content).str.strip()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    """Load CSV with stable dtype inference. Cached to avoid repeated disk I/O."""
    return pd.read_csv(path, low_memory=False)


def preprocess(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline.

    Steps
    ─────
    1. Validate required columns are present.
    2. Coerce rating to numeric; drop unparseable rows.
    3. Remove neutral reviews (rating == 3) — these add noise to binary classification.
    4. Create binary sentiment label: rating >= 4 → 1 (positive), rating <= 2 → 0 (negative).
    5. Parse monetary and count columns into float (handles Indian currency format).
    6. Engineer discount_pct feature: (actual - discounted) / actual * 100.
    7. Fill text NaNs with empty strings.

    Returns
    -------
    df : pd.DataFrame   cleaned DataFrame (features intact)
    y  : pd.Series      binary labels aligned with df.index
    """
    df = df_raw.copy()

    required = TEXT_COLS + NUMERIC_COLS + [LABEL_COL, "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # --- Rating coercion & neutral removal ---
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL])
    df = df[df[LABEL_COL].between(1, 5)].copy()   # valid rating range
    df = df[df[LABEL_COL] != 3].copy()            # drop neutral class

    # --- Binary sentiment label ---
    y = (df[LABEL_COL] >= 4).astype(int)

    # --- Numeric feature parsing ---
    df["discounted_price"] = df["discounted_price"].map(parse_money)
    df["actual_price"]     = df["actual_price"].map(parse_money)
    df["rating_count"]     = df["rating_count"].map(parse_count)

    # --- Engineered numeric feature: discount percentage ---
    # Captures relative deal quality; deeper discounts may correlate with
    # "value-for-money" positivity bias in reviews.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["discount_pct"] = np.where(
            df["actual_price"] > 0,
            (df["actual_price"] - df["discounted_price"]) / df["actual_price"] * 100,
            np.nan,
        )

    # --- Text normalization ---
    for c in TEXT_COLS:
        df[c] = df[c].map(safe_text)

    df["category"] = df.get("category", pd.Series([""] * len(df), index=df.index)).map(safe_text)

    return df, y


def median_impute(X: np.ndarray) -> np.ndarray:
    """
    Column-wise median imputation for numeric features.
    Median is preferred over mean for price/count features, which are
    right-skewed due to premium products inflating the upper tail.
    """
    X = X.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        X[:, j] = np.where(np.isnan(col), 0.0 if np.isnan(med) else med, col)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
_NUMERIC_COLS_EXTENDED = NUMERIC_COLS + ["discount_pct"]


def make_text_vectorizer(cfg: FeaturizationConfig) -> TfidfVectorizer:
    """
    Construct a TF-IDF vectorizer from a FeaturizationConfig.

    Key design choices
    ──────────────────
    sublinear_tf=True  : log-normalizes term frequencies, dampening the
                         outsized influence of very frequent words.
    min_df=2           : ignores tokens in fewer than 2 documents,
                         reducing vocabulary noise and overfitting risk.
    strip_accents      : handles non-ASCII product/brand names robustly.
    stop_words         : removes English function words with no sentiment value.
    """
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        stop_words="english",
        sublinear_tf=cfg.sublinear_tf,
        min_df=cfg.min_df,
        strip_accents="unicode",
    )


def build_train_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_type: str,
    cfg: FeaturizationConfig,
) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer, Optional[StandardScaler]]:
    """
    Fit-transform on training split; transform-only on test split.
    This is critical to prevent data leakage: the vectorizer vocabulary and
    scaler statistics must never be influenced by test-set samples.

    Parameters
    ----------
    feature_type : "Text only" | "Hybrid"
        Hybrid appends scaled numeric features (price, count, discount%)
        to the TF-IDF sparse matrix via horizontal stacking.

    Hybrid motivation
    ─────────────────
    Numeric signals carry implicit sentiment bias:
    - High discount_pct → "value-for-money" positivity.
    - High rating_count → established product, potentially higher bar.
    Linear models can exploit both modalities when stacked in one feature matrix.

    Returns
    -------
    X_train, X_test : csr_matrix (sparse)
    vectorizer       : fitted TfidfVectorizer (serialized for inference)
    scaler           : fitted StandardScaler or None
    """
    vectorizer = make_text_vectorizer(cfg)
    X_tr_text  = vectorizer.fit_transform(build_text_series(df_train))
    X_te_text  = vectorizer.transform(build_text_series(df_test))

    if feature_type == "Text only":
        return X_tr_text.tocsr(), X_te_text.tocsr(), vectorizer, None

    # --- Numeric block ---
    X_tr_num = median_impute(df_train[_NUMERIC_COLS_EXTENDED].to_numpy(dtype=float))
    X_te_num = median_impute(df_test[_NUMERIC_COLS_EXTENDED].to_numpy(dtype=float))

    scaler       = StandardScaler()
    X_tr_num_sc  = scaler.fit_transform(X_tr_num)
    X_te_num_sc  = scaler.transform(X_te_num)

    X_tr = hstack([X_tr_text, csr_matrix(X_tr_num_sc)], format="csr")
    X_te = hstack([X_te_text, csr_matrix(X_te_num_sc)], format="csr")

    return X_tr, X_te, vectorizer, scaler


def build_inference_features(
    df_one: pd.DataFrame,
    feature_type: str,
    vectorizer: TfidfVectorizer,
    scaler: Optional[StandardScaler],
) -> csr_matrix:
    """
    Transform a single sample for real-time prediction.
    Uses the already-fitted vectorizer and scaler — no re-fitting occurs.
    """
    X_text = vectorizer.transform(build_text_series(df_one))
    if feature_type == "Text only":
        return X_text.tocsr()
    X_num    = median_impute(df_one[_NUMERIC_COLS_EXTENDED].to_numpy(dtype=float))
    assert scaler is not None, "Scaler must be provided for Hybrid mode."
    X_num_sc = scaler.transform(X_num)
    return hstack([X_text, csr_matrix(X_num_sc)], format="csr")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_model_registry(
    include_svm: bool,
    knn_k: int,
    rf_n_estimators: int,
    rf_max_depth: Optional[int],
    mlp_arch: str,
) -> Dict[str, BaseEstimator]:
    """
    Central registry of all supported classifiers with documented rationale.

    Algorithm justifications
    ────────────────────────
    Logistic Regression
        Strong linear baseline for high-dimensional sparse TF-IDF vectors.
        Bag-of-words spaces for sentiment tasks are often nearly linearly
        separable; LR exploits this efficiently with L2 regularization.

    KNN (k-Nearest Neighbours)
        Non-parametric comparison using cosine distance (more meaningful
        for TF-IDF than Euclidean). Included as a lower-bound reference;
        the curse of dimensionality limits its effectiveness in TF-IDF space.

    Random Forest
        Ensemble of decision trees via bagging. Less competitive on raw
        TF-IDF, but can gain an edge in Hybrid mode where numeric features
        introduce structured non-linear signals. class_weight="balanced"
        corrects for label imbalance.

    MLP (Multi-Layer Perceptron)
        Feed-forward neural network with ReLU activations and adam optimizer.
        Two architectures are compared:
          (50,)     — shallower, lower capacity, less overfitting risk.
          (100, 50) — deeper, higher capacity, needs stronger regularization.
        Early stopping (n_iter_no_change=5) prevents overfitting automatically.

    Linear SVM (bonus)
        Typically the top performer on TF-IDF sentiment tasks. Maximum-margin
        classification in sparse space. LinearSVC uses liblinear's efficient
        primal solver, scaling well to large vocabularies.
    """
    hidden = (50,) if mlp_arch == "(50,)" else (100, 50)

    registry: Dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=knn_k, metric="cosine",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        f"MLP {hidden}": MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=1e-3, max_iter=60,
            early_stopping=True, n_iter_no_change=5,
            random_state=RANDOM_STATE, verbose=False,
        ),
    }

    if include_svm:
        registry["Linear SVM"] = LinearSVC(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        )

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(
    model: BaseEstimator,
    name: str,
    X_train: csr_matrix,
    X_test: csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_type: str,
    cfg: FeaturizationConfig,
) -> ModelResult:
    """
    Fit a classifier and compute a comprehensive evaluation suite.

    Metrics
    ───────
    Accuracy   : overall fraction of correct predictions.
    Precision  : of all predicted positives, how many are truly positive.
    Recall     : of all true positives, how many did the model capture.
    F1 Score   : harmonic mean of precision and recall.
                 Primary metric — robust to class imbalance.
    ROC-AUC    : area under the ROC curve.
                 Uses predict_proba where available,
                 decision_function as a fallback for LinearSVC.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = model.predict(X_test)

    roc_auc = None
    y_proba = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        if y_proba is not None:
            roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    return ModelResult(
        name=name,
        feature_type=feature_type,
        tfidf_cfg=cfg,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        train_time_s=elapsed,
        y_pred=y_pred,
        y_proba=y_proba,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_logistic_regression(
    X_train: csr_matrix, y_train: np.ndarray, cv_splits: int,
) -> GridSearchCV:
    """
    GridSearchCV for Logistic Regression.

    Search space
    ────────────
    C        : inverse regularization strength.
               Smaller C → stronger regularization → simpler model.
    solver   : lbfgs (L2, dense) vs saga (supports L1 and L2, handles sparse).
    penalty  : L2 (ridge) induces weight shrinkage.

    CV strategy: StratifiedKFold preserves class ratios in every fold.
    Scoring: F1 (handles class imbalance better than accuracy).
    """
    param_grid = {
        "C":      [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "saga"],
    }
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(lr, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs


def tune_random_forest(
    X_train: csr_matrix, y_train: np.ndarray, cv_splits: int,
) -> GridSearchCV:
    """
    GridSearchCV for Random Forest.

    Search space
    ────────────
    n_estimators       : more trees reduce variance (at compute cost).
    max_depth          : controls overfitting; None = fully grown trees.
    min_samples_split  : minimum samples required to split an internal node.
    min_samples_leaf   : minimum samples in a leaf; smooths decision boundaries.
    """
    param_grid = {
        "n_estimators":      [200, 500],
        "max_depth":         [None, 12, 24],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(rf, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs


def tune_mlp(
    X_train: csr_matrix, y_train: np.ndarray, cv_splits: int,
) -> GridSearchCV:
    """
    GridSearchCV for MLP Classifier.

    Search space
    ────────────
    hidden_layer_sizes : compares the two required architectures.
                         (50,) vs (100, 50) — capacity vs generalization tradeoff.
    alpha              : L2 regularization weight.
                         Critical for preventing overfitting in high-dimensional
                         TF-IDF space; higher alpha → simpler decision boundary.
    learning_rate_init : initial step size for the adam optimizer.
    """
    param_grid = {
        "hidden_layer_sizes": [(50,), (100, 50)],
        "alpha":              [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 5e-4],
    }
    mlp = MLPClassifier(
        activation="relu", solver="adam",
        early_stopping=True, n_iter_no_change=5,
        max_iter=80, random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(mlp, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING (UNSUPERVISED)
# ─────────────────────────────────────────────────────────────────────────────
def run_kmeans_clustering(
    df: pd.DataFrame,
    y: pd.Series,
    cfg: FeaturizationConfig,
    k: int,
    sample_size: int = 1500,
) -> Dict[str, object]:
    
    if len(df) > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(df), sample_size, replace=False)
        df_s, y_s = df.iloc[idx], y.iloc[idx]
    else:
        df_s, y_s = df, y

    vectorizer = make_text_vectorizer(cfg)
    X = vectorizer.fit_transform(build_text_series(df_s))

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    cluster_labels = km.fit_predict(X)

    pca  = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X.toarray())

    ari = float(adjusted_rand_score(y_s.to_numpy(), cluster_labels))

    composition: List[Dict] = []
    for c in range(k):
        mask = cluster_labels == c
        if mask.sum() > 0:
            composition.append({
                "Cluster": c,
                "Size":    int(mask.sum()),
                "Positive Rate": float(y_s.to_numpy()[mask].mean()),
            })

    return {
        "k":              k,
        "cluster_labels": cluster_labels,
        "true_labels":    y_s.to_numpy(),
        "X_2d":           X_2d,
        "ari":            ari,
        "explained_var":  pca.explained_variance_ratio_.tolist(),
        "composition":    composition,
        "sampled":        len(df_s),
    }

def _apply_dark_style(fig: Figure, ax) -> None:
    """Apply a consistent dark academic theme to matplotlib figures."""
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#f1f5f9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Figure:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.5, linecolor="#334155", cbar=False, ax=ax,
        annot_kws={"size": 14, "color": "white"},
    )
    ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_xticklabels(["Negative (0)", "Positive (1)"], fontsize=8)
    ax.set_yticklabels(["Negative (0)", "Positive (1)"], fontsize=8, rotation=0)
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_roc_curves(results: List[ModelResult]) -> Figure:
    """Overlay ROC curves for all models that support probability/score estimates."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for res, color in zip(results, colors):
        if res.y_proba is not None and res.roc_auc is not None:
            fpr, tpr, _ = roc_curve(res.y_pred, res.y_proba)
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{res.name}  (AUC={res.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random baseline (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — All Models", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right",
              facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9")
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_metric_comparison(results: List[ModelResult]) -> Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = [r.name for r in results]
    x       = np.arange(len(labels))
    width   = 0.2
    colors  = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [getattr(r, metric) for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.87)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="#cbd5e1",
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Performance Comparison", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    label_names: Optional[List[str]] = None,
) -> Figure:
    fig, ax  = plt.subplots(figsize=(7, 5))
    unique   = np.unique(labels)
    cmap     = plt.cm.get_cmap("tab10", len(unique))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        name = label_names[i] if label_names else f"Cluster {lbl}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(i)], s=14, alpha=0.8, label=name, linewidths=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PCA Component 1", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9")
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    return fig

st.set_page_config(
    page_title="Amazon Review Analyzer · ML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #e2e8f0; }
    .stApp { background-color: #0f172a; }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem; transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #f1f5f9;
        border-left: 4px solid #3b82f6; padding-left: 0.75rem; margin: 1.5rem 0 0.75rem 0;
    }
    .insight-box {
        background: #1e293b; border: 1px solid #334155;
        border-left: 4px solid #22c55e; border-radius: 6px;
        padding: 0.9rem 1rem; margin: 0.8rem 0;
        font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;
    }
    .warn-box {
        background: #1e293b; border-left: 4px solid #f59e0b;
        border-radius: 6px; padding: 0.9rem 1rem; margin: 0.8rem 0;
        font-size: 0.88rem; color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }
    code { font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    data_path = st.text_input("📂 Dataset path", value=DATA_DEFAULT_PATH)

    st.markdown("### Feature Engineering")
    feature_type = st.selectbox(
        "Feature type", ["Text only", "Hybrid"], index=1,
        help="Hybrid adds normalized numeric features (price, count, discount%) to TF-IDF.",
    )

    st.markdown("### TF-IDF Vectorizer")
    max_features = st.selectbox("Vocabulary size (max_features)", [500, 1000, 2000, 5000], index=1)
    ngram_choice = st.selectbox(
        "n-gram range",
        ["(1,1) — Unigrams only", "(1,2) — Unigrams + Bigrams"], index=1,
        help="Bigrams capture phrases like 'not good' that unigrams miss.",
    )
    ngram_range = (1, 1) if "Unigrams only" in ngram_choice else (1, 2)
    tfidf_cfg   = FeaturizationConfig(max_features=int(max_features), ngram_range=ngram_range)

    st.markdown("### Model Selection")
    include_svm     = st.checkbox("Include Linear SVM (bonus model)", value=True)
    available_models = ["Logistic Regression", "KNN", "Random Forest", "MLP (50,)", "MLP (100,50)"]
    if include_svm:
        available_models.append("Linear SVM")
    default_models   = ["Logistic Regression", "Random Forest", "MLP (50,)", "Linear SVM"] \
                       if include_svm else ["Logistic Regression", "Random Forest", "MLP (50,)"]
    model_selection  = st.multiselect("Models to train", options=available_models, default=default_models)

    st.markdown("### Training Setup")
    test_size  = st.slider("Test split size", 0.10, 0.40, 0.20, 0.05)
    knn_k      = st.slider("KNN — k (n_neighbors)", 3, 25, 7, 2)
    rf_n       = st.selectbox("Random Forest — n_estimators", [100, 200, 500], index=1)
    rf_depth   = st.selectbox("Random Forest — max_depth", [None, 8, 12, 24], index=2)
    mlp_arch   = st.selectbox("MLP default architecture", ["(50,)", "(100, 50)"], index=0)

    st.markdown("### Hyperparameter Tuning")
    do_tune_lr  = st.checkbox("GridSearchCV: Logistic Regression", value=False)
    do_tune_rf  = st.checkbox("GridSearchCV: Random Forest",       value=True)
    do_tune_mlp = st.checkbox("GridSearchCV: MLP",                 value=True)
    cv_splits   = st.selectbox("Cross-validation folds", [3, 5], index=0)

    st.markdown("### Clustering")
    k_values = st.multiselect("KMeans — k values", [2, 3, 4, 5], default=[2, 3])

    st.markdown("---")
    train_btn   = st.button("🚀 Train & Evaluate", type="primary",   use_container_width=True)
    cluster_btn = st.button("🔍 Run Clustering",   type="secondary", use_container_width=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
            border:1px solid #334155; border-radius:12px;
            padding:2rem 2.5rem; margin-bottom:1.5rem;">
  <h1 style="margin:0; font-size:2rem; font-weight:700; color:#f1f5f9;">
    🔬 Intelligent Amazon Review Analyzer
  </h1>
  <p style="margin:0.5rem 0 0 0; color:#94a3b8; font-size:0.95rem;">
    Binary Sentiment Classification · Multi-Model Comparison ·
    Hyperparameter Tuning · Clustering · Live Prediction
  </p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_training, tab_tuning, tab_clustering, tab_predict = st.tabs([
    "Overview",
    "Train & Evaluate",
    "Hyperparameter Tuning",
    "Clustering",
    "Live Prediction",
])


# ─────────────────────────────────────────────────────────────────────────────
# CACHED PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_preprocess(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df_raw = load_raw(path)
    return preprocess(df_raw)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    try:
        with st.spinner("Loading and preprocessing dataset…"):
            df, y = cached_preprocess(data_path)

        n_pos   = int(y.sum())
        n_neg   = int((y == 0).sum())
        pos_pct = 100.0 * n_pos / len(y)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total reviews (after filtering)", f"{len(df):,}")
        col2.metric("Positive (rating ≥ 4)", f"{n_pos:,}")
        col3.metric("Negative (rating ≤ 2)", f"{n_neg:,}")
        col4.metric("Class balance — Positive %", f"{pos_pct:.1f}%")

        if pos_pct > 80 or pos_pct < 20:
            st.markdown(
                f'<div class="warn-box">⚠️ <strong>Class Imbalance Detected</strong>: '
                f'{pos_pct:.1f}% positive. Models use <code>class_weight="balanced"</code> '
                f'where applicable. <strong>F1 Score</strong> is the primary evaluation metric.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
        preview_cols = [c for c in
                        ["product_name", "category", "rating", "rating_count",
                         "discounted_price", "actual_price", "discount_pct",
                         "review_title", "review_content"]
                        if c in df.columns]
        st.dataframe(df[preview_cols].head(30), use_container_width=True, height=320)

        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        fig_dist, ax_dist = plt.subplots(figsize=(7, 3.5))
        rc = df[LABEL_COL].value_counts().sort_index()
        bar_colors = [PALETTE["negative"] if r <= 2 else PALETTE["positive"] for r in rc.index]
        ax_dist.bar(rc.index.astype(str), rc.values, color=bar_colors, width=0.6, alpha=0.9)
        ax_dist.set_xlabel("Rating", fontsize=10)
        ax_dist.set_ylabel("Count", fontsize=10)
        ax_dist.set_title("Review Rating Distribution (rating=3 removed)", fontsize=11, fontweight="bold")
        _apply_dark_style(fig_dist, ax_dist)
        plt.tight_layout()
        st.pyplot(fig_dist, clear_figure=True)

        st.markdown('<div class="section-header">Algorithm Selection Rationale</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="insight-box">
<strong>Why these five algorithms?</strong><br><br>
<strong>Logistic Regression</strong> — Gold-standard linear baseline for TF-IDF sentiment.
High-dimensional sparse bag-of-words vectors are nearly linearly separable; LR exploits
this efficiently with L2 regularization.<br><br>
<strong>KNN</strong> — Non-parametric comparison using cosine distance.
Typically underperforms due to the curse of dimensionality in TF-IDF space,
but included as an important lower-bound reference.<br><br>
<strong>Random Forest</strong> — Ensemble of decision trees.
Less competitive on raw TF-IDF, but gains advantage in Hybrid mode
where numeric features introduce structured non-linear signals.<br><br>
<strong>MLP</strong> — Neural network modeling complex feature interactions.
Two architectures compared: <code>(50,)</code> (lower capacity, generalizable)
vs <code>(100, 50)</code> (richer capacity, needs regularization tuning).<br><br>
<strong>Linear SVM</strong> — Often the best performer on TF-IDF tasks.
Maximum-margin classification; <code>LinearSVC</code> uses liblinear's
efficient primal solver scaling well to large vocabularies.
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Failed to load/preprocess dataset: {e}")
        st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_training:
    st.markdown('<div class="section-header">Baseline Model Training</div>', unsafe_allow_html=True)

    if not train_btn:
        st.info("Configure options in the sidebar, then click **Train & Evaluate**.")
    else:
        if not model_selection:
            st.warning("Please select at least one model in the sidebar.")
        else:
            try:
                with st.spinner("Preprocessing dataset…"):
                    df, y = cached_preprocess(data_path)

                X_tr_df, X_te_df, y_train, y_test = train_test_split(
                    df, y, test_size=float(test_size),
                    random_state=RANDOM_STATE, stratify=y,
                )
                y_train_arr = y_train.to_numpy()
                y_test_arr  = y_test.to_numpy()

                st.info(
                    f"Train: **{len(X_tr_df):,}** samples  |  "
                    f"Test: **{len(X_te_df):,}** samples  |  "
                    f"Feature type: **{feature_type}**  |  "
                    f"TF-IDF: **{tfidf_cfg}**"
                )

                with st.spinner("Building features (fit on train only — no data leakage)…"):
                    X_train, X_test, vectorizer, scaler = build_train_features(
                        X_tr_df, X_te_df, feature_type, tfidf_cfg,
                    )

                st.session_state["vectorizer"]   = vectorizer
                st.session_state["scaler"]       = scaler
                st.session_state["feature_type"] = feature_type

                # Build model instances for selected models
                model_registry = build_model_registry(
                    include_svm=include_svm,
                    knn_k=int(knn_k),
                    rf_n_estimators=int(rf_n),
                    rf_max_depth=rf_depth,
                    mlp_arch=mlp_arch,
                )

                selected: Dict[str, BaseEstimator] = {}
                for sel_name in model_selection:
                    for reg_key, reg_model in model_registry.items():
                        if sel_name.replace("(", "").replace(")", "").replace(",", "").replace(" ", "") in \
                           reg_key.replace("(", "").replace(")", "").replace(",", "").replace(" ", ""):
                            selected[reg_key] = reg_model
                            break

                # Ensure explicit MLP instances are present
                if "MLP (50,)" in model_selection and not any("50" in k for k in selected):
                    selected["MLP (50,)"] = MLPClassifier(
                        hidden_layer_sizes=(50,), activation="relu", solver="adam",
                        alpha=1e-4, learning_rate_init=1e-3, max_iter=80,
                        early_stopping=True, n_iter_no_change=5, random_state=RANDOM_STATE,
                    )
                if "MLP (100,50)" in model_selection and not any("100" in k for k in selected):
                    selected["MLP (100, 50)"] = MLPClassifier(
                        hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
                        alpha=1e-4, learning_rate_init=1e-3, max_iter=80,
                        early_stopping=True, n_iter_no_change=5, random_state=RANDOM_STATE,
                    )

                results: List[ModelResult] = []
                progress_bar = st.progress(0.0, text="Training models…")

                for i, (name, model) in enumerate(selected.items(), 1):
                    progress_bar.progress(i / len(selected), text=f"Training {name}…")
                    result = train_and_evaluate(
                        model, name,
                        X_train, X_test,
                        y_train_arr, y_test_arr,
                        feature_type, tfidf_cfg,
                    )
                    results.append(result)

                progress_bar.empty()

                best_result = sorted(results, key=lambda r: r.f1, reverse=True)[0]
                st.session_state["best_model"] = selected.get(
                    best_result.name, list(selected.values())[0]
                )
                st.session_state["results"] = results

                # ── Summary table ───────────────────────────────────────────
                st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
                results_df = pd.DataFrame([r.to_dict() for r in results]).sort_values("F1 Score", ascending=False)
                st.dataframe(results_df, use_container_width=True)

                best_row = results_df.iloc[0]
                st.success(
                    f"🏆 Best baseline model: **{best_row['Model']}** — "
                    f"F1={best_row['F1 Score']:.4f}  |  "
                    f"Accuracy={best_row['Accuracy']:.4f}  |  "
                    f"ROC-AUC={best_row['ROC-AUC']}"
                )

                # ── Visual comparison ───────────────────────────────────────
                st.markdown('<div class="section-header">Visual Comparison</div>', unsafe_allow_html=True)
                col_chart, col_roc = st.columns(2)
                with col_chart:
                    st.pyplot(plot_metric_comparison(results), clear_figure=True)
                with col_roc:
                    roc_results = [r for r in results if r.roc_auc is not None]
                    if roc_results:
                        st.pyplot(plot_roc_curves(roc_results), clear_figure=True)
                    else:
                        st.info("ROC curves unavailable (no probability-capable models selected).")

                # ── Confusion matrices ──────────────────────────────────────
                st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
                n_cols  = min(3, len(results))
                cm_cols = st.columns(n_cols)
                for idx, res in enumerate(results):
                    with cm_cols[idx % n_cols]:
                        st.pyplot(
                            plot_confusion_matrix(y_test_arr, res.y_pred, res.name),
                            clear_figure=True,
                        )

                # ── Detailed classification reports ─────────────────────────
                st.markdown('<div class="section-header">Detailed Classification Reports</div>',
                            unsafe_allow_html=True)
                for res in results:
                    with st.expander(f"📋 {res.name} — Full Report"):
                        report = classification_report(
                            y_test_arr, res.y_pred,
                            target_names=["Negative (0)", "Positive (1)"],
                            output_dict=True,
                        )
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
with tab_tuning:
    st.markdown('<div class="section-header">GridSearchCV Hyperparameter Optimization</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="insight-box">
<strong>What is GridSearchCV?</strong><br>
Exhaustive search over a pre-defined parameter grid. Each combination is evaluated using
<strong>Stratified K-Fold Cross-Validation</strong> (class ratios preserved in every fold).
The optimal combination is selected by <strong>mean F1 score</strong> across all folds —
robust to class imbalance and dataset variance.
</div>
""", unsafe_allow_html=True)

    if not train_btn:
        st.info("Run **🚀 Train & Evaluate** first to build features, then tune here.")
    elif not (do_tune_lr or do_tune_rf or do_tune_mlp):
        st.warning("No tuning enabled. Select at least one model to tune in the sidebar.")
    else:
        try:
            df, y = cached_preprocess(data_path)
            X_tr_df, X_te_df, y_train, y_test = train_test_split(
                df, y, test_size=float(test_size), random_state=RANDOM_STATE, stratify=y,
            )
            X_train, X_test, vec2, sc2 = build_train_features(
                X_tr_df, X_te_df, feature_type, tfidf_cfg,
            )
            y_train_arr = y_train.to_numpy()
            y_test_arr  = y_test.to_numpy()

            tuning_results: List[ModelResult] = []

            if do_tune_lr:
                with st.spinner(f"GridSearchCV: Logistic Regression ({cv_splits}-fold)…"):
                    gs_lr = tune_logistic_regression(X_train, y_train_arr, cv_splits)
                st.success("✅ Logistic Regression tuning complete.")
                with st.expander("🔍 Best params — Logistic Regression"):
                    st.json(gs_lr.best_params_)
                    st.write(f"**Best CV F1:** `{gs_lr.best_score_:.4f}`")
                    r_lr = train_and_evaluate(
                        gs_lr.best_estimator_, "LR (Tuned)",
                        X_train, X_test, y_train_arr, y_test_arr, feature_type, tfidf_cfg,
                    )
                    tuning_results.append(r_lr)
                    st.dataframe(pd.DataFrame([r_lr.to_dict()]), use_container_width=True)
                    st.pyplot(plot_confusion_matrix(
                        y_test_arr, r_lr.y_pred, "Tuned Logistic Regression",
                    ), clear_figure=True)

            if do_tune_rf:
                with st.spinner(f"GridSearchCV: Random Forest ({cv_splits}-fold) — may take a few minutes…"):
                    gs_rf = tune_random_forest(X_train, y_train_arr, cv_splits)
                st.success("✅ Random Forest tuning complete.")
                with st.expander("🔍 Best params — Random Forest"):
                    st.json(gs_rf.best_params_)
                    st.write(f"**Best CV F1:** `{gs_rf.best_score_:.4f}`")
                    r_rf = train_and_evaluate(
                        gs_rf.best_estimator_, "RF (Tuned)",
                        X_train, X_test, y_train_arr, y_test_arr, feature_type, tfidf_cfg,
                    )
                    tuning_results.append(r_rf)
                    st.dataframe(pd.DataFrame([r_rf.to_dict()]), use_container_width=True)
                    st.pyplot(plot_confusion_matrix(
                        y_test_arr, r_rf.y_pred, "Tuned Random Forest",
                    ), clear_figure=True)

            if do_tune_mlp:
                with st.spinner(f"GridSearchCV: MLP ({cv_splits}-fold) — may take several minutes…"):
                    gs_mlp = tune_mlp(X_train, y_train_arr, cv_splits)
                st.success("✅ MLP tuning complete.")
                with st.expander("🔍 Best params — MLP"):
                    st.json(gs_mlp.best_params_)
                    st.write(f"**Best CV F1:** `{gs_mlp.best_score_:.4f}`")
                    r_mlp = train_and_evaluate(
                        gs_mlp.best_estimator_, "MLP (Tuned)",
                        X_train, X_test, y_train_arr, y_test_arr, feature_type, tfidf_cfg,
                    )
                    tuning_results.append(r_mlp)
                    st.dataframe(pd.DataFrame([r_mlp.to_dict()]), use_container_width=True)
                    st.pyplot(plot_confusion_matrix(
                        y_test_arr, r_mlp.y_pred, "Tuned MLP",
                    ), clear_figure=True)

            if tuning_results:
                st.markdown('<div class="section-header">Tuned Models — Comparison</div>',
                            unsafe_allow_html=True)
                st.pyplot(plot_metric_comparison(tuning_results), clear_figure=True)

            st.markdown("""
<div class="insight-box">
<strong>MLP Architecture Analysis (required comparison)</strong><br><br>
<code>(50,)</code> — Single hidden layer, 50 neurons. Lower parameter count.
Faster convergence, lower overfitting risk. Sufficient when TF-IDF already captures
most semantic signal. Preferred when training data is limited.<br><br>
<code>(100, 50)</code> — Two hidden layers with decreasing widths.
Greater representational capacity; can model higher-order feature interactions.
Most beneficial in Hybrid feature mode. Requires stronger L2 regularization
(<code>alpha</code>) and benefits most from GridSearchCV tuning to avoid overfitting.
</div>
""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Tuning failed: {e}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tab_clustering:
    st.markdown('<div class="section-header">KMeans Clustering — Unsupervised Sentiment Discovery</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="insight-box">
KMeans is applied on <strong>TF-IDF features only</strong> — labels are completely hidden.
Cluster-to-label alignment is measured using <strong>Adjusted Rand Index (ARI)</strong>:<br>
• ARI ≈ 0 → clusters are random with respect to true sentiment labels.<br>
• ARI > 0.1 → meaningful sentiment structure has emerged from unsupervised learning.<br><br>
PCA compresses the high-dimensional sparse matrix to 2D for visualization.
The explained variance ratio quantifies how much information is preserved in 2D.
Cluster composition analysis reveals the positive-review fraction per cluster.
</div>
""", unsafe_allow_html=True)

    if not cluster_btn:
        st.info("Select k values in the sidebar and click **Run Clustering**.")
    else:
        try:
            df, y = cached_preprocess(data_path)

            for k in k_values:
                st.markdown(f"#### k = {k} clusters")
                with st.spinner(f"Running KMeans (k={k})…"):
                    out = run_kmeans_clustering(df, y, tfidf_cfg, k)

                col_info, col_comp = st.columns([1, 2])
                with col_info:
                    st.metric("Adjusted Rand Index (ARI)", f"{out['ari']:.4f}")
                    st.metric("PCA Explained Variance (PC1)",
                              f"{out['explained_var'][0] * 100:.1f}%")
                    st.metric("PCA Explained Variance (PC2)",
                              f"{out['explained_var'][1] * 100:.1f}%")
                    st.metric("Samples clustered", f"{out['sampled']:,}")

                with col_comp:
                    comp_df = pd.DataFrame(out["composition"])
                    if not comp_df.empty:
                        comp_df["Positive Rate"] = comp_df["Positive Rate"].map(lambda x: f"{x:.1%}")
                        st.markdown("**Cluster composition (positive review rate)**")
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)

                col_km, col_true = st.columns(2)
                with col_km:
                    st.pyplot(
                        plot_clusters_2d(
                            out["X_2d"], out["cluster_labels"],
                            f"KMeans Clusters (k={k}) — PCA 2D",
                        ),
                        clear_figure=True,
                    )
                with col_true:
                    st.pyplot(
                        plot_clusters_2d(
                            out["X_2d"], out["true_labels"],
                            "True Sentiment Labels — PCA 2D",
                            label_names=["Negative", "Positive"],
                        ),
                        clear_figure=True,
                    )
                st.markdown("---")

        except Exception as e:
            st.error(f"❌ Clustering failed: {e}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-header">Real-Time Sentiment Prediction</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="insight-box">
The system uses the <strong>best-performing baseline model</strong> (by F1 score from the
last training run) to predict sentiment in real time. Run <strong>🚀 Train & Evaluate</strong>
first to initialize the prediction engine. The fitted vectorizer and scaler are reused —
no re-training occurs, ensuring consistent inference.
</div>
""", unsafe_allow_html=True)

    if "best_model" not in st.session_state or "vectorizer" not in st.session_state:
        st.warning("⚠️ No trained model found. Please run ** Train & Evaluate** first.")
    else:
        best_model = st.session_state["best_model"]
        vectorizer = st.session_state["vectorizer"]
        scaler     = st.session_state["scaler"]
        feat_type  = st.session_state["feature_type"]

        user_title   = st.text_input(
            "Review Title", value="Great product, value for money!",
        )
        user_content = st.text_area(
            "Review Content",
            value="Works exactly as described. Build quality is solid and delivery was fast. Highly recommended!",
            height=120,
        )

        num_inputs: Dict[str, float] = {
            "discounted_price": np.nan,
            "actual_price":     np.nan,
            "rating_count":     np.nan,
            "discount_pct":     np.nan,
        }
        if feat_type == "Hybrid":
            st.markdown("**Optional: Product pricing context (Hybrid mode)**")
            c1, c2, c3, c4 = st.columns(4)
            num_inputs["discounted_price"] = parse_money(c1.text_input("Discounted Price", "₹999"))
            num_inputs["actual_price"]     = parse_money(c2.text_input("Actual Price",     "₹1,499"))
            num_inputs["rating_count"]     = parse_count(c3.text_input("Rating Count",     "1,234"))
            try:
                num_inputs["discount_pct"] = float(c4.text_input("Discount %", "33.3") or 0)
            except ValueError:
                num_inputs["discount_pct"] = np.nan

        predict_btn = st.button("⚡ Predict Sentiment", type="primary")

        if predict_btn:
            try:
                row    = {"review_title": user_title, "review_content": user_content, **num_inputs}
                df_one = pd.DataFrame([row])
                X_one  = build_inference_features(df_one, feat_type, vectorizer, scaler)

                pred = int(best_model.predict(X_one)[0])

                confidence_str = ""
                try:
                    if hasattr(best_model, "predict_proba"):
                        proba          = best_model.predict_proba(X_one)[0]
                        conf           = float(proba[pred])
                        confidence_str = f"  |  Confidence: **{conf:.1%}**"
                    elif hasattr(best_model, "decision_function"):
                        score          = float(best_model.decision_function(X_one)[0])
                        confidence_str = f"  |  Decision score: **{score:.3f}**"
                except Exception:
                    pass

                label = "Positive ✓" if pred == 1 else "Negative ✕"
                color = PALETTE["positive"] if pred == 1 else PALETTE["negative"]

                st.markdown(
                    f'<div style="background:#1e293b; border:2px solid {color}; border-radius:10px; '
                    f'padding:1.5rem 2rem; margin-top:1rem;">'
                    f'<span style="font-size:2rem; font-weight:700; color:{color};">{label}</span>'
                    f'<span style="color:#94a3b8; margin-left:1rem; font-size:0.9rem;">{confidence_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Model: **{type(best_model).__name__}**  |  "
                    f"Features: **{feat_type}**  |  "
                    f"TF-IDF: **{tfidf_cfg}**"
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.82rem;">'
    "Intelligent Amazon Review Analyzer · Machine Learning Course · Laboratoric Course · "
    f"scikit-learn + Streamlit · Random State: {RANDOM_STATE}"
    "</p>",
    unsafe_allow_html=True,
)