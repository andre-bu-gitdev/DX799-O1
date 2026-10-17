"""
prep.py - shared, model-agnostic helpers for every model notebook.

Nothing here is tied to a specific model: you pass in any scikit-learn
estimator/Pipeline and any parameter grid from the notebook. Every function
returns DataFrames / matplotlib Axes, so you can keep customizing in the notebook.

-------------------------------------------------------------------------------
TYPICAL NOTEBOOK
-------------------------------------------------------------------------------
    from prep import *
    from sklearn.linear_model import Lasso

    X_train, X_test, y_train, y_test, meta = load_model_data()

    # 1) build + tune (grid lives in the notebook)
    pipe = make_pipeline(*preprocess(scale=True), Lasso(max_iter=50_000))
    grid = {"lasso__alpha": np.logspace(-4, 1, 30)}
    search = tune(pipe, grid, X_train, y_train)

    # 2) see how the hyperparameter changes performance
    tuning_table(search)
    plot_tuning(search, "lasso__alpha")

    # 3) score the best model, look at it, save it for the final comparison
    best = search.best_estimator_
    score(best, X_train, X_test, y_train, y_test)
    plot_forecast(best, X_train, X_test, y_train, y_test)
    record("Lasso", best, X_train, X_test, y_train, y_test, search=search)

    # 4) any notebook (or a final one): compare every recorded model
    compare()
    plot_forecasts()

-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------
  Data            load_model_data, load_full, make_binary_target
  Preprocessing   preprocess
  Tuning          tune, tuning_table, plot_tuning, plot_tuning_heatmap
  Scoring         score, score_classifier, plot_forecast, walk_forward
  Comparison      record, load_results, compare, plot_forecasts
  Export (Excel)  export_predictions, export_recorded
  Explain         feature_weights, plot_feature_weights      what the fitted model put on each feature
                  weight_path, plot_weight_path              how weights change with a hyperparameter (e.g. Lasso alpha)
                  permutation_table, plot_permutation        model-agnostic importance, train vs test (overfit check)
                  weight_stability, plot_weight_stability    do weights hold up across time folds? (overfit check)
"""

import glob
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     TimeSeriesSplit)
from sklearn.pipeline import make_pipeline  # re-exported for notebooks
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/workspaces/DX799-O1/model_data"     # <- your path
RESULTS_DIR = os.path.join(DATA_DIR, "results")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
COVID = ("2020-01-01", "2021-12-31")

__all__ = ["np", "pd", "plt", "make_pipeline", "TimeSeriesSplit",
           "load_model_data", "load_full", "make_binary_target", "preprocess",
           "tune", "tuning_table", "plot_tuning", "plot_tuning_heatmap",
           "score", "score_classifier", "plot_forecast", "walk_forward",
           "record", "load_results", "compare", "plot_forecasts",
           "export_predictions", "export_recorded",
           "feature_weights", "plot_feature_weights", "weight_path", "plot_weight_path",
           "permutation_table", "plot_permutation", "weight_stability", "plot_weight_stability"]


# =============================================================================
# DATA
# =============================================================================
def load_model_data(folder=DATA_DIR):
    """Load the unscaled, lagged train/test sets saved by the EDA notebook."""
    X_train = pd.read_pickle(f"{folder}/X_train.pkl")
    X_test = pd.read_pickle(f"{folder}/X_test.pkl")
    y_train = pd.read_pickle(f"{folder}/y_train.pkl")
    y_test = pd.read_pickle(f"{folder}/y_test.pkl")
    with open(f"{folder}/meta.json") as f:
        meta = json.load(f)
    y_train, y_test = y_train.rename(meta["target"]), y_test.rename(meta["target"])
    print(f"train {X_train.index.min().date()} to {X_train.index.max().date()}  {X_train.shape}")
    print(f"test  {X_test.index.min().date()} to {X_test.index.max().date()}  {X_test.shape}")
    return X_train, X_test, y_train, y_test, meta


def load_full(folder=DATA_DIR):
    """Full lagged table (train + test together) for walk-forward evaluation."""
    data = pd.read_pickle(f"{folder}/data_lagged.pkl")
    with open(f"{folder}/meta.json") as f:
        meta = json.load(f)
    return data.drop(columns="y"), data["y"].rename(meta["target"])


def make_binary_target(y_train, y_test, quantile=0.5):
    """
    Yes/no version of the target for classifiers (logistic regression, SVC...).
    Threshold comes from the TRAINING period only (default: training median).
    Returns y_train_bin, y_test_bin, threshold.
    """
    thr = float(y_train.quantile(quantile))
    print(f"threshold = {thr:.3f} (training {quantile:.0%} quantile); "
          f"share above: train {(y_train > thr).mean():.0%}, test {(y_test > thr).mean():.0%}")
    return (y_train > thr).astype(int), (y_test > thr).astype(int), thr


# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess(scale=True, impute="median"):
    """
    Standard first steps for make_pipeline(*preprocess(...), model).
      scale=True  -> SimpleImputer + StandardScaler   (linear, PCR/PLS, SVM, logistic)
      scale=False -> SimpleImputer only               (trees / random forests)
    Add anything else (PolynomialFeatures, PCA, ...) after it in the notebook:
      make_pipeline(*preprocess(scale=False), PolynomialFeatures(2), StandardScaler(), Ridge())
    """
    steps = [SimpleImputer(strategy=impute)]
    if scale:
        steps.append(StandardScaler())
    return steps


# =============================================================================
# TUNING
# =============================================================================
def tune(estimator, param_grid, X, y, n_splits=5, scoring=None, search="grid",
         n_iter=40, random_state=0, cv=None, **kwargs):
    """
    Hyperparameter search with time-ordered folds (TimeSeriesSplit), never shuffled.

    estimator  : any sklearn estimator or Pipeline
    param_grid : dict, e.g. {"ridge__alpha": np.logspace(-3, 3, 25)}
                 (Pipeline step names are the lowercase class names: ridge, lasso,
                  randomforestregressor, svr, logisticregression, pca, ...)
    scoring    : default RMSE for regressors, accuracy for classifiers;
                 any sklearn scoring string works ("neg_mean_absolute_error", "roc_auc", ...)
    search     : "grid" (try every combination) or "random" (n_iter random combinations)
    kwargs     : passed straight to GridSearchCV / RandomizedSearchCV
    """
    if scoring is None:
        scoring = "accuracy" if is_classifier(estimator) else "neg_root_mean_squared_error"
    cv = cv or TimeSeriesSplit(n_splits=n_splits)
    common = dict(cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1, **kwargs)
    if search == "grid":
        s = GridSearchCV(estimator, param_grid, **common)
    else:
        s = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter,
                               random_state=random_state, **common)
    s.fit(X, y)
    name, sign = _metric_name(scoring)
    print(f"best params: {s.best_params_}")
    print(f"best CV {name}: {sign * s.best_score_:.4f}")
    return s


def _metric_name(scoring):
    """'neg_root_mean_squared_error' -> ('root_mean_squared_error', -1)."""
    s = scoring if isinstance(scoring, str) else "score"
    return (s[4:], -1) if s.startswith("neg_") else (s, 1)


def tuning_table(search, top=None):
    """Every combination tried: params, CV score (+/- sd), train score, overfit gap, rank."""
    r = pd.DataFrame(search.cv_results_)
    name, sign = _metric_name(search.scoring)
    params = [c for c in r.columns if c.startswith("param_")]
    out = r[params].copy()
    out.columns = [c[len("param_"):] for c in params]
    out[f"cv_{name}"] = sign * r["mean_test_score"]
    out[f"cv_{name}_sd"] = r["std_test_score"]
    out[f"train_{name}"] = sign * r["mean_train_score"]
    out["gap (cv - train)"] = out[f"cv_{name}"] - out[f"train_{name}"]
    out["rank"] = r["rank_test_score"]
    out = out.sort_values("rank").reset_index(drop=True)
    return out.head(top) if top else out


def _short(p):
    """'randomforestregressor__max_depth' -> 'max_depth' (display only)."""
    return p.split("__")[-1] if p else p


def _lower_is_better(name):
    return any(k in name for k in ("error", "loss", "deviance"))


def _unused(series):
    """True where a parameter does not apply to that row (e.g. gamma for a linear kernel).
    A deliberate None (e.g. max_depth=None) is kept."""
    return series.map(lambda v: isinstance(v, float) and np.isnan(v))


def _profile(table, search, keep, how="best"):
    """
    Reduce the table to one row per combination of the `keep` parameters.
      how="best"  : for each combination, the best CV score over all OTHER parameters
                    (works even when parameters only apply to some rows, e.g. gamma for rbf only)
      how="fixed" : other parameters held at the overall best values
    """
    name, _ = _metric_name(search.scoring)
    cv_col = f"cv_{name}"
    keep = [k for k in keep if k]
    t = table.copy()
    for k in keep:                       # only rows where the plotted parameters actually apply
        t = t[~_unused(t[k])]
    if how == "fixed":
        for p, v in search.best_params_.items():
            if p not in keep and p in t.columns:
                t = t[(t[p].map(str) == str(v)) | t[p].isna()]
        return t
    t["_key"] = t[keep].apply(lambda r: tuple(map(str, r)), axis=1)
    idx = (t.groupby("_key")[cv_col].idxmin() if _lower_is_better(name)
           else t.groupby("_key")[cv_col].idxmax())
    return t.loc[idx].drop(columns="_key")


def plot_tuning(search, param, by=None, logx=None, ax=None, others="best"):
    """
    Validation curve: CV score vs one hyperparameter, train score dashed so the
    overfitting gap is visible.
    by     : optional second hyperparameter -> one line per value (e.g. "svr__kernel")
    others : "best"  -> each point uses the best setting of all other hyperparameters
             "fixed" -> other hyperparameters held at the overall best values
    """
    t = tuning_table(search)
    name, _ = _metric_name(search.scoring)
    cv_col, sd_col, tr_col = f"cv_{name}", f"cv_{name}_sd", f"train_{name}"
    t = _profile(t, search, keep=[param, by], how=others)
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]

    numeric = pd.api.types.is_numeric_dtype(pd.to_numeric(t[param], errors="coerce").dropna()) \
        and pd.to_numeric(t[param], errors="coerce").notna().all()
    groups = [(None, t)] if by is None else list(t.groupby(t[by].map(str)))
    for label, g in groups:
        g = g.copy()
        g["_x"] = pd.to_numeric(g[param]) if numeric else g[param].map(str)
        g = g.sort_values("_x")
        lbl = "CV" if label is None else f"{_short(by)}={label}"
        line, = ax.plot(g["_x"], g[cv_col], marker="o", ms=3, label=f"{lbl} (validation)")
        ax.fill_between(g["_x"], g[cv_col] - g[sd_col], g[cv_col] + g[sd_col],
                        color=line.get_color(), alpha=0.15)
        ax.plot(g["_x"], g[tr_col], linestyle="--", color=line.get_color(), alpha=0.7,
                label=f"{lbl} (train)")

    best = search.best_params_.get(param)
    if best is not None:
        best_lbl = f"{best:.3g}" if isinstance(best, (int, float, np.number)) else best
        ax.axvline(best if numeric else str(best), color="grey", linestyle=":", label=f"best = {best_lbl}")
    if numeric and (logx or (logx is None and t[param].astype(float).min() > 0
                             and t[param].astype(float).max() / t[param].astype(float).min() >= 100)):
        ax.set_xscale("log")
    ax.set_xlabel(_short(param))
    ax.set_ylabel(name.replace("_", " "))
    ax.set_title(f"Tuning: {_short(param)}" + (f" by {_short(by)}" if by else ""))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_tuning_heatmap(search, param_x, param_y, ax=None, fmt=".3f", others="best"):
    """CV score for every combination of two hyperparameters (others: "best" or "fixed", see plot_tuning)."""
    t = tuning_table(search)
    name, _ = _metric_name(search.scoring)
    t = _profile(t, search, keep=[param_x, param_y], how=others)
    t[param_x], t[param_y] = t[param_x].map(str), t[param_y].map(str)   # keeps None / text values
    piv = t.pivot_table(index=param_y, columns=param_x, values=f"cv_{name}")
    order = lambda vals: sorted(vals, key=lambda v: (pd.to_numeric(v, errors="coerce"), v))
    piv = piv.reindex(index=order(piv.index), columns=order(piv.columns))
    ax = ax or plt.subplots(figsize=(2.5 + 0.9 * piv.shape[1], 1.5 + 0.5 * piv.shape[0]))[1]
    lower_better = _lower_is_better(name)
    im = ax.imshow(piv.values, cmap="viridis_r" if lower_better else "viridis", aspect="auto")
    short = lambda v: f"{float(v):.3g}" if pd.notna(pd.to_numeric(v, errors="coerce")) else v
    ax.set_xticks(range(piv.shape[1]), [short(v) for v in piv.columns], rotation=45)
    ax.set_yticks(range(piv.shape[0]), [short(v) for v in piv.index])
    lo, hi = np.nanmin(piv.values), np.nanmax(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v):
                frac = 0.5 if hi == lo else (v - lo) / (hi - lo)
                light = frac < 0.5 if lower_better else frac > 0.5     # light cells get dark text
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                        color="black" if light else "white")
    ax.set_xlabel(_short(param_x))
    ax.set_ylabel(_short(param_y))
    ax.set_title(f"CV {name.replace('_', ' ')}")
    plt.colorbar(im, ax=ax)
    ax.figure.tight_layout()
    return ax


# =============================================================================
# SCORING
# =============================================================================
def _pred(model, X):
    """predict() flattened to 1-D (PLSRegression and some others return a column)."""
    return np.ravel(model.predict(X))


def _regression_metrics(actual, pred, prev=None):
    ok = actual.notna() & pred.notna()
    a, p = actual[ok], pred[ok]
    out = {"RMSE": mean_squared_error(a, p) ** 0.5, "MAE": mean_absolute_error(a, p),
           "R2": r2_score(a, p)}
    if prev is not None:   # did we get the direction of change (up/down vs last quarter) right?
        pv = prev[ok]
        out["DirAcc"] = float((np.sign(p - pv) == np.sign(a - pv)).mean())
    out["n"] = int(len(a))
    return out


def _lag_col(y, X, target_lag_col):
    col = target_lag_col or f"{y.name}_L1"
    if col not in X.columns:
        raise KeyError(f"{col} not in X; pass target_lag_col='<target>_L1'")
    return col


def score(model, X_train, X_test, y_train, y_test, target_lag_col=None, covid=COVID):
    """
    Regression scorecard: train vs test vs naive ('next quarter = this quarter'),
    plus test without 2020-21.  DirAcc = share of quarters where the predicted
    direction of change matched the actual direction (an 'accuracy' for regression).
    """
    lag = _lag_col(y_train, X_test, target_lag_col)
    p_tr = pd.Series(_pred(model, X_train), index=X_train.index)
    p_te = pd.Series(_pred(model, X_test), index=X_test.index)
    naive = X_test[lag]
    keep = ~((y_test.index >= covid[0]) & (y_test.index <= covid[1]))
    t = pd.DataFrame({
        "model - train (in-sample)": _regression_metrics(y_train, p_tr, X_train[lag]),
        "model - test": _regression_metrics(y_test, p_te, naive),
        "naive - test": _regression_metrics(y_test, naive),
        "model - test excl. 2020-21": _regression_metrics(y_test[keep], p_te[keep], naive[keep]),
        "naive - test excl. 2020-21": _regression_metrics(y_test[keep], naive[keep]),
    }).T
    return t.astype(float).round(3)


def score_classifier(model, X_train, X_test, y_train_bin, y_test_bin):
    """Classification scorecard (accuracy, precision, recall, F1, ROC AUC) for train and test."""
    rows = {}
    for label, X, y in [("train (in-sample)", X_train, y_train_bin), ("test", X_test, y_test_bin)]:
        pred = model.predict(X)
        r = {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
             "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
        if hasattr(model, "predict_proba") and len(set(y)) > 1:
            r["ROC_AUC"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
        r["n"] = len(y)
        rows[label] = r
    rows["always-majority - test"] = {"Accuracy": max(y_test_bin.mean(), 1 - y_test_bin.mean()), "n": len(y_test_bin)}
    return pd.DataFrame(rows).T.astype(float).round(3)


def plot_forecast(model, X_train, X_test, y_train, y_test, target_lag_col=None, title=None, ax=None):
    """Actual vs fitted (train) vs forecast (test) vs naive, for one model."""
    lag = _lag_col(y_train, X_test, target_lag_col)
    ax = ax or plt.subplots(figsize=(12, 4.5))[1]
    actual = pd.concat([y_train, y_test])
    ax.plot(actual.index, actual, color="black", lw=1.4, label="Actual")
    ax.plot(X_train.index, _pred(model, X_train), color="tab:blue", ls="--", lw=1.1, label="Fit (train)")
    ax.plot(X_test.index, _pred(model, X_test), color="tab:orange", lw=1.6, label="Forecast (test)")
    ax.plot(X_test.index, X_test[lag], color="grey", ls=":", lw=1.1, label="Naive (last quarter)")
    ax.axvline(X_test.index.min(), color="grey", lw=1)
    ax.set_title(title or f"{y_train.name}: actual vs one-quarter-ahead forecast")
    ax.set_ylabel(y_train.name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return ax


def walk_forward(model, X_all, y_all, test_start, refit_every=1, min_train=None):
    """
    Realistic evaluation: for each test quarter, refit on everything before it, predict it.
    refit_every=4 refits once a year (faster for slow models). Returns predictions (Series).
    Use with load_full():  X_all, y_all = load_full()
    """
    idx = y_all.loc[test_start:].index
    preds, fitted = {}, None
    for i, q in enumerate(idx):
        if fitted is None or i % refit_every == 0:
            tr = y_all.index < q
            if min_train and tr.sum() < min_train:
                continue
            fitted = clone(model).fit(X_all[tr], y_all[tr])
        preds[q] = _pred(fitted, X_all.loc[[q]])[0]
    return pd.Series(preds, name="walk_forward_pred")


# =============================================================================
# COMPARISON ACROSS NOTEBOOKS
# =============================================================================
def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def record(name, model, X_train, X_test, y_train, y_test, search=None, notes="",
           target_lag_col=None, folder=RESULTS_DIR, extra=None):
    """
    Save one model's results so any notebook can compare them later.
    Works for regressors (continuous y) and classifiers (binary y).
    extra : optional dict of anything else you want to keep (e.g. {"n_features": 12}).
    Re-recording the same name overwrites it.
    """
    os.makedirs(folder, exist_ok=True)
    clf = is_classifier(model)
    res = {"name": name, "type": "classifier" if clf else "regressor", "notes": notes,
           "saved": datetime.now().strftime("%Y-%m-%d %H:%M"), "extra": extra or {},
           "best_params": getattr(search, "best_params_", None),
           "cv_score": None, "cv_metric": None}
    if search is not None:
        m, sign = _metric_name(search.scoring)
        res["cv_score"], res["cv_metric"] = float(sign * search.best_score_), m
    if clf:
        res["metrics"] = score_classifier(model, X_train, X_test, y_train, y_test)
        res["pred_test"] = pd.Series(_pred(model, X_test), index=X_test.index)
        res["actual_test"] = y_test
    else:
        res["metrics"] = score(model, X_train, X_test, y_train, y_test, target_lag_col)
        res["pred_train"] = pd.Series(_pred(model, X_train), index=X_train.index)
        res["pred_test"] = pd.Series(_pred(model, X_test), index=X_test.index)
        res["actual_train"], res["actual_test"] = y_train, y_test
        res["naive_test"] = X_test[_lag_col(y_train, X_test, target_lag_col)]
    with open(os.path.join(folder, f"{_slug(name)}.pkl"), "wb") as f:
        pickle.dump(res, f)
    print(f"recorded '{name}' -> {folder}")
    return res["metrics"]


def load_results(folder=RESULTS_DIR):
    """All recorded results as {name: dict}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.pkl"))):
        with open(path, "rb") as f:
            r = pickle.load(f)
        out[r["name"]] = r
    return out


def compare(metric="RMSE", threshold=None, results=None, folder=RESULTS_DIR, plot=True):
    """
    One table + chart comparing every recorded model.

    Regressors: test RMSE / MAE / R2 / DirAcc, test excl. 2020-21, train RMSE,
                overfit gap, best CV score and params, plus a naive-baseline row.
    threshold : optional cut-off (e.g. from make_binary_target) -> adds a classification
                'Accuracy' column for EVERY model (regressors: forecast > threshold),
                so regressors and classifiers can be compared on one scale.
    """
    results = results or load_results(folder)
    if not results:
        print("no recorded results yet - use record(...) first")
        return None
    rows, naive_row = {}, None
    for name, r in results.items():
        m = r["metrics"]
        row = {"type": r["type"]}
        if r["type"] == "regressor":
            row.update({f"test {k}": m.loc["model - test", k] for k in ["RMSE", "MAE", "R2", "DirAcc"]})
            row["test RMSE excl. 2020-21"] = m.loc["model - test excl. 2020-21", "RMSE"]
            row["train RMSE"] = m.loc["model - train (in-sample)", "RMSE"]
            row["overfit gap (test - train RMSE)"] = row["test RMSE"] - row["train RMSE"]
            if naive_row is None:
                naive_row = {"type": "baseline",
                             "test RMSE": m.loc["naive - test", "RMSE"], "test MAE": m.loc["naive - test", "MAE"],
                             "test R2": m.loc["naive - test", "R2"],
                             "test RMSE excl. 2020-21": m.loc["naive - test excl. 2020-21", "RMSE"]}
                if threshold is not None:
                    a, n = r["actual_test"], r["naive_test"]
                    naive_row["Accuracy (vs threshold)"] = float(((n > threshold) == (a > threshold)).mean())
            if threshold is not None:
                a, p = r["actual_test"], r["pred_test"]
                row["Accuracy (vs threshold)"] = float(((p > threshold) == (a > threshold)).mean())
        else:
            row["Accuracy (vs threshold)"] = m.loc["test", "Accuracy"]
            if "ROC_AUC" in m.columns:
                row["test ROC_AUC"] = m.loc["test", "ROC_AUC"]
        row["CV score"] = r["cv_score"]
        row["CV metric"] = r["cv_metric"]
        row["best params"] = r["best_params"]
        row["saved"] = r["saved"]
        rows[name] = row
    if naive_row:
        rows["Naive (last quarter)"] = naive_row
    table = pd.DataFrame(rows).T
    sort_col = f"test {metric}" if f"test {metric}" in table.columns else None
    if sort_col:
        table = table.sort_values(sort_col, ascending=metric != "R2" and metric != "DirAcc")

    if plot:
        cols = [c for c in [f"test {metric}", "test RMSE excl. 2020-21" if metric == "RMSE" else None,
                            "train RMSE" if metric == "RMSE" else None] if c and c in table.columns]
        regs = table[table["type"].isin(["regressor", "baseline"])]
        n_panels = 1 + (threshold is not None)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 0.45 * len(table) + 2), squeeze=False)
        if cols and len(regs):
            regs[cols].astype(float).iloc[::-1].plot.barh(ax=axes[0, 0])
            axes[0, 0].set_title(f"{metric} by model (lower is better)" if metric in ("RMSE", "MAE")
                                 else f"{metric} by model")
            axes[0, 0].grid(axis="x", alpha=0.3)
        if threshold is not None:
            acc = table["Accuracy (vs threshold)"].dropna().astype(float).sort_values()
            acc.plot.barh(ax=axes[0, -1], color="tab:green")
            axes[0, -1].set_xlim(0, 1)
            axes[0, -1].set_title(f"Accuracy: above/below {threshold:.2f} (higher is better)")
            axes[0, -1].grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()
    return table


def plot_forecasts(names=None, start=None, results=None, folder=RESULTS_DIR, ax=None):
    """Test-period forecasts of several recorded regressors on one chart, with actual and naive."""
    results = results or load_results(folder)
    regs = {k: v for k, v in results.items() if v["type"] == "regressor" and (names is None or k in names)}
    if not regs:
        print("no recorded regressors to plot")
        return None
    ax = ax or plt.subplots(figsize=(12, 5))[1]
    first = next(iter(regs.values()))
    a = first["actual_test"].loc[start:] if start else first["actual_test"]
    ax.plot(a.index, a, color="black", lw=1.8, label="Actual")
    n = first["naive_test"].loc[a.index]
    ax.plot(n.index, n, color="grey", ls=":", lw=1.2, label="Naive")
    for name, r in regs.items():
        p = r["pred_test"].loc[a.index]
        ax.plot(p.index, p, lw=1.3, label=f"{name} (RMSE {r['metrics'].loc['model - test', 'RMSE']:.3f})")
    ax.set_title("Test-period forecasts by model")
    ax.set_ylabel(a.name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return ax


# =============================================================================
# EXPORT TO CSV (open in Excel)
# =============================================================================
def export_predictions(model, X_train, X_test, y_train, y_test, name="model",
                       include_features=False, target_lag_col=None, folder=EXPORT_DIR):
    """
    One CSV, one row per quarter: date, set (train/test), actual, predicted, error,
    abs error, naive forecast. include_features=True adds every X column too.
    Works for regressors and classifiers (binary y). Returns the DataFrame.
    """
    os.makedirs(folder, exist_ok=True)
    parts = []
    for label, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        d = pd.DataFrame({"set": label, "actual": y, "predicted": _pred(model, X)}, index=X.index)
        d["error (pred - actual)"] = d["predicted"] - d["actual"]
        d["abs error"] = d["error (pred - actual)"].abs()
        lag = target_lag_col or f"{y.name}_L1"
        if not is_classifier(model) and lag in X.columns:
            d["naive (last quarter)"] = X[lag]
        if include_features:
            d = d.join(X)
        parts.append(d)
    out = pd.concat(parts)
    out.index.name = "date"
    path = os.path.join(folder, f"{_slug(name)}_predictions.csv")
    out.to_csv(path)
    print(f"saved {len(out)} rows -> {path}")
    return out


def export_recorded(names=None, folder=RESULTS_DIR, out_folder=EXPORT_DIR):
    """
    Everything you recorded, side by side, in two CSVs:
      all_models_test_predictions.csv : date, actual, naive, one column per model (test period)
      all_models_metrics.csv          : the compare() table
    """
    os.makedirs(out_folder, exist_ok=True)
    results = load_results(folder)
    if names:
        results = {k: v for k, v in results.items() if k in names}
    regs = {k: v for k, v in results.items() if v["type"] == "regressor"}
    if regs:
        first = next(iter(regs.values()))
        wide = pd.DataFrame({"actual": first["actual_test"], "naive (last quarter)": first["naive_test"]})
        for k, v in regs.items():
            wide[k] = v["pred_test"]
        wide.index.name = "date"
        p1 = os.path.join(out_folder, "all_models_test_predictions.csv")
        wide.to_csv(p1)
        print(f"saved -> {p1}")
    table = compare(results=results, plot=False)
    if table is not None:
        p2 = os.path.join(out_folder, "all_models_metrics.csv")
        table.to_csv(p2)
        print(f"saved -> {p2}")
    return table


# =============================================================================
# EXPLAIN: HOW EACH MODEL TREATS EACH FEATURE
# =============================================================================
import re
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def _base_var(name):
    """'unemp_rate_L3' -> 'unemp_rate' (interaction terms like 'a_L1 b_L2' are left as-is)."""
    return name if " " in name else re.sub(r"_L\d+$", "", name)


def _final_and_names(model, X):
    """Final estimator + the feature names it actually sees (after selection / polynomial terms)."""
    if isinstance(model, Pipeline):
        final = model[-1]
        if len(model.steps) > 1:
            try:
                names = list(model[:-1].get_feature_names_out())
            except Exception:
                names = [f"f{i}" for i in range(np.ravel(getattr(final, "coef_", getattr(final, "feature_importances_", []))).size)]
        else:
            names = list(X.columns)
        return final, names
    return model, list(X.columns)


def feature_weights(model, X, by_variable=False):
    """
    What a FITTED model put on each feature.
      Linear models (Lasso, Ridge, ElasticNet, LinearRegression, Logistic, linear SVR/SVC, PLS):
          coefficient = change in the prediction per 1 sd of the feature (when the Pipeline scales).
          0 = feature dropped (Lasso / Elastic Net / selection).
      PCR (PCA -> linear): coefficients mapped back to the original features.
      Trees / random forests: impurity-based importance (always >= 0, sums to 1; no direction).
      Models with neither (e.g. rbf/poly-kernel SVR): use permutation_table() instead.
    by_variable=True adds up |weights| across lags (unemp_rate_L1..L4 -> unemp_rate).
    """
    final, names = _final_and_names(model, X)
    kind = None
    if hasattr(final, "coef_"):
        w = np.ravel(final.coef_)
        kind = "coefficient"
        steps = dict(model.named_steps) if isinstance(model, Pipeline) else {}
        pca = next((v for v in steps.values() if v.__class__.__name__ == "PCA"), None)
        if pca is not None and len(w) == pca.n_components_:        # PCR: back to original features
            w = pca.components_.T @ w
            names = list(X.columns)
            kind = "coefficient (PCR, mapped to features)"
    elif hasattr(final, "feature_importances_"):
        w = np.ravel(final.feature_importances_)
        kind = "importance"
    else:
        raise TypeError(f"{final.__class__.__name__} has no coef_ or feature_importances_ "
                        "(e.g. rbf/poly-kernel SVM) - use permutation_table(model, ...) instead")
    if len(names) != len(w):
        names = [f"f{i}" for i in range(len(w))]
    t = pd.DataFrame({"feature": names, "weight": w})
    # features removed by selection (SequentialFeatureSelector) appear with weight 0
    if kind == "coefficient" and set(names) <= set(X.columns) and len(names) < X.shape[1]:
        t = pd.DataFrame({"feature": X.columns}).merge(t, how="left").fillna({"weight": 0.0})
    t["abs weight"] = t["weight"].abs()
    t["used"] = t["abs weight"] > 1e-12
    t["variable"] = t["feature"].map(_base_var)
    t.attrs["kind"] = kind
    if by_variable:
        g = t.groupby("variable").agg(**{"total abs weight": ("abs weight", "sum"),
                                         "lags used": ("used", "sum"),
                                         "lags available": ("used", "size")})
        return g.sort_values("total abs weight", ascending=False)
    return t.sort_values("abs weight", ascending=False).reset_index(drop=True)


def plot_feature_weights(model, X, top=20, by_variable=False, ax=None, title=None):
    """Bar chart of feature_weights(). Red = pushes forecast up, blue = down, grey = dropped."""
    t = feature_weights(model, X, by_variable=by_variable)
    kind = "total |weight| across lags" if by_variable else t.attrs.get("kind", "weight")
    if by_variable:
        vals, labels, colors = t["total abs weight"].head(top), t.index[:top], "tab:purple"
    else:
        h = t.head(top)
        vals, labels = h["weight"], h["feature"]
        colors = ["lightgrey" if not u else ("tab:red" if v > 0 else "tab:blue") for v, u in zip(h["weight"], h["used"])]
    n = len(vals)
    ax = ax or plt.subplots(figsize=(8, 0.32 * n + 1.2))[1]
    ax.barh(list(labels)[::-1], list(vals)[::-1], color=colors[::-1] if isinstance(colors, list) else colors)
    ax.axvline(0, color="black", lw=0.8)
    if not by_variable and "used" in t:
        dropped = int((~t["used"]).sum())
        if dropped:
            ax.text(0.99, 0.01, f"{dropped} of {len(t)} features set to 0", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_title(title or f"Feature weights ({kind})")
    ax.grid(axis="x", alpha=0.3)
    return ax


def weight_path(estimator, param, values, X, y):
    """
    Refit the model at each value of one hyperparameter and record every feature's weight.
    e.g. weight_path(pipe, "lasso__alpha", np.logspace(-4, 0, 40), X_train, y_train)
    Returns a DataFrame: rows = parameter values, columns = features.
    """
    rows = {}
    for v in values:
        m = clone(estimator).set_params(**{param: v}).fit(X, y)
        t = feature_weights(m, X)
        rows[v] = t.set_index("feature")["weight"]
    out = pd.DataFrame(rows).T
    out.index.name = param
    return out


def plot_weight_path(path, best=None, top=10, logx=True, ax=None):
    """
    One line per feature across the hyperparameter (e.g. Lasso alpha): watch weights shrink
    and hit 0. The `top` features (largest weight anywhere on the path) are labelled; the rest grey.
    Also shows how many features are still in use at each value.
    """
    ax = ax or plt.subplots(figsize=(11, 5))[1]
    order = path.abs().max().sort_values(ascending=False).index
    for f in order[top:]:
        ax.plot(path.index, path[f], color="lightgrey", lw=0.8)
    for f in order[:top]:
        ax.plot(path.index, path[f], lw=1.6, label=f)
    if best is not None:
        ax.axvline(best, color="black", ls=":", label=f"chosen = {best:.3g}")
    ax.axhline(0, color="black", lw=0.6)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(path.index.name)
    ax.set_ylabel("weight (per 1 sd of feature)")
    ax.set_title("How each feature's weight changes with the hyperparameter")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.10, 0.5))
    ax2 = ax.twinx()
    ax2.plot(path.index, (path.abs() > 1e-12).sum(axis=1), color="black", lw=1, ls="--", alpha=0.5)
    ax2.set_ylabel("features in use (dashed)")
    return ax


def permutation_table(model, X_train, y_train, X_test, y_test, n_repeats=30, scoring=None, random_state=0):
    """
    Works for ANY fitted model (incl. rbf SVM, random forests, polynomial pipelines).
    Shuffles one feature at a time and measures how much the score gets worse.
    Done on train AND test: a feature that matters on train but not on test is a sign
    the model learned noise from it (overfitting).
    """
    if scoring is None:
        scoring = "accuracy" if is_classifier(model) else "neg_root_mean_squared_error"
    res = {}
    for label, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        r = permutation_importance(model, X, y, n_repeats=n_repeats, scoring=scoring,
                                   random_state=random_state, n_jobs=-1)
        res[f"{label} importance"] = r.importances_mean
        res[f"{label} sd"] = r.importances_std
    t = pd.DataFrame(res, index=X_train.columns)
    t["train - test"] = t["train importance"] - t["test importance"]
    t.attrs["scoring"] = scoring
    return t.sort_values("test importance", ascending=False)


def plot_permutation(table, top=20, ax=None):
    """Train vs test permutation importance side by side."""
    t = table.head(top)
    ax = ax or plt.subplots(figsize=(8, 0.35 * len(t) + 1.2))[1]
    y = np.arange(len(t))[::-1]
    ax.barh(y + 0.2, t["train importance"], height=0.4, xerr=t["train sd"], label="train", color="tab:blue", alpha=0.7)
    ax.barh(y - 0.2, t["test importance"], height=0.4, xerr=t["test sd"], label="test", color="tab:orange")
    ax.set_yticks(y, t.index)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"drop in {table.attrs.get('scoring', 'score').replace('neg_', '')} when shuffled")
    ax.set_title("Permutation importance (big train, small test = likely overfit)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return ax


def weight_stability(estimator, X, y, n_splits=5):
    """
    Refit the model on each expanding time fold (TimeSeriesSplit) and compare the weights.
    Returns (weights by fold, summary). Summary columns:
      mean, sd, share of folds used, sign flips (weight changes sign across folds).
    Unstable or sign-flipping weights = the model is fitting noise for that feature.
    """
    rows = {}
    for i, (tr, _) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        end = X.index[tr[-1]]
        rows[f"fold {i + 1} (to {end.date() if hasattr(end, 'date') else end})"] = \
            feature_weights(m, X.iloc[tr]).set_index("feature")["weight"]
    m = clone(estimator).fit(X, y)
    rows["full training"] = feature_weights(m, X).set_index("feature")["weight"]
    W = pd.DataFrame(rows).T
    folds = W.iloc[:-1]
    signs = np.sign(folds.where(folds.abs() > 1e-12))
    summary = pd.DataFrame({
        "full-training weight": W.iloc[-1],
        "mean across folds": folds.mean(),
        "sd across folds": folds.std(),
        "share of folds used": (folds.abs() > 1e-12).mean(),
        "sign flips": (signs.nunique() > 1),
    }).sort_values("full-training weight", key=abs, ascending=False)
    return W, summary


def plot_weight_stability(W, top=15, ax=None):
    """Weights per fold for the largest features: tight dots = stable, spread/crossing 0 = unstable."""
    order = W.iloc[-1].abs().sort_values(ascending=False).index[:top]
    ax = ax or plt.subplots(figsize=(9, 0.35 * len(order) + 1.2))[1]
    folds = W.iloc[:-1]
    for i, f in enumerate(order[::-1]):
        ax.scatter(folds[f], [i] * len(folds), color="tab:blue", alpha=0.6, s=18)
        ax.scatter(W.iloc[-1][f], i, color="black", marker="|", s=200)
    ax.set_yticks(range(len(order)), order[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Weight stability across time folds (dots = folds, bar = full training)")
    ax.grid(axis="x", alpha=0.3)
    return ax