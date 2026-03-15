#!/usr/bin/env python3
"""
Standalone k-NN grid search over k and distance metrics using 10-fold CV.
Uses the same splits for every (k, metric) test. Parallelized; reports best k and metric.
"""
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import os
import time

from .distance_metrics import METRIC_NAMES, get_metric_config


def load_and_prepare(data_path: str, test_path: str = None):
    """Load CSV(s), extract X/y, normalize, return arrays and metadata."""
    if test_path:
        train_df = pd.read_csv(data_path)
        test_df = pd.read_csv(test_path)
        class_col = [c for c in train_df.columns if c.lower() == "class"][0]
        X = train_df.drop(columns=[class_col]).values
        y_raw = train_df[class_col].astype(str).values
        X_test = test_df.drop(columns=[class_col]).values
        y_test_raw = test_df[class_col].astype(str).values
        le = LabelEncoder()
        le.fit(np.unique(np.concatenate([y_raw, y_test_raw])))
        y = le.transform(y_raw)
        y_test = le.transform(y_test_raw)
    else:
        df = pd.read_csv(data_path)
        class_col = [c for c in df.columns if c.lower() == "class"][0]
        X = df.drop(columns=[class_col]).values
        y_raw = df[class_col].astype(str).values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        X_test = y_test = None

    col_mean = np.nanmean(X, axis=0)
    if np.isnan(X).any():
        X = np.where(np.isnan(X), col_mean, X)
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    X = (X - mn) / rng
    if X_test is not None:
        if np.isnan(X_test).any():
            X_test = np.where(np.isnan(X_test), col_mean, X_test)
        X_test = (X_test - mn) / rng

    n_features = X.shape[1]
    n_samples = X.shape[0]
    return X, y, X_test, y_test, n_features, n_samples, le


def _build_class_distance_keys(labels, distances):
    """Precompute sorted distance lists and padded lexicographic keys per class for one sample."""
    class_to_dist = {}
    for d, c in zip(distances, labels):
        class_to_dist.setdefault(c, []).append(d)

    k = len(distances)
    class_to_sorted = {}
    class_to_key = {}
    for c, dists in class_to_dist.items():
        arr = np.sort(np.asarray(dists, dtype=float))
        class_to_sorted[c] = arr
        key = np.full(k, np.inf, dtype=float)
        key[: len(arr)] = arr
        class_to_key[c] = key
    return class_to_sorted, class_to_key


def _lex_is_better(key_a, key_b):
    """Return -1 if key_a is lexicographically better (smaller) than key_b,
    1 if worse, 0 if identical."""
    if np.array_equal(key_a, key_b):
        return 0
    for a, b in zip(key_a, key_b):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def _predict_with_tiebreak(labels, distances):
    """Custom majority vote with distance-based deterministic tie-breaking for one sample."""
    labels = np.asarray(labels)
    distances = np.asarray(distances, dtype=float)

    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    tied_classes = unique[counts == max_count]

    if len(tied_classes) == 1:
        return tied_classes[0]

    class_to_sorted, class_to_key = _build_class_distance_keys(labels, distances)
    tied = list(tied_classes)

    scores = {c: 0 for c in tied}
    for i in range(len(tied)):
        for j in range(i + 1, len(tied)):
            ci, cj = tied[i], tied[j]
            ki, kj = class_to_key[ci], class_to_key[cj]
            cmp = _lex_is_better(ki, kj)
            if cmp < 0:
                scores[ci] += 1
                scores[cj] -= 1
            elif cmp > 0:
                scores[ci] -= 1
                scores[cj] += 1

    max_score = max(scores.values())
    best_classes = [c for c, s in scores.items() if s == max_score]

    if len(best_classes) == 1:
        return best_classes[0]

    best_c = None
    best_key = None
    for c in best_classes:
        key = class_to_key[c]
        if best_c is None:
            best_c, best_key = c, key
        else:
            cmp = _lex_is_better(key, best_key)
            if cmp < 0:
                best_c, best_key = c, key
            elif cmp == 0 and c < best_c:
                best_c = c
                best_key = key
    return best_c


def run_one_metric_phase(args):
    """Worker: (metric_name, k_list, max_k, X_train, y_train, X_test, y_test)."""
    metric_name, k_list, max_k, X_train, y_train, X_test, y_test = args
    start = time.time()

    metric, metric_params, algorithm = get_metric_config(metric_name)
    if metric_name.lower() == "mahalanobis":
        try:
            cov = np.cov(X_train.T)
            cov += 1e-6 * np.eye(cov.shape[0])
            VI = np.linalg.inv(cov)
            metric_params = {"VI": VI}
        except np.linalg.LinAlgError:
            return ([(k, metric_name, 0.0) for k in k_list], np.inf)

    try:
        clf = KNeighborsClassifier(
            n_neighbors=max_k,
            metric=metric,
            metric_params=metric_params if metric_params else None,
            algorithm=algorithm,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        dists, inds = clf.kneighbors(X_test, n_neighbors=max_k, return_distance=True)
    except Exception:
        return ([(k, metric_name, np.nan) for k in k_list], time.time() - start)

    results = []
    for k in k_list:
        preds = []
        for i in range(len(X_test)):
            dist_row = dists[i, :k]
            ind_row = inds[i, :k]
            neighbor_labels = y_train[ind_row]
            preds.append(_predict_with_tiebreak(neighbor_labels, dist_row))
        preds = np.asarray(preds)
        acc = accuracy_score(y_test, preds)
        results.append((k, metric_name, acc))

    elapsed = time.time() - start
    return (results, elapsed)


def main():
    parser = argparse.ArgumentParser(description="k-NN k and distance metric search with 10-fold CV")
    parser.add_argument("--data", required=True, help="Training data CSV (or single dataset)")
    parser.add_argument("--test-data", default=None, help="Optional test CSV (CV is on train only)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs (default: CPU count - 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for folds")
    parser.add_argument(
        "--k-values",
        help="Comma-separated list of k values to test (overrides automatic sqrt/full ranges)",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of distance metrics to test "
             f"(subset of: {', '.join(METRIC_NAMES)})",
    )
    parser.add_argument(
        "--log-each-test",
        action="store_true",
        help="Print per-(k, metric) accuracy and timing while still showing tqdm",
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=60.0,
        help="Seconds above which a single (k, metric) run is considered slow and triggers a warning",
    )
    args = parser.parse_args()

    print("Loading data...")
    X, y, X_test, y_test, n_features, n_samples, le = load_and_prepare(args.data, args.test_data)
    if X_test is None or y_test is None:
        raise SystemExit("This script now requires --test-data so we can train on the training set and evaluate on a held-out test set.")
    print(f"Samples: {n_samples}, features: {n_features}, test samples: {X_test.shape[0]}")

    k_sqrt = max(1, int(sqrt(n_features)))
    train_size = n_samples
    k_max_full = min(n_features, train_size)

    if args.metrics:
        requested = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        selected_metrics = []
        for m in METRIC_NAMES:
            if m.lower() in requested:
                selected_metrics.append(m)
        if not selected_metrics:
            raise ValueError(f"No valid metrics selected from {requested}; available: {METRIC_NAMES}")
        metric_names = selected_metrics
    else:
        metric_names = METRIC_NAMES

    if args.k_values:
        k_values = sorted({
            int(x) for x in args.k_values.split(",") if x.strip()
        })
        k_values = [k for k in k_values if 1 <= k <= train_size]
        if not k_values:
            raise ValueError("No valid k values after filtering; ensure 1 <= k <= number of training samples")
        max_k = max(k_values)
        phases = [("All k", k_values, max_k)]
    else:
        k_phase1 = list(range(1, k_sqrt + 1))
        k_phase2 = list(range(k_sqrt + 1, k_max_full + 1))
        phases = [("Phase 1", k_phase1, max(k_phase1))]
        if k_phase2:
            phases.append(("Phase 2", k_phase2, max(k_phase2)))

    def run_tasks(phase_name, k_list, max_k):
        task_args = [
            (metric_name, k_list, max_k, X, y, X_test, y_test)
            for metric_name in metric_names
        ]
        n_jobs = args.n_jobs or max(1, multiprocessing.cpu_count() - 1)
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_one_metric_phase, a): a for a in task_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc=phase_name):
                try:
                    result_list, elapsed = future.result()
                except Exception:
                    a = futures[future]
                    result_list = [(k, a[0], np.nan) for k in a[1]]
                    elapsed = np.nan
                if result_list and elapsed is not None and not np.isnan(elapsed) and elapsed > args.slow_threshold:
                    tqdm.write(
                        f"WARNING: [{phase_name}] metric={result_list[0][1]} (max_k={max_k}) took "
                        f"{elapsed:.2f}s (> {args.slow_threshold:.1f}s); "
                        f"search may be slowing toward infeasible runtimes."
                    )
                for k, metric_name, acc in result_list:
                    results.append((k, metric_name, acc, elapsed))
                    if args.log_each_test:
                        tqdm.write(
                            f"[{phase_name}] k={k}, metric={metric_name}, "
                            f"accuracy={acc:.4f}, time={elapsed:.2f}s"
                        )
        return results

    all_results = []
    for phase_info in phases:
        phase_name = phase_info[0]
        k_list = phase_info[1]
        max_k = phase_info[2]
        if phase_name == "Phase 1":
            print(f"\nPhase 1: k from 1 to sqrt(dim) = {k_sqrt} (one NN search per metric, max_k={max_k})")
        elif phase_name == "Phase 2":
            print(f"\nPhase 2: k from {k_sqrt + 1} to {k_max_full} (one NN search per metric, max_k={max_k})")
        else:
            print(f"\n{phase_name} (one NN search per metric, max_k={max_k})")
        all_results.extend(run_tasks(phase_name, k_list, max_k))

    all_results.sort(key=lambda r: (r[0], r[1]))

    print("\n--- Results (k, metric, test-set accuracy) ---")
    for k, metric, acc, elapsed in all_results:
        print(f"k={k:3d}  metric={metric:22s}  accuracy={acc:.4f}  time={elapsed:.2f}s")

    valid = [(k, m, a, t) for k, m, a, t in all_results if not np.isnan(a)]
    if not valid:
        print("No valid results.")
        return
    best = max(valid, key=lambda x: x[2])
    print("\n--- Best (train on train, evaluate on test) ---")
    print(f"Best k: {best[0]}, Best metric: {best[1]}, test accuracy: {best[2]:.4f}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "knn_metric_search_results.csv")
    with open(out_path, "w") as f:
        f.write("k,metric,accuracy_test,time_seconds\n")
        for k, metric, acc, elapsed in all_results:
            f.write(f"{k},{metric},{acc:.6f},{elapsed:.4f}\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
