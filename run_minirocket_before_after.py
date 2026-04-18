import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tsai.basics import *
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *

from ngafiddataset.dataset.dataset import NGAFID_Dataset_Manager


RESULTS_BASENAME = "minirocket_before_after_cv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MiniRocket 5-fold cross-validation for NGAFID before-vs-after maintenance classification."
    )
    parser.add_argument("--dataset-dir", type=str, default=".", help="Directory where the 2days dataset is stored/downloaded.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for metrics outputs.")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Specific fold indices to run. Default: all 5 folds.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per fold.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-5, help="Learning rate for fit_one_cycle.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for tsai dataloaders.")
    parser.add_argument("--chunksize", type=int, default=64, help="MiniRocket feature extraction chunk size.")
    parser.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length used by the dataset manager.")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=("global", "fold"),
        default="global",
        help="Use paper-style global min/max stats or recompute min/max from each training fold.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def normalize_features(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    scaled = (data - mins) / ranges
    return np.nan_to_num(scaled, copy=False)


def compute_fold_minmax(dm: NGAFID_Dataset_Manager, sample_ids) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(dm.channels, np.inf, dtype=np.float32)
    maxs = np.full(dm.channels, -np.inf, dtype=np.float32)

    for sample_id in sample_ids:
        flight = np.asarray(dm.flight_data_array[sample_id], dtype=np.float32)
        if flight.size == 0:
            continue

        finite = np.isfinite(flight)
        if not finite.any():
            continue

        mins = np.minimum(mins, np.where(finite, flight, np.inf).min(axis=0))
        maxs = np.maximum(maxs, np.where(finite, flight, -np.inf).max(axis=0))

    mins = np.where(np.isfinite(mins), mins, dm.mins)
    maxs = np.where(np.isfinite(maxs), maxs, dm.maxs)
    return mins, maxs


def build_dataset_manager(dataset_dir: Path, max_length: int) -> NGAFID_Dataset_Manager:
    dm = NGAFID_Dataset_Manager("2days", destination=str(dataset_dir), max_length=max_length)
    dm.data_dict = dm.construct_data_dictionary(numpy=True)
    return dm


def select_positive_class_scores(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    return probabilities[:, 1]


def build_prediction_records(
    dm: NGAFID_Dataset_Manager,
    fold: int,
    test_dict: dict,
    pred_labels: np.ndarray,
    positive_scores: np.ndarray,
    args,
) -> pd.DataFrame:
    sample_ids = np.asarray(test_dict["id"])
    true_labels = np.asarray(test_dict["before_after"], dtype=np.int64)
    seq_lens = [int(min(dm.flight_data_array[sample_id].shape[0], args.max_length)) for sample_id in sample_ids]

    return pd.DataFrame(
        {
            "fold": fold,
            "id": sample_ids.astype(int),
            "target_class": np.asarray(test_dict["target_class"], dtype=np.int64),
            "class": np.asarray(test_dict["class"]),
            "hclass": np.asarray(test_dict["hclass"]),
            "true_label": true_labels,
            "pred_label": pred_labels.astype(int),
            "positive_score": positive_scores.astype(float),
            "correct": (pred_labels == true_labels).astype(int),
            "seq_len": seq_lens,
        }
    )


def train_single_fold(dm: NGAFID_Dataset_Manager, fold: int, args) -> tuple[dict, pd.DataFrame]:
    train_dict = dm.get_numpy_dataset(fold=fold, training=True)
    test_dict = dm.get_numpy_dataset(fold=fold, training=False)

    train_x = np.asarray(train_dict["data"], dtype=np.float32)
    test_x = np.asarray(test_dict["data"], dtype=np.float32)

    if args.normalization == "fold":
        mins, maxs = compute_fold_minmax(dm, train_dict["id"])
    else:
        mins, maxs = dm.mins, dm.maxs

    train_x = normalize_features(train_x, mins, maxs)
    test_x = normalize_features(test_x, mins, maxs)

    train_y = np.asarray(train_dict["before_after"], dtype=np.int64)
    test_y = np.asarray(test_dict["before_after"], dtype=np.int64)

    splits = [list(np.arange(len(train_y))), list(np.arange(len(test_y)) + len(train_y))]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mrf = MiniRocketFeatures(train_x.shape[1], train_x.shape[2]).to(default_device())
    mrf.fit(train_x, chunksize=args.chunksize)

    all_x = np.concatenate([train_x, test_x], axis=0)
    all_y = np.concatenate([train_y, test_y], axis=0)
    x_features = get_minirocket_features(all_x, mrf, chunksize=args.chunksize, to_np=True)

    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(x_features, all_y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=args.batch_size)
    model = build_ts_model(MiniRocketHead, dls=dls)

    learner = Learner(dls, model, metrics=accuracy)
    learner.fit_one_cycle(args.epochs, args.learning_rate)

    probabilities, targets = learner.get_preds(dl=dls.valid)
    probabilities = probabilities.cpu().numpy()
    targets = targets.cpu().numpy().astype(int)

    pred_labels = probabilities.argmax(axis=1) if probabilities.ndim > 1 else (probabilities >= 0.5).astype(int)
    positive_scores = select_positive_class_scores(probabilities)

    fold_metrics = {
        "fold": fold,
        "train_size": int(len(train_y)),
        "test_size": int(len(test_y)),
        "normalization": args.normalization,
        "accuracy": float(accuracy_score(targets, pred_labels)),
        "f1": float(f1_score(targets, pred_labels, zero_division=0)),
    }

    try:
        fold_metrics["roc_auc"] = float(roc_auc_score(targets, positive_scores))
    except ValueError:
        fold_metrics["roc_auc"] = None

    prediction_records = build_prediction_records(dm, fold, test_dict, pred_labels, positive_scores, args)
    return fold_metrics, prediction_records


def summarize_results(results_df: pd.DataFrame) -> dict:
    summary = {
        "num_folds": int(len(results_df)),
        "accuracy_mean": float(results_df["accuracy"].mean()),
        "accuracy_std": float(results_df["accuracy"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "f1_mean": float(results_df["f1"].mean()),
        "f1_std": float(results_df["f1"].std(ddof=1)) if len(results_df) > 1 else 0.0,
    }

    roc_auc_series = pd.to_numeric(results_df["roc_auc"], errors="coerce")
    if roc_auc_series.notna().any():
        summary["roc_auc_mean"] = float(roc_auc_series.mean())
        summary["roc_auc_std"] = float(roc_auc_series.std(ddof=1)) if roc_auc_series.notna().sum() > 1 else 0.0
    else:
        summary["roc_auc_mean"] = None
        summary["roc_auc_std"] = None

    return summary


def save_outputs(results_df: pd.DataFrame, summary: dict, results_dir: Path, predictions_df: pd.DataFrame | None = None):
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{RESULTS_BASENAME}.csv"
    json_path = results_dir / f"{RESULTS_BASENAME}_summary.json"
    predictions_path = results_dir / f"{RESULTS_BASENAME}_predictions.csv"

    results_df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if predictions_df is not None:
        predictions_df.to_csv(predictions_path, index=False)

    print(f"\nSaved fold metrics to: {csv_path}")
    print(f"Saved summary metrics to: {json_path}")
    if predictions_df is not None:
        print(f"Saved per-sample predictions to: {predictions_path}")


def print_summary(results_df: pd.DataFrame, summary: dict):
    printable = results_df.copy()
    print("\nPer-fold results:")
    print(printable.to_string(index=False))

    print("\nCross-validation summary:")
    print(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"F1:       {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    if summary["roc_auc_mean"] is None:
        print("ROC-AUC:  unavailable")
    else:
        print(f"ROC-AUC:  {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    folds = args.folds if args.folds else list(range(5))
    invalid_folds = [fold for fold in folds if fold not in range(5)]
    if invalid_folds:
        raise ValueError(f"Fold indices must be in [0, 1, 2, 3, 4]. Got: {invalid_folds}")

    dataset_dir = Path(args.dataset_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    print(f"Using dataset directory: {dataset_dir}")
    print(f"Using results directory: {results_dir}")
    print(f"Running folds: {folds}")
    print(f"Normalization: {args.normalization}")
    print(f"Device: {default_device()}")

    dm = build_dataset_manager(dataset_dir=dataset_dir, max_length=args.max_length)

    fold_results = []
    prediction_records = []
    for fold in folds:
        print(f"\n===== Fold {fold} / 4 =====")
        fold_result, fold_predictions = train_single_fold(dm=dm, fold=fold, args=args)
        fold_results.append(fold_result)
        prediction_records.append(fold_predictions)
        print(
            f"Fold {fold} metrics -> accuracy: {fold_result['accuracy']:.4f}, "
            f"f1: {fold_result['f1']:.4f}, roc_auc: {fold_result['roc_auc']}"
        )

    results_df = pd.DataFrame(fold_results).sort_values("fold").reset_index(drop=True)
    predictions_df = pd.concat(prediction_records, ignore_index=True)
    summary = summarize_results(results_df)
    summary["normalization"] = args.normalization
    summary["max_length"] = args.max_length
    save_outputs(results_df, summary, results_dir, predictions_df=predictions_df)
    print_summary(results_df, summary)


if __name__ == "__main__":
    main()
