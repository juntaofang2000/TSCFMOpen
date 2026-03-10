import random
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import argparse
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
import sys

# 1. Import from custom modules
from data_reader import DataReader
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from mantis_dev.architecture import Mantis8MRelu  # noqa: F401

from sklearn.metrics.pairwise import cosine_similarity

# --- local mantis_dev path setup ---
CURRENT_DIR = Path(__file__).resolve().parent
MANTIS_DEV_DIR = CURRENT_DIR / "mantis_dev"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if MANTIS_DEV_DIR.exists() and str(MANTIS_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(MANTIS_DEV_DIR))


def per_class_centroid_cosines(Ztr, ytr, Zte, yte):
    """Calculate per-class centroid cosine similarity"""
    classes = np.unique(ytr)
    out = {}
    for c in classes:
        ztrc = Ztr[ytr == c]
        ztec = Zte[yte == c]
        if len(ztrc) == 0 or len(ztec) == 0:
            out[c] = np.nan
            continue
        cent = ztrc.mean(0, keepdims=True)
        out[c] = cosine_similarity(ztec, cent).mean()
    return out


def class_margin_ratio(Ztr, ytr, metric='euclidean'):
    """Calculate class margin ratio (min inter-centroid dist / mean within-class scatter)"""
    classes = np.unique(ytr)
    cents = []
    within = []
    for c in classes:
        z = Ztr[ytr == c]
        if len(z) <= 1:
            continue
        cents.append(z.mean(0))
        within.append(pairwise_distances(z, z.mean(0, keepdims=True), metric=metric).mean())

    if len(cents) < 2:
        return np.nan

    cents = np.vstack(cents)
    min_inter = pairwise_distances(cents).astype(float)
    np.fill_diagonal(min_inter, np.inf)
    min_inter = min_inter.min()
    return min_inter / (np.mean(within) + 1e-9)


def train_test_discriminator(Ztr, Zte, seed=42):
    """Train a discriminator to distinguish between train and test samples"""
    X = np.vstack([Ztr, Zte])
    y = np.hstack([np.zeros(len(Ztr)), np.ones(len(Zte))])
    clf = LogisticRegression(max_iter=1000, solver='saga', random_state=seed)
    return cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _encode_labels(y_train: np.ndarray, y_test: np.ndarray):
    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ytr = np.asarray([class_to_idx[c] for c in y_train], dtype=np.int64)
    yte = np.asarray([class_to_idx.get(c, -1) for c in y_test], dtype=np.int64)
    if (yte < 0).any():
        raise ValueError("Test labels contain unseen class not present in train set")
    return ytr, yte, len(classes)


def train_eval_mlp(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: str,
    hidden_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    """Single-layer linear head (1-layer MLP): Linear, trained with CE on frozen features."""
    set_global_seed(seed)

    ytr, yte, num_classes = _encode_labels(y_train, y_test)

    Xtr = torch.from_numpy(np.asarray(Z_train, dtype=np.float32))
    Xte = torch.from_numpy(np.asarray(Z_test, dtype=np.float32))
    ytr_t = torch.from_numpy(ytr)
    yte_t = torch.from_numpy(yte)

    in_dim = int(Xtr.shape[1])
    model = nn.Linear(in_dim, num_classes).to(device)

    ds = TensorDataset(Xtr, ytr_t)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return float((preds == yte).mean())


def stratified_min1_split_from_test(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.1,
    seed: int = 42,
):
    """Split (X, y) into train/test by sampling from the *given* set.

    Requirements:
    - Train comes from this set (UCR test set in our use-case)
    - Stratified sampling
    - Ensure train has at least 1 sample per class (when possible)

    Notes:
    - If a class has only 1 sample total, test will inevitably miss that class.
    - We keep the split deterministic given `seed`.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n = int(len(y))
    if n == 0:
        raise ValueError("Empty dataset")

    classes, counts = np.unique(y, return_counts=True)
    num_classes = int(len(classes))

    # Desired train size: 10% of the provided set, but at least 1 per class.
    desired_train = int(np.ceil(float(train_frac) * n))
    desired_train = max(desired_train, num_classes)

    # Ensure we leave at least 1 sample for test.
    if desired_train >= n:
        desired_train = n - 1
    if desired_train <= 0:
        raise ValueError("Not enough samples to create a non-empty test set")

    rng = np.random.default_rng(seed)

    # Step 1: mandatory 1 sample per class for train
    train_indices = []
    remaining_indices = []
    remaining_labels = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        pick = int(rng.choice(idx_c, size=1, replace=False)[0])
        train_indices.append(pick)

        # keep the rest for proportional allocation
        rest = idx_c[idx_c != pick]
        remaining_indices.append(rest)
        remaining_labels.append(np.full(len(rest), c, dtype=y.dtype))

    train_indices = np.asarray(train_indices, dtype=np.int64)

    # Step 2: allocate the remaining train slots proportionally (stratified)
    additional_needed = int(desired_train - len(train_indices))
    if additional_needed > 0:
        rem_idx = np.concatenate(remaining_indices) if remaining_indices else np.array([], dtype=np.int64)
        rem_y = np.concatenate(remaining_labels) if remaining_labels else np.array([], dtype=y.dtype)

        if len(rem_idx) > 0:
            rem_classes, rem_counts = np.unique(rem_y, return_counts=True)
            total_rem = int(rem_counts.sum())
            if total_rem > 0:
                expected = (additional_needed * (rem_counts / total_rem)).astype(float)
                take = np.floor(expected).astype(int)

                # distribute leftovers by largest fractional parts, respecting availability
                leftover = int(additional_needed - take.sum())
                frac = expected - np.floor(expected)
                order = np.argsort(-frac)
                for j in order:
                    if leftover <= 0:
                        break
                    if take[j] < rem_counts[j]:
                        take[j] += 1
                        leftover -= 1

                # If still leftover (due to caps), fill from any class with remaining
                if leftover > 0:
                    avail = rem_counts - take
                    while leftover > 0 and avail.sum() > 0:
                        j = int(np.argmax(avail))
                        take[j] += 1
                        avail[j] -= 1
                        leftover -= 1

                # sample per class
                extra_train = []
                for c, k in zip(rem_classes, take):
                    if k <= 0:
                        continue
                    idx_c = rem_idx[rem_y == c]
                    k = min(int(k), int(len(idx_c)))
                    if k <= 0:
                        continue
                    extra_train.append(rng.choice(idx_c, size=k, replace=False))

                if extra_train:
                    extra_train = np.concatenate(extra_train).astype(np.int64)
                    train_indices = np.concatenate([train_indices, extra_train])

    train_indices = np.unique(train_indices)

    # Final test indices are all remaining
    mask = np.ones(n, dtype=bool)
    mask[train_indices] = False
    test_indices = np.where(mask)[0].astype(np.int64)

    # Shuffle for cleanliness (deterministic)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pretrained Mantis on UCR datasets, where BOTH train/test are split from UCR TEST set: "
            "train=10% stratified (>=1 per class), test=remaining 90%."
        )
    )
    parser.add_argument(
        "--dataPath",
        type=str,
        default='/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/',
        help="Path to the root folder containing UCR datasets",
    )
    parser.add_argument(
        "--ueaPath",
        type=str,
        default='/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/',
        help="Optional path to UEA datasets (required by DataReader interface)",
    )
    parser.add_argument(
        "--modelPath",
        type=str,
        default="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint",
        help="Path to the pretrained checkpoint",
    )
    parser.add_argument(
        "--outputPath",
        type=str,
        default='results_testMantis_fromTest_split10_20260117.log',
        help="Path to save the log results",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logreg", "rf", "svm", "mlp", "knn", "nc", "both", "all"],
        default="all",
        help="Choose which downstream classifier(s) to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base random seed. If --seeds is not provided, 5 reproducible random seeds will be generated from this base seed. "
            "Also used for the UCR-test split.")
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run evaluation multiple times with these seeds (e.g. --seeds 42 43 44).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Only run these dataset names (e.g. --datasets ECG200). Default: run all available UCR datasets.",
    )
    parser.add_argument(
        "--splitSeed",
        type=int,
        default=None,
        help="Seed used for the (UCR test -> 10%/90%) split. Default: use --seed.",
    )
    parser.add_argument(
        "--trainFrac",
        type=float,
        default=0.1,
        help="Train fraction sampled from UCR test set (default: 0.1).",
    )

    args = parser.parse_args()

    DATA_PATH = args.dataPath
    UEA_PATH = args.ueaPath
    MODEL_PATH = args.modelPath
    classifier = args.classifier

    split_seed = args.seed if args.splitSeed is None else int(args.splitSeed)

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        rng = np.random.default_rng(args.seed)
        seeds = rng.integers(low=0, high=1_000_000_000, size=5, dtype=np.int64).tolist()

    logerName = args.outputPath
    print(f"Logger name: {logerName}")

    #############################
    # Logging Configuration
    #############################
    logging.basicConfig(level=logging.INFO, filename=logerName, format='%(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Seeds: {seeds}")
    logger.info(f"Split seed: {split_seed}")
    logger.info(f"Train fraction from UCR test: {args.trainFrac}")

    # ---------------------------
    # 2. Fix random seed for reproducibility
    # ---------------------------
    set_global_seed(seeds[0])

    # ---------------------------
    # 3. Read UCR data (but we will split FROM the UCR test portion)
    # ---------------------------
    reader = DataReader(UCR_data_path=DATA_PATH, UEA_data_path=UEA_PATH)

    all_data = {}
    dataset_list = args.datasets if args.datasets is not None else reader.dataset_list_ucr

    for ds_name in dataset_list:
        try:
            if ds_name not in reader.dataset_list_ucr:
                raise ValueError(f"Dataset '{ds_name}' not found under UCRArchive_2018")

            # Only read the official test set, then split it into (train/test) = (10%/90%)
            X_full, y_full = reader.read_dataset(ds_name, which_set='test')

            X_train, y_train, X_test, y_test = stratified_min1_split_from_test(
                X_full,
                y_full,
                train_frac=float(args.trainFrac),
                seed=int(split_seed),
            )

            logger.info(f"Successfully read dataset: {ds_name}")
            logger.info(
                f"From UCR test: full={len(X_full)} -> train={len(X_train)} (frac={args.trainFrac}), test={len(X_test)}")

            # Quick check: train has all classes that exist in full
            full_classes = set(np.unique(y_full).tolist())
            train_classes = set(np.unique(y_train).tolist())
            missing_in_train = sorted(list(full_classes - train_classes))
            if missing_in_train:
                logger.warning(f"[Split warning] {ds_name}: missing classes in train: {missing_in_train}")

            all_data[ds_name] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }

        except Exception as e:
            logger.warning(f"Failed to read/split dataset {ds_name}: {e}")
            continue

    if len(all_data) == 0:
        raise RuntimeError("No dataset was loaded from UCRArchive_2018. Please check your path or data format.")

    # ---------------------------
    # 4. Load pretrained Mantis model
    # ---------------------------
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    mantis_model = Mantis8M(
        seq_len=512,
        hidden_dim=256,
        num_patches=32,
        scalar_scales=None,
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1,
        transf_depth=6,
        transf_num_heads=8,
        transf_mlp_dim=512,
        transf_dim_head=128,
        transf_dropout=0.1,
        device=device,
        pre_training=False,
    )

    mantis_model = mantis_model.from_pretrained(MODEL_PATH)
    trainer = MantisTrainer(device=device, network=mantis_model)
    logger.info(f"Pretrained model loaded successfully from {MODEL_PATH}.")

    # ---------------------------
    # 5. Per-dataset feature extraction and classification
    # ---------------------------
    logger.info("Start feature extraction and classification for each UCR dataset...")

    run_logreg = classifier in ("logreg", "both", "all")
    run_rf = classifier in ("rf", "both", "all")
    run_svm = classifier in ("svm", "all")
    run_mlp = classifier in ("mlp", "all")
    run_knn = classifier in ("knn", "all")
    run_nc = classifier in ("nc", "all")

    if not (run_logreg or run_rf or run_svm or run_mlp or run_knn or run_nc):
        raise ValueError("At least one classifier (logreg, rf, svm, mlp, knn, nc) must be selected")

    results = []

    for ds_name, data_dict in all_data.items():
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]

        # Extract features once
        Z_train = trainer.transform(X_train, batch_size=512, to_numpy=True)
        Z_test = trainer.transform(X_test, batch_size=512, to_numpy=True)

        entry = {
            "dataset": ds_name,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "lr_test_accs": [],
            "rf_test_accs": [],
            "svm_test_accs": [],
            "mlp_test_accs": [],
            "knn_test_accs": [],
            "nc_test_accs": [],
            "lr_test_mean": None,
            "lr_test_range": None,
            "rf_test_mean": None,
            "rf_test_range": None,
            "svm_test_mean": None,
            "svm_test_range": None,
            "mlp_test_mean": None,
            "mlp_test_range": None,
            "knn_test_mean": None,
            "knn_test_range": None,
            "nc_test_mean": None,
            "nc_test_range": None,
        }

        # Run multiple seeds for downstream heads, on the same extracted features.
        for seed in seeds:
            set_global_seed(seed)

            if run_logreg:
                lr = LogisticRegression(penalty='l2', random_state=seed, max_iter=100)
                lr.fit(Z_train, y_train)
                y_pred_test = lr.predict(Z_test)
                test_acc = float((y_pred_test == y_test).mean())
                entry["lr_test_accs"].append(test_acc)

            if run_rf:
                rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                rf.fit(Z_train, y_train)
                y_pred_test_rf = rf.predict(Z_test)
                rf_test_acc = float((y_pred_test_rf == y_test).mean())
                entry["rf_test_accs"].append(rf_test_acc)

            if run_svm:
                svm = LinearSVC(random_state=seed, max_iter=100)
                svm.fit(Z_train, y_train)
                y_pred_test_svm = svm.predict(Z_test)
                svm_test_acc = float((y_pred_test_svm == y_test).mean())
                entry["svm_test_accs"].append(svm_test_acc)

            if run_mlp:
                mlp_test_acc = train_eval_mlp(
                    Z_train=Z_train,
                    y_train=y_train,
                    Z_test=Z_test,
                    y_test=y_test,
                    seed=seed,
                    device=device,
                )
                entry["mlp_test_accs"].append(mlp_test_acc)

            if run_knn:
                knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
                knn.fit(Z_train, y_train)
                y_pred_test_knn = knn.predict(Z_test)
                knn_test_acc = float((y_pred_test_knn == y_test).mean())
                entry["knn_test_accs"].append(knn_test_acc)

            if run_nc:
                nc = NearestCentroid()
                nc.fit(Z_train, y_train)
                y_pred_test_nc = nc.predict(Z_test)
                nc_test_acc = float((y_pred_test_nc == y_test).mean())
                entry["nc_test_accs"].append(nc_test_acc)

        # Aggregate per-dataset metrics (mean + range across seeds)
        if run_logreg and entry["lr_test_accs"]:
            arr = np.asarray(entry["lr_test_accs"], dtype=float)
            entry["lr_test_mean"] = float(arr.mean())
            entry["lr_test_range"] = float(arr.max() - arr.min())

        if run_rf and entry["rf_test_accs"]:
            arr = np.asarray(entry["rf_test_accs"], dtype=float)
            entry["rf_test_mean"] = float(arr.mean())
            entry["rf_test_range"] = float(arr.max() - arr.min())

        if run_svm and entry["svm_test_accs"]:
            arr = np.asarray(entry["svm_test_accs"], dtype=float)
            entry["svm_test_mean"] = float(arr.mean())
            entry["svm_test_range"] = float(arr.max() - arr.min())

        if run_mlp and entry["mlp_test_accs"]:
            arr = np.asarray(entry["mlp_test_accs"], dtype=float)
            entry["mlp_test_mean"] = float(arr.mean())
            entry["mlp_test_range"] = float(arr.max() - arr.min())

        if run_knn and entry["knn_test_accs"]:
            arr = np.asarray(entry["knn_test_accs"], dtype=float)
            entry["knn_test_mean"] = float(arr.mean())
            entry["knn_test_range"] = float(arr.max() - arr.min())

        if run_nc and entry["nc_test_accs"]:
            arr = np.asarray(entry["nc_test_accs"], dtype=float)
            entry["nc_test_mean"] = float(arr.mean())
            entry["nc_test_range"] = float(arr.max() - arr.min())

        # Console preview (mean/range)
        if run_logreg and entry["lr_test_mean"] is not None:
            print(
                f"[{ds_name}] Logistic Regression - Test Mean: {entry['lr_test_mean']:.4f}, Range: {entry['lr_test_range']:.4f}, Accs: {entry['lr_test_accs']}"
            )
        if run_rf and entry["rf_test_mean"] is not None:
            print(
                f"[{ds_name}] Random Forest - Test Mean: {entry['rf_test_mean']:.4f}, Range: {entry['rf_test_range']:.4f}, Accs: {entry['rf_test_accs']}"
            )
        if run_svm and entry["svm_test_mean"] is not None:
            print(
                f"[{ds_name}] SVM (Linear) - Test Mean: {entry['svm_test_mean']:.4f}, Range: {entry['svm_test_range']:.4f}, Accs: {entry['svm_test_accs']}"
            )
        if run_mlp and entry["mlp_test_mean"] is not None:
            print(
                f"[{ds_name}] MLP (1-layer) - Test Mean: {entry['mlp_test_mean']:.4f}, Range: {entry['mlp_test_range']:.4f}, Accs: {entry['mlp_test_accs']}"
            )
        if run_knn and entry["knn_test_mean"] is not None:
            print(
                f"[{ds_name}] KNN (k=1) - Test Mean: {entry['knn_test_mean']:.4f}, Range: {entry['knn_test_range']:.4f}, Accs: {entry['knn_test_accs']}"
            )
        if run_nc and entry["nc_test_mean"] is not None:
            print(
                f"[{ds_name}] Nearest Centroid - Test Mean: {entry['nc_test_mean']:.4f}, Range: {entry['nc_test_range']:.4f}, Accs: {entry['nc_test_accs']}"
            )

        results.append(entry)

    # ---------------------------
    # 6. Summary
    # ---------------------------
    def _mean_and_range(values):
        if not values:
            return None, None
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.max() - arr.min())

    def _overall_seed_dataset_stats(accs_key: str):
        per_seed_means = []
        per_seed_medians = []

        for i, _seed in enumerate(seeds):
            vals = []
            for r in results:
                accs = r.get(accs_key, [])
                if len(accs) > i:
                    vals.append(float(accs[i]))
            if not vals:
                continue
            per_seed_means.append(float(np.mean(vals)))
            per_seed_medians.append(float(np.median(vals)))

        mean_of_means, range_of_means = _mean_and_range(per_seed_means)
        mean_of_medians, range_of_medians = _mean_and_range(per_seed_medians)

        return {
            "per_seed_means": per_seed_means,
            "per_seed_medians": per_seed_medians,
            "mean_of_means": mean_of_means,
            "range_of_means": range_of_means,
            "mean_of_medians": mean_of_medians,
            "range_of_medians": range_of_medians,
        }

    if run_logreg:
        lr_entries = [r for r in results if r["lr_test_mean"] is not None]
        lr_entries.sort(key=lambda x: x["lr_test_mean"])
        logger.info("Classification summary (Logistic Regression) sorted by mean test accuracy, worst first:")
        logger.info("Format: Dataset, Train size, Test size, Test Mean, Test Range(max-min), Test Accs(per seed)")

        for idx, entry in enumerate(lr_entries, start=1):
            logger.info(
                f"Rank {idx}: Dataset: {entry['dataset']}, Train size: {entry['train_size']}, Test size: {entry['test_size']}, "
                f"Test Mean: {entry['lr_test_mean']:.4f}, Test Range: {entry['lr_test_range']:.4f}, Accs: {entry['lr_test_accs']}"
            )

        stats = _overall_seed_dataset_stats("lr_test_accs")
        if stats["mean_of_means"] is not None:
            logger.info("\nOverall stats over datasets per seed (Logistic Regression):")
            logger.info(f"Seeds: {seeds}")
            logger.info(f"Per-seed dataset-mean acc: {[round(x, 6) for x in stats['per_seed_means']]}")
            logger.info(f"Mean over seeds (dataset-mean): {stats['mean_of_means']:.4f}")
            logger.info(f"Range over seeds (dataset-mean, max-min): {stats['range_of_means']:.4f}")
            logger.info(f"Per-seed dataset-median acc: {[round(x, 6) for x in stats['per_seed_medians']]}")
            logger.info(f"Mean over seeds (dataset-median): {stats['mean_of_medians']:.4f}")
            logger.info(f"Range over seeds (dataset-median, max-min): {stats['range_of_medians']:.4f}")

    if run_rf:
        rf_entries = [r for r in results if r["rf_test_mean"] is not None]
        rf_entries.sort(key=lambda x: x["rf_test_mean"])
        logger.info("\nClassification summary (Random Forest) sorted by mean test accuracy, worst first:")
        logger.info("Format: Dataset, Train size, Test size, Test Mean, Test Range(max-min), Test Accs(per seed)")

        for idx, entry in enumerate(rf_entries, start=1):
            logger.info(
                f"Rank {idx}: Dataset: {entry['dataset']}, Train size: {entry['train_size']}, Test size: {entry['test_size']}, "
                f"Test Mean: {entry['rf_test_mean']:.4f}, Test Range: {entry['rf_test_range']:.4f}, Accs: {entry['rf_test_accs']}"
            )

        stats = _overall_seed_dataset_stats("rf_test_accs")
        if stats["mean_of_means"] is not None:
            logger.info("\nOverall stats over datasets per seed (Random Forest):")
            logger.info(f"Seeds: {seeds}")
            logger.info(f"Per-seed dataset-mean acc: {[round(x, 6) for x in stats['per_seed_means']]}")
            logger.info(f"Mean over seeds (dataset-mean): {stats['mean_of_means']:.4f}")
            logger.info(f"Range over seeds (dataset-mean, max-min): {stats['range_of_means']:.4f}")
            logger.info(f"Per-seed dataset-median acc: {[round(x, 6) for x in stats['per_seed_medians']]}")
            logger.info(f"Mean over seeds (dataset-median): {stats['mean_of_medians']:.4f}")
            logger.info(f"Range over seeds (dataset-median, max-min): {stats['range_of_medians']:.4f}")

    if run_svm:
        svm_entries = [r for r in results if r["svm_test_mean"] is not None]
        svm_entries.sort(key=lambda x: x["svm_test_mean"])
        logger.info("\nClassification summary (SVM - Linear) sorted by mean test accuracy, worst first:")
        logger.info("Format: Dataset, Train size, Test size, Test Mean, Test Range(max-min), Test Accs(per seed)")

        for idx, entry in enumerate(svm_entries, start=1):
            logger.info(
                f"Rank {idx}: Dataset: {entry['dataset']}, Train size: {entry['train_size']}, Test size: {entry['test_size']}, "
                f"Test Mean: {entry['svm_test_mean']:.4f}, Test Range: {entry['svm_test_range']:.4f}, Accs: {entry['svm_test_accs']}"
            )

        stats = _overall_seed_dataset_stats("svm_test_accs")
        if stats["mean_of_means"] is not None:
            logger.info("\nOverall stats over datasets per seed (SVM - Linear):")
            logger.info(f"Seeds: {seeds}")
            logger.info(f"Per-seed dataset-mean acc: {[round(x, 6) for x in stats['per_seed_means']]}")
            logger.info(f"Mean over seeds (dataset-mean): {stats['mean_of_means']:.4f}")
            logger.info(f"Range over seeds (dataset-mean, max-min): {stats['range_of_means']:.4f}")
            logger.info(f"Per-seed dataset-median acc: {[round(x, 6) for x in stats['per_seed_medians']]}")
            logger.info(f"Mean over seeds (dataset-median): {stats['mean_of_medians']:.4f}")
            logger.info(f"Range over seeds (dataset-median, max-min): {stats['range_of_medians']:.4f}")

    if run_mlp:
        mlp_entries = [r for r in results if r["mlp_test_mean"] is not None]
        mlp_entries.sort(key=lambda x: x["mlp_test_mean"])
        logger.info("\nClassification summary (MLP - 1-layer) sorted by mean test accuracy, worst first:")
        logger.info("Format: Dataset, Train size, Test size, Test Mean, Test Range(max-min), Test Accs(per seed)")

        for idx, entry in enumerate(mlp_entries, start=1):
            logger.info(
                f"Rank {idx}: Dataset: {entry['dataset']}, Train size: {entry['train_size']}, Test size: {entry['test_size']}, "
                f"Test Mean: {entry['mlp_test_mean']:.4f}, Test Range: {entry['mlp_test_range']:.4f}, Accs: {entry['mlp_test_accs']}"
            )

        stats = _overall_seed_dataset_stats("mlp_test_accs")
        if stats["mean_of_means"] is not None:
            logger.info("\nOverall stats over datasets per seed (MLP - 1-layer):")
            logger.info(f"Seeds: {seeds}")
            logger.info(f"Per-seed dataset-mean acc: {[round(x, 6) for x in stats['per_seed_means']]}")
            logger.info(f"Mean over seeds (dataset-mean): {stats['mean_of_means']:.4f}")
            logger.info(f"Range over seeds (dataset-mean, max-min): {stats['range_of_means']:.4f}")
            logger.info(f"Per-seed dataset-median acc: {[round(x, 6) for x in stats['per_seed_medians']]}")
            logger.info(f"Mean over seeds (dataset-median): {stats['mean_of_medians']:.4f}")
            logger.info(f"Range over seeds (dataset-median, max-min): {stats['range_of_medians']:.4f}")

    if run_knn:
        knn_entries = [r for r in results if r["knn_test_mean"] is not None]
        knn_entries.sort(key=lambda x: x["knn_test_mean"])
        logger.info("\nClassification summary (KNN - k=1) sorted by mean test accuracy, worst first:")
        logger.info("Format: Dataset, Train size, Test size, Test Mean, Test Range(max-min), Test Accs(per seed)")

        for idx, entry in enumerate(knn_entries, start=1):
            logger.info(
                f"Rank {idx}: Dataset: {entry['dataset']}, Train size: {entry['train_size']}, Test size: {entry['test_size']}, "
                f"Test Mean: {entry['knn_test_mean']:.4f}, Test Range: {entry['knn_test_range']:.4f}, Accs: {entry['knn_test_accs']}"
            )

        stats = _overall_seed_dataset_stats("knn_test_accs")
        if stats["mean_of_means"] is not None:
            logger.info("\nOverall stats over datasets per seed (KNN - k=1):")
            logger.info(f"Seeds: {seeds}")
            logger.info(f"Per-seed dataset-mean acc: {[round(x, 6) for x in stats['per_seed_means']]}")
            logger.info(f"Mean over seeds (dataset-mean): {stats['mean_of_means']:.4f}")
            logger.info(f"Range over seeds (dataset-mean, max-min): {stats['range_of_means']:.4f}")
            logger.info(f"Per-seed dataset-median acc: {[round(x, 6) for x in stats['per_seed_medians']]}")
            logger.info(f"Mean over seeds (dataset-median): {stats['mean_of_medians']:.4f}")
            logger.info(f"Range over seeds (dataset-median, max-min): {stats['range_of_medians']:.4f}")

    if run_nc:
        nc_entries = [r for r in results if r["nc_test_mean"] is not None]
        nc_entries.sort(key=lambda x: x["nc_test_mean"])
        logger.info("\nClassification summary (Nearest Centroid) sorted by mean test accuracy, worst first:")
        logger.info("Format: Dataset, Train size, Test size, Test Mean, Test Range(max-min), Test Accs(per seed)")

        for idx, entry in enumerate(nc_entries, start=1):
            logger.info(
                f"Rank {idx}: Dataset: {entry['dataset']}, Train size: {entry['train_size']}, Test size: {entry['test_size']}, "
                f"Test Mean: {entry['nc_test_mean']:.4f}, Test Range: {entry['nc_test_range']:.4f}, Accs: {entry['nc_test_accs']}"
            )

        stats = _overall_seed_dataset_stats("nc_test_accs")
        if stats["mean_of_means"] is not None:
            logger.info("\nOverall stats over datasets per seed (Nearest Centroid):")
            logger.info(f"Seeds: {seeds}")
            logger.info(f"Per-seed dataset-mean acc: {[round(x, 6) for x in stats['per_seed_means']]}")
            logger.info(f"Mean over seeds (dataset-mean): {stats['mean_of_means']:.4f}")
            logger.info(f"Range over seeds (dataset-mean, max-min): {stats['range_of_means']:.4f}")
            logger.info(f"Per-seed dataset-median acc: {[round(x, 6) for x in stats['per_seed_medians']]}")
            logger.info(f"Mean over seeds (dataset-median): {stats['mean_of_medians']:.4f}")
            logger.info(f"Range over seeds (dataset-median, max-min): {stats['range_of_medians']:.4f}")

    if not results:
        logger.warning("No valid datasets to compute accuracy.")
