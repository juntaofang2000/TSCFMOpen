import re
from pathlib import Path
import argparse
import numpy as np


# ============== 1) paths (defaults; overridable via CLI) ==============
CODE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = CODE_DIR / "log"

MOMENT_FILE_DEFAULT = LOG_DIR / "ucr_2018_moment_resultsFinal"
MANTIS_FILE_DEFAULT = LOG_DIR / "results_testMantisHugganFace_20260117.log"
TICFM_FILE_DEFAULT = LOG_DIR / "eval_mantis_orion_icl_adapter_ckpt_ucr_clssfierv2202601181107.log"


def _parse_args():
    p = argparse.ArgumentParser(description="Compute Avg/Med accuracy and AvgRank across methods (UCR128).")
    p.add_argument("--moment", type=Path, default=MOMENT_FILE_DEFAULT, help="Path to MOMENT result file")
    p.add_argument("--mantis", type=Path, default=MANTIS_FILE_DEFAULT, help="Path to Mantis result file")
    p.add_argument("--ticfm", type=Path, default=TICFM_FILE_DEFAULT, help="Path to TIC-FM result file")
    p.add_argument("--include-lr", action="store_true", help="Include Logistic Regression if available")
    p.add_argument(
        "--median-mode",
        choices=["log", "computed"],
        default="log",
        help=(
            "How to compute Median. "
            "'log' uses the log's 'Mean over seeds (dataset-median)' when available (mean of per-seed dataset medians). "
            "'computed' uses median over per-dataset mean accuracies."
        ),
    )

    p.add_argument(
        "--dataset-table",
        action="store_true",
        help="Print a LaTeX table with per-dataset mean±std for selected methods.",
    )
    p.add_argument(
        "--dataset-table-out",
        type=Path,
        default=None,
        help="Write the per-dataset LaTeX table to this file instead of stdout.",
    )
    p.add_argument(
        "--dataset-table-log",
        type=Path,
        default=LOG_DIR / "per_dataset_mean_range_bold.tex",
        help="Write the per-dataset LaTeX table to this log file (default in code/log).",
    )
    p.add_argument(
        "--dataset-table-percent",
        action="store_true",
        help="Format per-dataset mean/std as percent (x100). Default prints raw acc in [0,1].",
    )
    return p.parse_args()


def _parse_overall_seed_medians_from_log(path: Path, model: str):
    """Parse 'Mean over seeds (dataset-median): <float>' blocks.

    Returns: dict head -> float (0..1)

    model:
      - 'moment': headers like 'Overall stats over datasets per seed (RandomForest):'
      - 'mantis': headers like 'Overall stats over datasets per seed (MLP - 1-layer):'
    """
    if model == "moment":
        head_map = {
            "LogisticRegression": "LR",
            "RandomForest": "RF",
            "SVM (fit_svm: RBF+CV)": "SVM",
            "LinearHead(MLP-1-layer)": "MLP",
            "KNN(k=1)": "kNN",
            "NearestCentroid": "NC",
        }
    elif model == "mantis":
        head_map = {
            "Logistic Regression": "LR",
            "Random Forest": "RF",
            "SVM - Linear": "SVM",
            "MLP - 1-layer": "MLP",
            "KNN - k=1": "kNN",
            "Nearest Centroid": "NC",
        }
    else:
        raise ValueError(f"Unknown model={model!r}")

    header_re = re.compile(r"^Overall stats over datasets per seed \((.+?)\):\s*$")
    median_re = re.compile(r"^Mean over seeds \(dataset-median\):\s*([0-9]*\.?[0-9]+)\s*$")

    out = {}
    cur = None
    for line in path.read_text(errors="ignore").splitlines():
        mh = header_re.match(line.strip())
        if mh:
            cur = head_map.get(mh.group(1).strip())
            continue
        if cur is None:
            continue
        mm = median_re.match(line.strip())
        if mm:
            out[cur] = float(mm.group(1))
            cur = None
    return out


# ============== 2) parsing helpers ==============
def parse_moment(path: Path):
    """
    Parse MOMENT log lines like:
    [1/128] ACSF1 ... logreg=0.5400±... rf=0.8040±... svm=0.6800±... mlp=...
    Return: dict head -> {dataset -> acc_float}
    """
    ms = parse_moment_mean_std(path)
    return {head: {ds: mean for ds, (mean, _std) in ds_map.items()} for head, ds_map in ms.items()}


def parse_moment_mean_maxmin(path: Path):
    """Parse MOMENT per-dataset mean±(max-min) from the [i/128] lines.

    Returns: dict head -> {dataset -> (mean, maxmin)}
    """
    heads = {"logreg": "LR", "rf": "RF", "svm": "SVM", "mlp": "MLP", "knn": "kNN", "nc": "NC"}
    out = {v: {} for v in heads.values()}

    line_re = re.compile(r"^\[\d+/\d+\]\s+(\S+)")
    kv_re = re.compile(r"\b(logreg|rf|svm|mlp|knn|nc)=([0-9.]+)±([0-9.]+)")

    for line in path.read_text(errors="ignore").splitlines():
        m = line_re.match(line)
        if not m:
            continue
        ds = m.group(1)
        for k, mean_s, maxmin_s in kv_re.findall(line):
            head = heads.get(k)
            if head is None:
                continue
            out[head][ds] = (float(mean_s), float(maxmin_s))

    return out


def parse_mantis(path: Path):
    """
    Parse Mantis log sections like:
    Classification summary (Random Forest) ...
    Rank 1: Dataset: Phoneme, ... Test Mean: 0.2911, ...
    Return: dict head -> {dataset -> acc_float}
    """
    name_map = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "SVM - Linear": "SVM",
        "MLP - 1-layer": "MLP",
        "KNN - k=1": "kNN",
        "Nearest Centroid": "NC",
    }

    ms = parse_mantis_mean_std(path)
    return {head: {ds: mean for ds, (mean, _std) in ds_map.items()} for head, ds_map in ms.items()}


def parse_mantis_mean_maxmin(path: Path):
    """Parse Mantis per-dataset mean and (max-min) across seeds.

    Prefer parsing from the per-seed list:
      Rank ... Test Mean: 0.2911, ... Accs: [0.29, 0.30, ...]

    Returns: dict head -> {dataset -> (mean, maxmin)}
    """
    name_map = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "SVM - Linear": "SVM",
        "MLP - 1-layer": "MLP",
        "KNN - k=1": "kNN",
        "Nearest Centroid": "NC",
    }

    out = {v: {} for v in name_map.values()}
    cur = None

    header_re = re.compile(r"^Classification summary \((.+?)\)\s*")
    row_re_with_accs = re.compile(
        r"^Rank\s+\d+:\s+Dataset:\s+([^,]+),.*Test Mean:\s+([0-9.]+),.*Accs:\s+\[([^\]]+)\]"
    )
    row_re_mean_only = re.compile(r"^Rank\s+\d+:\s+Dataset:\s+([^,]+),.*Test Mean:\s+([0-9.]+),")

    for line in path.read_text(errors="ignore").splitlines():
        mh = header_re.match(line)
        if mh:
            title = mh.group(1).strip()
            cur = name_map.get(title, None)
            continue

        if cur is None:
            continue

        mr = row_re_with_accs.match(line)
        if mr:
            ds = mr.group(1).strip()
            mean = float(mr.group(2))
            accs_raw = mr.group(3)
            try:
                accs = [float(x.strip()) for x in accs_raw.split(",") if x.strip()]
            except ValueError:
                accs = []

            if len(accs) > 0:
                mean = float(np.mean(accs))
                maxmin = float(np.max(accs) - np.min(accs))
            else:
                maxmin = 0.0

            out[cur][ds] = (mean, maxmin)
            continue

        mr2 = row_re_mean_only.match(line)
        if mr2:
            ds = mr2.group(1).strip()
            mean = float(mr2.group(2))
            out[cur][ds] = (mean, 0.0)

    return out


def parse_ticfm(path: Path):
    """
    Parse TIC-FM log lines like:
    Earthquakes: 0.7482
    Return: {dataset -> acc_float}
    """
    out = {}
    # dataset lines are "TokenNoSpace: float" and end exactly with float
    re_ds = re.compile(r"^([A-Za-z0-9_]+):\s+([0-9]*\.[0-9]+|[01])\s*$")

    for line in path.read_text(errors="ignore").splitlines():
        m = re_ds.match(line.strip())
        if m:
            ds, val = m.group(1), float(m.group(2))
            out[ds] = val

    return out


# ============== 3) ranking (1=best; ties get average rank) ==============
def rank_desc_with_ties(values):
    """
    values: list/np.array of floats, higher is better.
    return: np.array ranks (float), 1=best, ties -> average rank.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="mergesort")  # stable
    sorted_vals = values[order]

    ranks_sorted = np.empty_like(sorted_vals, dtype=float)
    i, n = 0, len(sorted_vals)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # positions i..j (0-based) correspond to ranks (i+1)..(j+1)
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks_sorted[i : j + 1] = avg_rank
        i = j + 1

    ranks = np.empty_like(ranks_sorted, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


# ============== 4) load per-dataset accuracies ==============
args = _parse_args()

for label, p in [("MOMENT", args.moment), ("Mantis", args.mantis), ("TIC-FM", args.ticfm)]:
    if not p.exists():
        raise FileNotFoundError(f"{label} file not found: {p} (cwd={Path.cwd()})")
    if not p.is_file():
        raise IsADirectoryError(f"{label} path is not a file: {p}")

moment_ms = parse_moment_mean_maxmin(args.moment)
mantis_ms = parse_mantis_mean_maxmin(args.mantis)
moment = {head: {ds: mean for ds, (mean, _std) in ds_map.items()} for head, ds_map in moment_ms.items()}
mantis = {head: {ds: mean for ds, (mean, _std) in ds_map.items()} for head, ds_map in mantis_ms.items()}
ticfm  = parse_ticfm(args.ticfm)

moment_seed_medians = _parse_overall_seed_medians_from_log(args.moment, model="moment") if args.median_mode == "log" else {}
mantis_seed_medians = _parse_overall_seed_medians_from_log(args.mantis, model="mantis") if args.median_mode == "log" else {}

# methods in your table
METHODS = {
    "MOMENT + RF":  moment["RF"],
    "MOMENT + SVM": moment["SVM"],
    "MOMENT + MLP": moment["MLP"],
    "MOMENT + kNN": moment["kNN"],
    "MOMENT + NC":  moment["NC"],
    "Mantis + RF":  mantis["RF"],
    "Mantis + SVM": mantis["SVM"],
    "Mantis + MLP": mantis["MLP"],
    "Mantis + kNN": mantis["kNN"],
    "Mantis + NC":  mantis["NC"],
    "TIC-FM":       ticfm,
}

if args.include_lr:
    if "LR" in moment and len(moment["LR"]) > 0:
        METHODS["MOMENT + LR"] = moment["LR"]
    if "LR" in mantis and len(mantis["LR"]) > 0:
        METHODS["Mantis + LR"] = mantis["LR"]

# common dataset set (should be 128)
key_sets = [set(d.keys()) for d in METHODS.values()]
common = set.intersection(*key_sets) if key_sets else set()
common = sorted(common)
print(f"[Info] common datasets = {len(common)}")

if len(common) == 0:
    sizes = {m: len(d) for m, d in METHODS.items()}
    raise RuntimeError(
        "No common datasets across methods. "
        f"Parsed counts per method: {sizes}. "
        "This usually means dataset names don't match across logs or a parser regex didn't match."
    )


# ============== 5) compute avg/med + avg rank ==============
def mean_percent(x):   return 100.0 * float(np.mean(x))
def median_percent(x): return 100.0 * float(np.median(x))

# per-method stats
stats = {}
for m, d in METHODS.items():
    arr = np.array([d[ds] for ds in common], dtype=float)

    # Median definition:
    # - computed: median over per-dataset mean accuracies (arr)
    # - log: mean over seeds of per-seed dataset medians, as printed in logs
    med = median_percent(arr)
    if args.median_mode == "log":
        if m.startswith("MOMENT + "):
            head = m.split(" + ", 1)[1]
            if head in moment_seed_medians:
                med = 100.0 * float(moment_seed_medians[head])
        elif m.startswith("Mantis + "):
            head = m.split(" + ", 1)[1]
            if head in mantis_seed_medians:
                med = 100.0 * float(mantis_seed_medians[head])

    stats[m] = {
        "avg_acc": mean_percent(arr),
        "med_acc": med,
        "avg_rank": None,  # filled later
    }

# ranks per dataset across all methods in METHODS
method_list = list(METHODS.keys())
acc_matrix = np.array([[METHODS[m][ds] for m in method_list] for ds in common], dtype=float)  # [128, M]

ranks = np.vstack([rank_desc_with_ties(acc_matrix[i]) for i in range(acc_matrix.shape[0])])  # [128, M]
avg_ranks = ranks.mean(axis=0)

for m, r in zip(method_list, avg_ranks.tolist()):
    stats[m]["avg_rank"] = float(r)

# print summary
print("\n=== Summary (computed) ===")
for m in method_list:
    s = stats[m]
    print(f"{m:12s}  Avg={s['avg_acc']:.2f}%  Med={s['med_acc']:.2f}%  AvgRank={s['avg_rank']:.3f}")


# ============== 6) optional: verify against your table numbers ==============
REF = {
    "MOMENT + RF":  (77.51, 81.64),
    "MOMENT + SVM": (77.98, 80.16),
    "MOMENT + MLP": (44.51, 45.54),
    "MOMENT + kNN": (75.71, 79.10),
    "MOMENT + NC":  (66.05, 66.27),
    "Mantis + RF":  (78.67, 80.36),
    "Mantis + SVM": (78.95, 82.34),
    "Mantis + MLP": (63.53, 65.58),
    "Mantis + kNN": (77.07, 80.69),
    "Mantis + NC":  (70.53, 71.14),
    "TIC-FM":       (80.01, 82.06),
}

print("\n=== Diff check vs your table (Avg, Med) ===")
for m, (ref_avg, ref_med) in REF.items():
    d_avg = stats[m]["avg_acc"] - ref_avg
    d_med = stats[m]["med_acc"] - ref_med
    print(f"{m:12s}  dAvg={d_avg:+.2f}  dMed={d_med:+.2f}")


# ============== 7) print LaTeX rows with Avg Rank filled ==============
print("\n=== LaTeX rows (with Avg Rank) ===")
for m in method_list:
    s = stats[m]
    print(f"{m} & {s['avg_acc']:.2f}\\% & {s['med_acc']:.2f}\\% & {s['avg_rank']:.2f} \\\\")


def _latex_escape(text: str) -> str:
    # Minimal escaping for UCR dataset names.
    return (
        text.replace("\\", "\\\\")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )



def _fmt_mean_maxmin(mean: float, maxmin: float, as_percent: bool, bold: bool = False) -> str:
    if as_percent:
        mean_fmt = f"{mean*100.0:.4f}"
    else:
        mean_fmt = f"{mean:.4f}"
    cell = f"{mean_fmt}"
    if bold:
        return r"\textbf{" + cell + "}"
    return cell


if args.dataset_table:
    # Exactly the methods requested by the user.
    dataset_methods = [
        ("MOMENT + RF", moment.get("RF", {})),
        ("MOMENT + SVM", moment.get("SVM", {})),
        ("MOMENT + MLP", moment.get("MLP", {})),
        ("MOMENT + kNN", moment.get("kNN", {})),
        ("MOMENT + NC", moment.get("NC", {})),
        ("Mantis + RF", mantis.get("RF", {})),
        ("Mantis + SVM", mantis.get("SVM", {})),
        ("Mantis + MLP", mantis.get("MLP", {})),
        ("Mantis + kNN", mantis.get("kNN", {})),
        ("Mantis + NC", mantis.get("NC", {})),
        ("TIC-FM", ticfm),
    ]

    ds_sets = [set(d.keys()) for _name, d in dataset_methods]
    common_ds = set.intersection(*ds_sets) if ds_sets else set()
    common_ds = sorted(common_ds)

    lines = []
    header = "Dataset" + "".join([f" & {_latex_escape(name)}" for name, _ in dataset_methods]) + r" \\"
    lines.append(header)
    lines.append(r"\\hline")

    for ds in common_ds:
        row = [_latex_escape(ds)]
        # 先收集所有方法的 mean
        means = [d[ds] for _name, d in dataset_methods]
        max_mean = max(means)
        # 允许并列最优
        is_best = [abs(m - max_mean) < 1e-10 for m in means]
        for (_name, d), best in zip(dataset_methods, is_best):
            mean = d[ds]
            row.append(_fmt_mean_maxmin(mean, 0.0, as_percent=args.dataset_table_percent, bold=best))
        lines.append(" & ".join(row) + " \\\\")

    out_text = "\n".join(lines) + "\n"
    log_path = args.dataset_table_out if args.dataset_table_out is not None else args.dataset_table_log
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out_text)
        print(f"\n[Info] Wrote per-dataset LaTeX table to: {log_path}")
    if args.dataset_table_out is None:
        print("\n=== LaTeX per-dataset mean±std table ===")
        print(out_text, end="")
