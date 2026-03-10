import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def compute_figsize(rel_width, height_to_width_ratio, nrows=1, ncols=1):
    base_width_in = 5.5
    width_in = base_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    return width_in, height_in


def setup_mlp_config():
    import matplotlib as mpl

    config = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "NewComputerModernSans10",
            "NewComputerModernMath",
            "Encode Sans",
            "Ubuntu",
            "Liberation Sans",
        ],
        "axes.labelsize": 9,
        "font.size": 10,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": compute_figsize(rel_width=0.5, height_to_width_ratio=0.7, nrows=1, ncols=1),
        "figure.dpi": 400,
        "figure.autolayout": False,
        "figure.titlesize": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.05,
        "axes.titlesize": 8,
        "legend.labelspacing": 0.2,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "legend.title_fontsize": 6,
        "savefig.dpi": 500,
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "legend.markerscale": 0.5,
        "savefig.format": "pdf",
        "figure.subplot.wspace": 0.15,
        "figure.subplot.hspace": 0.1,
        "figure.constrained_layout.wspace": 0.01,
        "figure.constrained_layout.hspace": 0.01,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.h_pad": 0.02,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "axes.labelpad": 0.0,
        "xtick.major.pad": 1,
        "xtick.major.size": 3,
        "ytick.major.pad": 1,
        "ytick.major.size": 3,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(config)

# x-axis: labeled fraction
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# y-axis: average accuracy (%)
moment_rf = [v * 100 for v in [0.691888, 0.746018, 0.760073, 0.783858, 0.787844, 0.80298]]
moment_svm = [v * 100 for v in [0.7013, 0.753038, 0.76962, 0.786891, 0.783647, 0.79876]]
mantis_rf = [v * 100 for v in [0.6957, 0.7492, 0.7687, 0.7871, 0.7923, 0.8092]]
mantis_svm = [v * 100 for v in [0.7117, 0.7482, 0.7715, 0.7845, 0.7945, 0.8079]]
tic_fm = [v * 100 for v in [0.723, 0.7649, 0.7866, 0.8022, 0.8119, 0.8228]]

setup_mlp_config()

fig, ax = plt.subplots()

ax.plot(
    fractions,
    moment_rf,
    color="#1f77b4",
    marker="o",
    linewidth=2.0,
    markersize=5.5,
    alpha=0.95,
    label="MOMENT+RF",
)
ax.plot(
    fractions,
    moment_svm,
    color="#ff7f0e",
    marker="s",
    linewidth=2.0,
    markersize=5.5,
    alpha=0.95,
    label="MOMENT+SVM",
)
ax.plot(
    fractions,
    mantis_rf,
    color="#2ca02c",
    marker="^",
    linewidth=2.0,
    markersize=5.5,
    alpha=0.95,
    label="Mantis+RF",
)
ax.plot(
    fractions,
    mantis_svm,
    color="#d62728",
    marker="D",
    linewidth=2.0,
    markersize=5.5,
    alpha=0.95,
    label="Mantis+SVM",
)
ax.plot(
    fractions,
    tic_fm,
    color="#9467bd",
    marker="o",
    linewidth=2.0,
    markersize=5.5,
    alpha=0.95,
    label="TIC-FM",
)

ax.set_xlabel("Labeled fraction")
ax.set_ylabel("Average accuracy (%)")

ax.set_xticks(fractions)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x * 100))}%"))

all_y = moment_rf + moment_svm + mantis_rf + mantis_svm + tic_fm
y_min, y_max = min(all_y), max(all_y)
pad = 0.8
ax.set_ylim(y_min - pad, y_max + pad)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.35)

ax.legend(loc="lower right", frameon=False, handlelength=2.2)

fig.tight_layout(pad=0.35)

fig.savefig("train_fraction_scaling.pdf", bbox_inches="tight")
fig.savefig("train_fraction_scaling.svg", bbox_inches="tight")
plt.show()
