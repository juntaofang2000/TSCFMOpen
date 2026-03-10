import matplotlib.pyplot as plt
import numpy as np
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

# multipliers m = N_ctx / K
m = np.array([1, 5, 10, 15, 20], dtype=float)

# Crop (C=24, K=240)
acc_crop = np.array([0.5391, 0.6196, 0.6397, 0.6470, 0.6502]) * 100

#  ElectricDevices (C=7, K=70)
acc_ed = np.array([0.5897, 0.7786, 0.8124, 0.8132, 0.8219]) * 100

# ECG5000
acc_ECG = np.array([0.6969, 0.8653, 0.9009, 0.9173, 0.9227]) * 100

setup_mlp_config()

fig, ax = plt.subplots()

# Plot mean accuracy curves (no std / error bars)
ax.plot(
    m,
    acc_crop,
    color="#1f77b4",
    marker="o",
    linewidth=2.2,
    markersize=5.5,
    alpha=0.95,
    label="Crop",
)
ax.plot(
    m,
    acc_ed,
    color="#ff7f0e",
    marker="s",
    linewidth=2.2,
    markersize=5.5,
    alpha=0.95,
    label="ElectricDevices",
)
ax.plot(
    m,
    acc_ECG,
    color="#2ca02c",
    marker="^",
    linewidth=2.2,
    markersize=5.5,
    alpha=0.95,
    label="ECG5000",
)

ax.set_xscale("log")
ax.set_xticks(m)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}×" if x in m else ""))

ax.set_xlabel(r"Context multiplier $m=N_{\mathrm{ctx}}/N_{0}$")
ax.set_ylabel("Accuracy (%)")

all_y = np.concatenate([acc_crop, acc_ed, acc_ECG])
ax.set_ylim(all_y.min() - 1.0, all_y.max() + 1.0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), frameon=False, handlelength=2.2)

fig.tight_layout(pad=0.35)

fig.savefig("context_length_scaling.svg", bbox_inches="tight")
fig.savefig("context_length_scaling.pdf", bbox_inches="tight")
plt.show()
