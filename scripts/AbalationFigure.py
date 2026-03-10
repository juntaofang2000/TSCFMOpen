import matplotlib.pyplot as plt
import numpy as np


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
        "axes.labelsize": 7,
        "font.size": 8,
        "legend.fontsize": 5,
        "legend.frameon": True,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
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

# Data
labels = ["w/o ICL", "w/o ens.",  "TIC-FM"]
acc = np.array([ 78.13, 79.42, 80.01])

setup_mlp_config()
fig, ax = plt.subplots()

x = np.arange(len(labels))

# Use matplotlib default color cycle, then soften it via alpha
default_blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

bars = ax.bar(
    x,
    acc,
    width=0.62,
    color=default_blue,
    alpha=0.25,          # lighter / more paper-like
    edgecolor="black",
    linewidth=0.8,
)

# Highlight the full model: same hue, but stronger alpha + hatch
bars[-1].set_alpha(0.35)
bars[-1].set_hatch("///")

ax.set_ylabel("Average accuracy (%)")

# Slightly wider y-range for readability
ymin = np.floor(acc.min()) - 1
ymax = np.ceil(acc.max()) + 1
ax.set_ylim(ymin, ymax)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha="center")

# Light y-grid only
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels (small and close)
for rect, yi in zip(bars, acc):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        yi + 0.05,
        f"{yi:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

fig.savefig("ablation_bar.svg", bbox_inches="tight")
fig.savefig("ablation_bar.pdf", bbox_inches="tight")
plt.show()