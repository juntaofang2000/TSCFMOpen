import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 你给的底色（从示例图估出来的主色）
BG_COLOR = "#D6DCE5"   # RGB(214, 220, 229)
LINE_COLOR = "#FF8C1A" # 橙色

def save_timeseries_image(
    y,
    out_path="timeseries.png",
    bg_color=BG_COLOR,
    line_color=LINE_COLOR,
    rounded_panel=True,    # True: 圆角底板 + 黑边；False: 纯色背景
    border_color="black",
    border_width=3.0,
    figsize=(3.2, 2.2),
    dpi=300,
    draw_line=False,
):
    y = np.asarray(y).ravel()
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 去掉坐标轴（更像论文示意图的小图块）
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 背景处理
    if rounded_panel:
        # 让 figure/axes 透明，用圆角面板当背景
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        panel = FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            transform=ax.transAxes,
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=0,
            clip_on=False,
        )
        ax.add_patch(panel)
    else:
        # 纯色背景
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

    # 画时间序列（可选）
    if draw_line:
        ax.plot(x, y, color=line_color, linewidth=2.5, zorder=2)

        # 留一点边距，让线条不贴边
        ax.margins(x=0.06, y=0.20)

    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight", transparent=rounded_panel)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # 示例：生成一条“像时序”的曲线（你也可以换成自己的 y 数组）
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.15, size=120)) + 0.5*np.sin(np.linspace(0, 6*np.pi, 120))

    save_timeseries_image(
        y,
        out_path="timeseries.png",
        rounded_panel=True,   # 改成 False 就是纯背景不带圆角边框
    )
