import numpy as np
import matplotlib.pyplot as plt

m = 30
emb = np.random.randn(m)  # 用你的 embedding 替换这里，shape: (m,)

# 可选：标准化（更稳定的颜色分布）
# emb = (emb - emb.mean()) / (emb.std() + 1e-8)

# 渐变蓝色条（从浅到深的蓝）
plt.figure(figsize=(10, 1.0))
ax = plt.gca()

ax.imshow(
    emb[np.newaxis, :],
    aspect="auto",
    cmap="Greens",          # 蓝色渐变
    interpolation="nearest"
)

# 去掉所有刻度线和数字标注
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")

# 去掉边框（可选，但更像论文里的纯色条）
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0)
plt.savefig("embedding_plotGreen.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()

# ====== 3) 画法 B：折线/点图（看数值走势更直观）======
plt.figure(figsize=(10, 2.5))
plt.plot(np.arange(m), emb, marker=".", linewidth=1)
plt.axhline(0, linewidth=1)
plt.xlabel("Dimension index")
plt.ylabel("Value")
plt.title(f"Embedding (1×{m}) as line plot")
plt.tight_layout()
plt.show()
# 如果想保存图片，可以用 plt.savefig("embedding_plot.png", dpi=300)