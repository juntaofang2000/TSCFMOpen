import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize

# ====== data ======
m = 30
emb = np.random.randn(m)  # 替换成你的 embedding, shape: (m,)

# 可选：标准化（让颜色分布更稳定）
# emb = (emb - emb.mean()) / (emb.std() + 1e-8)

# ====== make smooth gradient along width (upsample) ======
upsample = 20  # 越大越平滑
x = np.arange(m)
x_new = np.linspace(0, m - 1, m * upsample)
emb_smooth = np.interp(x_new, x, emb)             # shape: (m*upsample,)
img = emb_smooth[np.newaxis, :]                   # shape: (1, m*upsample)

# ====== plot ======
fig, ax = plt.subplots(figsize=(6, 1.0), dpi=300)
ax.set_axis_off()

# 归一化到 colormap（蓝色渐变）
norm = Normalize(vmin=img.min(), vmax=img.max())

im = ax.imshow(
    img,
    aspect="auto",
    cmap="Blues",          # 蓝色渐变
    norm=norm,
    interpolation="bicubic"  # 更平滑（可改成 'bilinear'）
)

# ====== rounded "capsule" clip ======
# 在 axes 坐标系(0~1)里做一个圆角矩形作为裁剪区域
clip = FancyBboxPatch(
    (0.02, 0.25), 0.96, 0.5,  # (x, y, w, h) in Axes coords
    boxstyle="round,pad=0.02,rounding_size=0.25",
    transform=ax.transAxes,
    linewidth=0.0,
    facecolor="none"
)
ax.add_patch(clip)
im.set_clip_path(clip)

# 可选：加一个很淡的边框（更像论文图），不想要就注释掉
border = FancyBboxPatch(
    (0.02, 0.25), 0.96, 0.5,
    boxstyle="round,pad=0.02,rounding_size=0.25",
    transform=ax.transAxes,
    linewidth=1.0,
    edgecolor=(0, 0, 0, 0.15),
    facecolor="none"
)
ax.add_patch(border)

# ====== save (no background) ======
plt.savefig("embedding_bar_blue.png", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()
