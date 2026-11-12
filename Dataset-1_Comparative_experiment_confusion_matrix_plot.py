# """原始绘图代码"""
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ========================================
# # 1. 配置参数（根据需求修改）
# # ========================================
# # 文件路径（替换为你的两个混淆矩阵CSV文件）
# FILE_PATH_METHOD1 = 'evaluation_results_GIGNN.csv'  # 方法1的混淆矩阵（无标题行）
# FILE_PATH_METHOD2 = 'evaluation_results_GCTGNN.csv'  # 方法2的混淆矩阵（无标题行）
# plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
# plt.rcParams['font.size'] = 16  # 可选：设置字体大小
# CLASS_COUNT = 16                   # 类别数量（你的数据是0-15，共16类）
# CLASS_LABELS = [f'Fault {i}' for i in range(CLASS_COUNT)]  # 类别标签（可自定义，如['Cat', 'Dog']）
# FIG_SIZE = (16, 8)                 # 图表大小（宽×高，根据类别数量调整）
# TITLE_FONT_SIZE = 18               # 总标题字体大小
# SUBTITLE_FONT_SIZE = 16            # 子图标题字体大小
# LABEL_FONT_SIZE = 16               # 坐标轴标签字体大小
#
#
# # ========================================
# # 2. 读取并预处理数据（关键：转换为整数）
# # ========================================
# # 读取CSV（无标题行），填充缺失值（如果有），转换为整数
# def load_confusion_matrix(file_path):
#     return pd.read_csv(file_path, header=None).fillna(0).astype(int)
#
# # 加载两个方法的混淆矩阵
# cm_method1 = load_confusion_matrix(FILE_PATH_METHOD1)
# cm_method2 = load_confusion_matrix(FILE_PATH_METHOD2)
#
#
# # ========================================
# # 3. 创建子图布局（1行2列，并排显示）
# # ========================================
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
#
# # 蓝色Blues 绿色Greens
#
# # ========================================
# # 4. 绘制第一个热力图（方法1）
# # ========================================
# sns.heatmap(
#     data=cm_method1,               # 混淆矩阵数据（已转换为整数）
#     ax=ax1,                        # 指定子图 axes
#     annot=True,                    # 显示单元格内的数值
#     fmt='d',                       # 数值格式（整数）
#     cmap='Blues',                 # 渐变颜色（黄橙棕，适合显示计数）
#     xticklabels=CLASS_LABELS,      # x轴标签（预测类别）
#     yticklabels=CLASS_LABELS,      # y轴标签（真实类别）
#     cbar_kws={'label': 'Sample Count', 'shrink': 0.85},  # 颜色条（缩小比例，避免超出图框）
#     vmin=0,                        # 颜色条最小值（固定为0，符合计数逻辑）
#     vmax=max(cm_method1.max().max(), cm_method2.max().max())  # 颜色条最大值（统一两个图的范围，便于对比）
# )
# # 设置子图1的标题和标签
# ax1.set_xlabel('Predicted Class', fontsize=LABEL_FONT_SIZE)
# ax1.set_ylabel('True Class', fontsize=LABEL_FONT_SIZE)
# ax1.tick_params(axis='x', rotation=45, labelsize=16)  # 旋转x轴标签（避免重叠），缩小字体
# ax1.tick_params(axis='y', labelsize=16)
#
#
# # ========================================
# # 5. 绘制第二个热力图（方法2）
# # ========================================
# sns.heatmap(
#     data=cm_method2,               # 方法2的混淆矩阵数据
#     ax=ax2,                        # 指定子图 axes
#     annot=True,
#     fmt='d',
#     cmap='Blues',                 # 渐变颜色（紫蓝绿，与方法1区分）
#     xticklabels=CLASS_LABELS,
#     yticklabels=CLASS_LABELS,
#     cbar_kws={'label': 'Sample Count', 'shrink': 0.85},
#     vmin=0,
#     vmax=max(cm_method1.max().max(), cm_method2.max().max())  # 统一颜色范围
# )
# # 设置子图2的标题和标签
# ax2.set_xlabel('Predicted Class', fontsize=LABEL_FONT_SIZE)
# ax2.set_ylabel('True Class', fontsize=LABEL_FONT_SIZE)
# ax2.tick_params(axis='x', rotation=45, labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)
#
#
# # ========================================
# # 6. 调整布局（避免重叠）
# # ========================================
# plt.tight_layout()  # 调整总标题的位置（底部留白，避免覆盖子图标题）
# plt.show()  # 显示图表
#
#
# """ 方案 1：PowerNorm（推荐，增强小值但保留 """
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import matplotlib.colors as mcolors
#
# # ========================================
# # 1. 配置参数（根据需求修改）
# # ========================================
# FILE_PATH_METHOD1 = 'evaluation_results_GIGNN.csv'
# FILE_PATH_METHOD2 = 'evaluation_results_GCTGNN.csv'
#
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 16
#
# CLASS_COUNT = 16
# CLASS_LABELS = [f'Fault {i}' for i in range(CLASS_COUNT)]
# FIG_SIZE = (16, 8)
# LABEL_FONT_SIZE = 16
#
# # PowerNorm参数：gamma 越小，小值越“显色”
# GAMMA = 0.5
#
# # 可选：用分位数限制 vmax（避免极端最大值统治颜色）
# USE_PERCENTILE_VMAX = True
# VMAX_PERCENTILE = 99  # 98或99都可以试
#
# # ========================================
# # 2. 读取并预处理数据
# # ========================================
# def load_confusion_matrix(file_path):
#     return pd.read_csv(file_path, header=None).fillna(0).astype(int)
#
# cm_method1 = load_confusion_matrix(FILE_PATH_METHOD1)
# cm_method2 = load_confusion_matrix(FILE_PATH_METHOD2)
#
# # 统一vmax
# all_vals = np.concatenate([cm_method1.values.ravel(), cm_method2.values.ravel()])
# if USE_PERCENTILE_VMAX:
#     vmax = np.percentile(all_vals, VMAX_PERCENTILE)
# else:
#     vmax = all_vals.max()
#
# # PowerNorm（非线性映射）
# norm = mcolors.PowerNorm(gamma=GAMMA, vmin=0, vmax=vmax)
#
# # ========================================
# # 3. 创建子图布局
# # ========================================
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
#
# # ========================================
# # 4. 绘图（方法1）
# # ========================================
# sns.heatmap(
#     data=cm_method1,
#     ax=ax1,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=CLASS_LABELS,
#     yticklabels=CLASS_LABELS,
#     cbar_kws={'label': 'Sample Count', 'shrink': 0.85},
#     norm=norm
# )
# ax1.set_xlabel('Predicted Class', fontsize=LABEL_FONT_SIZE)
# ax1.set_ylabel('True Class', fontsize=LABEL_FONT_SIZE)
# ax1.tick_params(axis='x', rotation=45, labelsize=14)
# ax1.tick_params(axis='y', labelsize=14)
#
# # ========================================
# # 5. 绘图（方法2）
# # ========================================
# sns.heatmap(
#     data=cm_method2,
#     ax=ax2,
#     annot=True,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=CLASS_LABELS,
#     yticklabels=CLASS_LABELS,
#     cbar_kws={'label': 'Sample Count', 'shrink': 0.85},
#     norm=norm
# )
# ax2.set_xlabel('Predicted Class', fontsize=LABEL_FONT_SIZE)
# ax2.set_ylabel('True Class', fontsize=LABEL_FONT_SIZE)
# ax2.tick_params(axis='x', rotation=45, labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)
#
# plt.tight_layout()
# plt.show()
#
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import matplotlib.colors as mcolors
#
# # ========================================
# # 1. 配置参数（根据需求修改）
# # ========================================
# FILE_PATH_METHOD1 = 'evaluation_results_MAG.csv'
# FILE_PATH_METHOD2 = 'evaluation_results_LSTM.csv'
#
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 16
#
# CLASS_COUNT = 16
# CLASS_LABELS = [f'Fault {i}' for i in range(CLASS_COUNT)]
# FIG_SIZE = (16, 8)
# LABEL_FONT_SIZE = 16
#
# # ========================================
# # 2. 读取并预处理数据
# # ========================================
# def load_confusion_matrix(file_path):
#     return pd.read_csv(file_path, header=None).fillna(0).astype(int)
#
# cm_method1 = load_confusion_matrix(FILE_PATH_METHOD1)
# cm_method2 = load_confusion_matrix(FILE_PATH_METHOD2)
#
# # 关键：绘图数据 +1，避免log(0)；但 annot 显示原始计数
# plot1 = cm_method1 + 1
# plot2 = cm_method2 + 1
#
# all_vals_plus1 = np.concatenate([plot1.values.ravel(), plot2.values.ravel()])
# norm = mcolors.LogNorm(vmin=1, vmax=all_vals_plus1.max())
#
# # ========================================
# # 3. 创建子图布局
# # ========================================
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
#
# # ========================================
# # 4. 绘图（方法1）
# # ========================================
# sns.heatmap(
#     data=plot1,
#     ax=ax1,
#     annot=cm_method1,   # 标注原始值（不+1）
#     fmt='d',
#     cmap='Blues',
#     xticklabels=CLASS_LABELS,
#     yticklabels=CLASS_LABELS,
#     # cbar_kws={'label': 'Sample Count (LogNorm, plotted as count+1)', 'shrink': 0.85},
#     norm=norm
# )
# ax1.set_xlabel('Predicted Class', fontsize=LABEL_FONT_SIZE)
# ax1.set_ylabel('True Class', fontsize=LABEL_FONT_SIZE)
# ax1.tick_params(axis='x', rotation=45, labelsize=14)
# ax1.tick_params(axis='y', labelsize=14)
#
# # ========================================
# # 5. 绘图（方法2）
# # ========================================
# sns.heatmap(
#     data=plot2,
#     ax=ax2,
#     annot=cm_method2,
#     fmt='d',
#     cmap='Blues',
#     xticklabels=CLASS_LABELS,
#     yticklabels=CLASS_LABELS,
#     # cbar_kws={'label': 'Sample Count (LogNorm, plotted as count+1)', 'shrink': 0.85},
#     norm=norm
# )
# ax2.set_xlabel('Predicted Class', fontsize=LABEL_FONT_SIZE)
# ax2.set_ylabel('True Class', fontsize=LABEL_FONT_SIZE)
# ax2.tick_params(axis='x', rotation=45, labelsize=14)
# ax2.tick_params(axis='y', labelsize=14)
#
# plt.tight_layout()
# plt.savefig('Comparative_Methods_MAG&LSTM', dpi=600)
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================
INPUT_FILES = {
    "(a) Proposed method": Path(
        "evaluation_results_Proposed_method.csv"
    ),
    "(b) ACBCNN": Path(
        "evaluation_results_ACBCNN.csv"
    ),
    "(c) MAG": Path(
        "evaluation_results_MAG.csv"
    ),
    "(d) QNN-BiLSTM": Path(
        "evaluation_results_LSTM.csv"
    ),
    "(e) CIGNN": Path(
        "evaluation_results_GIGNN.csv"
    ),
    "(f) GCT-GNN": Path(
        "evaluation_results_GCTGNN.csv"
    ),
}

OUTPUT_DIR = Path(".")
OUTPUT_FILENAME = (
    "TAES-Fig.11-confusion-matrices-six-methods.png"
)
OUTPUT_PNG = OUTPUT_DIR / OUTPUT_FILENAME


# ============================================================
# 2. Global drawing settings
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["axes.unicode_minus"] = False


FONT_SIZE = 12
TICK_SIZE = 11
ANNOT_SIZE = 10.0

COLORBAR_TICK_SIZE = 10.5
COLORBAR_LABEL_SIZE = 12.5

CLASS_COUNT = 16
CLASS_LABELS = [
    f"Fault {index}"
    for index in range(CLASS_COUNT)
]

CMAP = "Blues"
DPI = 600


# ============================================================
# 3. Figure and layout parameters
# ============================================================
# 将画布宽度收窄，减少每个方形热力图在网格内的横向空白。
# 插入 Word 后可以整体等比例拉大。
FIG_SIZE = (12.0, 8.4)

LEFT = 0.048
RIGHT = 0.932
TOP = 0.985
BOTTOM = 0.105

# 三列之间尽量紧凑
WSPACE = 0.004

# 保持上下两排有足够空间容纳上排横轴刻度和子图题
HSPACE = 0.20

# 色柱与第三列图体之间的距离
COLORBAR_GAP = 0.0025

# 色柱宽度
COLORBAR_WIDTH = 0.013

# 保存图片时的外围留白
SAVE_PAD = 0.015


# ============================================================
# 4. Read confusion matrices
# ============================================================
def load_confusion_matrix(
    file_path: Path
) -> pd.DataFrame:
    """Read a confusion matrix without a header row."""
    return (
        pd.read_csv(
            file_path,
            header=None
        )
        .fillna(0)
        .astype(int)
    )


matrices = {
    method_name: load_confusion_matrix(file_path)
    for method_name, file_path in INPUT_FILES.items()
}


# ============================================================
# 5. Shared logarithmic color mapping
# ============================================================
all_plot_values = []

for matrix in matrices.values():
    # Add one only for color mapping to avoid log(0).
    all_plot_values.append(
        (matrix + 1).values.ravel()
    )

all_plot_values = np.concatenate(
    all_plot_values
)

shared_norm = mcolors.LogNorm(
    vmin=1,
    vmax=all_plot_values.max()
)


# ============================================================
# 6. Create figure and 2 × 3 layout
# ============================================================
fig = plt.figure(
    figsize=FIG_SIZE,
    constrained_layout=False
)

# 这里只保留三列。
# 不再为色柱预留第四列，避免产生额外空白。
grid = fig.add_gridspec(
    nrows=2,
    ncols=3,
    wspace=WSPACE,
    hspace=HSPACE
)

fig.subplots_adjust(
    left=LEFT,
    right=RIGHT,
    top=TOP,
    bottom=BOTTOM
)


axes = []
method_names = list(matrices.keys())


# ============================================================
# 7. Draw six confusion matrices
# ============================================================
for index, method_name in enumerate(method_names):
    row = index // 3
    column = index % 3

    ax = fig.add_subplot(
        grid[row, column]
    )
    axes.append(ax)

    matrix = matrices[method_name]

    # Only used for color mapping.
    plot_matrix = matrix + 1

    sns.heatmap(
        data=plot_matrix,
        ax=ax,

        # Show original integer counts.
        annot=matrix,
        fmt="d",

        cmap=CMAP,
        norm=shared_norm,
        cbar=False,

        linewidths=0.35,
        linecolor="#d9d9d9",

        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,

        annot_kws={
            "fontsize": ANNOT_SIZE
        },

        square=True
    )

    # --------------------------------------------------------
    # Align square heatmaps toward one another.
    #
    # Since each heatmap is square, it may not fill the complete
    # GridSpec cell. Anchoring reduces the visible gaps.
    # --------------------------------------------------------
    if column == 0:
        ax.set_anchor("E")
    elif column == 1:
        ax.set_anchor("C")
    else:
        ax.set_anchor("W")

    # --------------------------------------------------------
    # Tick labels
    # --------------------------------------------------------
    ax.tick_params(
        axis="x",
        rotation=50,
        labelsize=TICK_SIZE,
        pad=0.3,
        length=0
    )

    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=TICK_SIZE,
        pad=0.9,
        length=0
    )

    # --------------------------------------------------------
    # Y-axis labels
    # --------------------------------------------------------
    if column == 0:
        ax.set_ylabel(
            "True Class",
            fontsize=FONT_SIZE,
            labelpad=3
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    # --------------------------------------------------------
    # X-axis labels
    # --------------------------------------------------------
    if row == 1:
        ax.set_xlabel(
            "Predicted Class",
            fontsize=FONT_SIZE,
            labelpad=0
        )
    else:
        ax.set_xlabel("")

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#666666")


# ============================================================
# 8. Shared colorbar positioned relative to the third column
# ============================================================
scalar_mappable = plt.cm.ScalarMappable(
    cmap=CMAP,
    norm=shared_norm
)
scalar_mappable.set_array([])


# Obtain actual positions after square-aspect adjustment.
top_right_position = axes[2].get_position()
bottom_right_position = axes[5].get_position()

colorbar_left = (
    max(
        top_right_position.x1,
        bottom_right_position.x1
    )
    + COLORBAR_GAP
)

colorbar_bottom = (
    bottom_right_position.y0
)

colorbar_top = (
    top_right_position.y1
)

colorbar_height = (
    colorbar_top
    - colorbar_bottom
)

colorbar_axis = fig.add_axes([
    colorbar_left,
    colorbar_bottom,
    COLORBAR_WIDTH,
    colorbar_height
])

colorbar = fig.colorbar(
    scalar_mappable,
    cax=colorbar_axis
)

colorbar.set_label(
    "Sample Count",
    fontsize=COLORBAR_LABEL_SIZE,
    rotation=90,
    labelpad=0
)

colorbar.ax.tick_params(
    labelsize=COLORBAR_TICK_SIZE,
    length=3
)

colorbar.outline.set_linewidth(0.8)


# ============================================================
# 9. Subfigure captions
# ============================================================
caption_map = list(
    INPUT_FILES.keys()
)


# ------------------------------------------------------------
# Top-row captions
# ------------------------------------------------------------
for index in range(3):
    position = axes[index].get_position()

    fig.text(
        (
            position.x0
            + position.x1
        ) / 2,

        position.y0 - 0.058,

        caption_map[index],

        ha="center",
        va="top",

        fontsize=FONT_SIZE,
        fontweight="bold"
    )


# ------------------------------------------------------------
# Bottom-row captions
# ------------------------------------------------------------
for index in range(3, 6):
    position = axes[index].get_position()

    fig.text(
        (
            position.x0
            + position.x1
        ) / 2,

        position.y0 - 0.085,

        caption_map[index],

        ha="center",
        va="top",

        fontsize=FONT_SIZE,
        fontweight="bold"
    )


# ============================================================
# 10. Save figure
# ============================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    OUTPUT_PNG,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=SAVE_PAD,
    facecolor="white"
)

plt.show()