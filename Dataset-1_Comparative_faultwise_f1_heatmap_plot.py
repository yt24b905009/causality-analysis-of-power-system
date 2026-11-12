# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import ListedColormap
# from typing import Dict
#
#
# # =============================
# # 1. Config
# # =============================
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 14
#
# CLASS_COUNT = 16
# FAULT_LABELS = [f"Fault {i}" for i in range(CLASS_COUNT)]
#
# # Display name -> confusion matrix file
# # If you want to change compared methods, edit this dict only.
# METHOD_FILES = {
#     "ACBCNN": "evaluation_results_ACBCNN.csv",
#     "QNN-BiLSTM": "evaluation_results_LSTM.csv",
#     "MAG": "evaluation_results_MAG.csv",
#     "CIGNN": "evaluation_results_GIGNN.csv",
#     "GCT-GNN": "evaluation_results_GCTGNN.csv",
#     "Proposed": "evaluation_results_Proposed_method.csv",
# }
#
# FIG_SIZE = (12, 6)
# SAVE_PATH = "Fig_faultwise_f1_heatmap_dataset1_palette5.png"
# ANNOT_DECIMALS = 3
#
# # Balanced discrete bins from quantiles (equal-frequency style).
# BIN_COUNT = 5
# BIN_COLORS = [
#     "#77295D",  # R119 G041 B093
#     "#C34FA2",  # R195 G079 B162
#     "#EDF1FD",  # R237 G241 B253
#     "#77B7F0",  # R119 G183 B240
#     "#5364C0",  # R083 G100 B192
# ]
#
#
# def load_confusion_matrix(file_path: str, class_count: int = 16) -> np.ndarray:
#     cm_df = pd.read_csv(file_path, header=None).fillna(0)
#     cm = cm_df.to_numpy(dtype=np.float64)
#
#     if cm.shape != (class_count, class_count):
#         raise ValueError(
#             f"{file_path} shape is {cm.shape}, expected ({class_count}, {class_count})."
#         )
#     return cm
#
#
# def calculate_faultwise_f1(cm: np.ndarray) -> np.ndarray:
#     tp = np.diag(cm)
#     fp = cm.sum(axis=0) - tp
#     fn = cm.sum(axis=1) - tp
#     denom = 2 * tp + fp + fn
#     f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
#     return f1
#
#
# def build_f1_matrix(method_files: Dict[str, str]) -> pd.DataFrame:
#     rows = []
#     method_names = []
#     for method_name, file_path in method_files.items():
#         cm = load_confusion_matrix(file_path, CLASS_COUNT)
#         f1 = calculate_faultwise_f1(cm)
#         rows.append(f1)
#         method_names.append(method_name)
#
#     f1_df = pd.DataFrame(rows, index=method_names, columns=FAULT_LABELS)
#     return f1_df
#
#
# def summarize_distribution(values: np.ndarray) -> None:
#     q_list = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
#     quantiles = np.quantile(values, q_list)
#     print("\nF1 distribution summary:")
#     for q, v in zip(q_list, quantiles):
#         print(f"Q{int(q * 100):>3}: {v:.6f}")
#     print(f"Mean: {np.mean(values):.6f}")
#     print(f"Std : {np.std(values):.6f}")
#
#
# def build_quantile_boundaries(values: np.ndarray, bin_count: int) -> np.ndarray:
#     probs = np.linspace(0.0, 1.0, bin_count + 1)
#     boundaries = np.quantile(values, probs)
#
#     # Keep boundaries strictly increasing for BoundaryNorm.
#     eps = 1e-6
#     for i in range(1, len(boundaries)):
#         if boundaries[i] <= boundaries[i - 1]:
#             boundaries[i] = boundaries[i - 1] + eps
#
#     # Clamp to [0, 1] and round display precision only for readability.
#     boundaries[0] = max(0.0, boundaries[0])
#     boundaries[-1] = min(1.0, boundaries[-1])
#     return boundaries
#
#
# def print_bin_counts(values: np.ndarray, boundaries: np.ndarray) -> None:
#     counts, _ = np.histogram(values, bins=boundaries)
#     print("\nQuantile bin boundaries and counts:")
#     for i, c in enumerate(counts):
#         left = boundaries[i]
#         right = boundaries[i + 1]
#         print(f"Bin {i + 1}: [{left:.6f}, {right:.6f}] -> {c} values")
#
#
# def plot_f1_heatmap(f1_df: pd.DataFrame) -> None:
#     values = f1_df.to_numpy().ravel()
#     summarize_distribution(values)
#     print("\nBIN_COLORS:", BIN_COLORS)
#
#     boundaries = build_quantile_boundaries(values, BIN_COUNT)
#     print_bin_counts(values, boundaries)
#
#     # Convert continuous F1 values to discrete bin indices [0, BIN_COUNT-1].
#     # This avoids BoundaryNorm inverse() cursor errors in some matplotlib versions.
#     bin_idx = np.digitize(f1_df.to_numpy(), boundaries[1:-1], right=True)
#     bin_idx_df = pd.DataFrame(bin_idx, index=f1_df.index, columns=f1_df.columns)
#
#     cmap = ListedColormap(BIN_COLORS[:BIN_COUNT], name="purple_red_blue_discrete")
#
#     plt.figure(figsize=FIG_SIZE)
#     ax = sns.heatmap(
#         bin_idx_df,
#         annot=f1_df,
#         fmt=f".{ANNOT_DECIMALS}f",
#         cmap=cmap,
#         vmin=-0.5,
#         vmax=BIN_COUNT - 0.5,
#         linewidths=0.4,
#         linecolor="white",
#         cbar_kws={
#             "label": "F1 Score",
#             "shrink": 0.9,
#             "ticks": np.arange(BIN_COUNT),
#             "pad": 0.01,
#         },
#     )
#     cbar = ax.collections[0].colorbar
#     tick_labels = [
#         f"[{boundaries[i]:.3f}, {boundaries[i + 1]:.3f}]"
#         for i in range(BIN_COUNT)
#     ]
#     cbar.set_ticklabels(tick_labels)
#     ax.set_xlabel("Fault modes")
#     ax.set_ylabel("Methods", labelpad=0)
#     ax.tick_params(axis="x", rotation=35)
#     plt.setp(ax.get_xticklabels(), ha="right")
#     ax.tick_params(axis="y", rotation=0)
#     plt.tight_layout()
#     plt.savefig(SAVE_PATH, dpi=600)
#     plt.show()
#
#
# if __name__ == "__main__":
#     f1_matrix = build_f1_matrix(METHOD_FILES)
#     print("Per-fault F1 matrix:")
#     print(f1_matrix.round(3))
#     plot_f1_heatmap(f1_matrix)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path


# ============================================================
# 1. Global settings
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 17
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. Data settings
# ============================================================
CLASS_COUNT = 16

FAULT_LABELS = [
    "Fault {0}".format(index)
    for index in range(CLASS_COUNT)
]

METHOD_FILES = {
    "ACBCNN": "evaluation_results_ACBCNN.csv",
    "QNN-BiLSTM": "evaluation_results_LSTM.csv",
    "MAG": "evaluation_results_MAG.csv",
    "CIGNN": "evaluation_results_GIGNN.csv",
    "GCT-GNN": "evaluation_results_GCTGNN.csv",
    "Proposed": "evaluation_results_Proposed_method.csv",
}


# ============================================================
# 3. Figure settings
# ============================================================
FIG_SIZE = (11.0, 5.8)

FONT_SIZE = 15
ANNOT_FONT_SIZE = 15
COLORBAR_FONT_SIZE = 15

ANNOT_DECIMALS = 3
DPI = 800

OUTPUT_DIR = Path(".")
OUTPUT_FILENAME = "TAES-Fig.12-faultwise-F1-heatmap.png"
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME


# ============================================================
# 4. Manual layout settings
# ============================================================
# 主热力图区域：[左边界, 下边界, 宽度, 高度]
MAIN_AX_POSITION = [
    0.135,
    0.245,
    0.690,
    0.700,
]

# 色柱区域单独设置。
# 这样色柱刻度区间不会进入热力图。
COLORBAR_AX_POSITION = [
    0.835,
    0.285,
    0.012,
    0.620,
]


# ============================================================
# 5. Discrete color settings
# ============================================================
BIN_COUNT = 5

BIN_COLORS = [
    "#77295D",
    "#C34FA2",
    "#EDF1FD",
    "#77B7F0",
    "#5364C0",
]


# ============================================================
# 6. Load confusion matrix
# ============================================================
def load_confusion_matrix(
    file_path,
    class_count=16
):
    cm_dataframe = (
        pd.read_csv(
            file_path,
            header=None
        )
        .fillna(0)
    )

    confusion_matrix = cm_dataframe.to_numpy(
        dtype=np.float64
    )

    expected_shape = (
        class_count,
        class_count
    )

    if confusion_matrix.shape != expected_shape:
        raise ValueError(
            "{0} shape is {1}, expected {2}.".format(
                file_path,
                confusion_matrix.shape,
                expected_shape
            )
        )

    return confusion_matrix


# ============================================================
# 7. Calculate per-fault F1 score
# ============================================================
def calculate_faultwise_f1(
    confusion_matrix
):
    true_positive = np.diag(
        confusion_matrix
    )

    false_positive = (
        confusion_matrix.sum(axis=0)
        - true_positive
    )

    false_negative = (
        confusion_matrix.sum(axis=1)
        - true_positive
    )

    denominator = (
        2.0 * true_positive
        + false_positive
        + false_negative
    )

    f1_score = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(
            true_positive,
            dtype=float
        ),
        where=denominator > 0
    )

    return f1_score


# ============================================================
# 8. Build F1-score matrix
# ============================================================
def build_f1_matrix(
    method_files
):
    result_rows = []
    method_names = []

    for method_name, file_path in method_files.items():
        confusion_matrix = load_confusion_matrix(
            file_path,
            CLASS_COUNT
        )

        faultwise_f1 = calculate_faultwise_f1(
            confusion_matrix
        )

        result_rows.append(
            faultwise_f1
        )

        method_names.append(
            method_name
        )

    f1_dataframe = pd.DataFrame(
        result_rows,
        index=method_names,
        columns=FAULT_LABELS
    )

    return f1_dataframe


# ============================================================
# 9. Build quantile boundaries
# ============================================================
def build_quantile_boundaries(
    values,
    bin_count
):
    probabilities = np.linspace(
        0.0,
        1.0,
        bin_count + 1
    )

    boundaries = np.quantile(
        values,
        probabilities
    )

    epsilon = 1e-6

    for index in range(
        1,
        len(boundaries)
    ):
        if boundaries[index] <= boundaries[index - 1]:
            boundaries[index] = (
                boundaries[index - 1]
                + epsilon
            )

    boundaries[0] = max(
        0.0,
        boundaries[0]
    )

    boundaries[-1] = min(
        1.0,
        boundaries[-1]
    )

    return boundaries


# ============================================================
# 10. Print distribution information
# ============================================================
def summarize_distribution(
    values
):
    quantile_list = [
        0.00,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        1.00,
    ]

    quantiles = np.quantile(
        values,
        quantile_list
    )

    print("\nF1 distribution summary:")

    for quantile, value in zip(
        quantile_list,
        quantiles
    ):
        print(
            "Q{0:>3}: {1:.6f}".format(
                int(quantile * 100),
                value
            )
        )

    print(
        "Mean: {0:.6f}".format(
            np.mean(values)
        )
    )

    print(
        "Std : {0:.6f}".format(
            np.std(values)
        )
    )


def print_bin_counts(
    values,
    boundaries
):
    counts, _ = np.histogram(
        values,
        bins=boundaries
    )

    print(
        "\nQuantile bin boundaries and counts:"
    )

    for index, count in enumerate(counts):
        print(
            "Bin {0}: [{1:.6f}, {2:.6f}] -> {3} values".format(
                index + 1,
                boundaries[index],
                boundaries[index + 1],
                count
            )
        )


# ============================================================
# 11. Plot heatmap
# ============================================================
def plot_f1_heatmap(
    f1_dataframe
):
    values = (
        f1_dataframe
        .to_numpy()
        .ravel()
    )

    summarize_distribution(
        values
    )

    boundaries = build_quantile_boundaries(
        values,
        BIN_COUNT
    )

    print_bin_counts(
        values,
        boundaries
    )

    # Continuous F1 scores -> discrete color-bin indices
    bin_indices = np.digitize(
        f1_dataframe.to_numpy(),
        boundaries[1:-1],
        right=True
    )

    bin_index_dataframe = pd.DataFrame(
        bin_indices,
        index=f1_dataframe.index,
        columns=f1_dataframe.columns
    )

    color_map = ListedColormap(
        BIN_COLORS[:BIN_COUNT],
        name="purple_red_blue_discrete"
    )

    # --------------------------------------------------------
    # Create figure and independent axes
    # --------------------------------------------------------
    fig = plt.figure(
        figsize=FIG_SIZE,
        constrained_layout=False
    )

    ax = fig.add_axes(
        MAIN_AX_POSITION
    )

    colorbar_ax = fig.add_axes(
        COLORBAR_AX_POSITION
    )

    # --------------------------------------------------------
    # Draw heatmap
    # --------------------------------------------------------
    heatmap = sns.heatmap(
        bin_index_dataframe,
        ax=ax,

        annot=f1_dataframe,
        fmt=".{0}f".format(
            ANNOT_DECIMALS
        ),

        cmap=color_map,
        vmin=-0.5,
        vmax=BIN_COUNT - 0.5,

        linewidths=0.55,
        linecolor="white",

        annot_kws={
            "fontsize": ANNOT_FONT_SIZE
        },

        cbar=True,
        cbar_ax=colorbar_ax,

        cbar_kws={
            "ticks": np.arange(
                BIN_COUNT
            )
        }
    )

    # --------------------------------------------------------
    # Configure colorbar
    # --------------------------------------------------------
    colorbar = heatmap.collections[0].colorbar

    # --------------------------------------------------------
    # Configure colorbar: use boundary values instead of intervals
    # --------------------------------------------------------
    colorbar = heatmap.collections[0].colorbar

    # 5个色块 -> 6个边界位置
    boundary_positions = np.arange(-0.5, BIN_COUNT + 0.5, 1.0)

    # 边界数值标签，例如：0.815, 0.934, ...
    boundary_labels = [
        "{0:.3f}".format(value)
        for value in boundaries
    ]

    # 把刻度放在每个色块的上下边界
    colorbar.set_ticks(boundary_positions)
    colorbar.set_ticklabels(boundary_labels)

    # 刻度仍放在右侧
    colorbar.ax.yaxis.set_ticks_position("right")

    colorbar.ax.tick_params(
        labelleft=False,
        labelright=True,
        labelsize=COLORBAR_FONT_SIZE,
        length=2,
        pad=1
    )

    # 给色柱数值标签加一点旋转
    plt.setp(
        colorbar.ax.get_yticklabels(),
        rotation=90,  # 你也可以改成 45 试试
        va="center",
        ha="center"
    )

    colorbar.outline.set_linewidth(0.8)


    plt.setp(
        colorbar.ax.get_yticklabels(),
        rotation=45,
        ha="left",
        va="center",
        rotation_mode="anchor"
    )
    colorbar.outline.set_linewidth(
        0.8
    )

    # 将 F1 Score 水平放置在色柱正下方
    colorbar.ax.text(
        2.5,
        -0.05,
        "F1 Score",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=COLORBAR_FONT_SIZE
    )

    # --------------------------------------------------------
    # Configure heatmap axes
    # --------------------------------------------------------
    ax.set_xlabel(
        "Fault modes",
        fontsize=FONT_SIZE,
        labelpad=6
    )

    ax.tick_params(
        axis="x",
        labelsize=FONT_SIZE,
        rotation=35,
        pad=1,
        length=3
    )

    plt.setp(
        ax.get_xticklabels(),
        ha="right",
        rotation_mode="anchor"
    )

    ax.tick_params(
        axis="y",
        labelsize=FONT_SIZE,
        pad=0,
        length=0
    )

    plt.setp(
        ax.get_yticklabels(),
        rotation=45,
        ha="right",
        va="center",
        rotation_mode="anchor"
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(
            "#666666"
        )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white"
    )

    plt.show()

    print(
        "Saved figure to: {0}".format(
            OUTPUT_PATH.resolve()
        )
    )


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    f1_matrix = build_f1_matrix(
        METHOD_FILES
    )

    print(
        "Per-fault F1 matrix:"
    )

    print(
        f1_matrix.round(
            ANNOT_DECIMALS
        )
    )

    plot_f1_heatmap(
        f1_matrix
    )