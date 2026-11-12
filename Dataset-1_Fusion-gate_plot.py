# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Set matplotlib to support English
# plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
# plt.rcParams['font.size'] = 14  # 可选：设置字体大小
#
# # Load data
# df = pd.read_csv('fusion_gates_labels.csv')
#
# # Get gate weight columns
# gate_columns = [col for col in df.columns if col.startswith('gate_')]
#
# # Calculate fusion weights for each fault class
# num_classes = 16
# class_weights = {i: {'scm': [], 'gc': []} for i in range(num_classes)}
#
# for idx, row in df.iterrows():
#     label = int(row['true_label'])
#     gate_values = row[gate_columns].values
#
#     # Calculate structural causal weight (mean of gate weights)
#     scm_weight = np.mean(gate_values)
#     # Calculate Granger causal weight (1 - structural causal weight)
#     gc_weight = 1 - scm_weight
#
#     class_weights[label]['scm'].append(scm_weight)
#     class_weights[label]['gc'].append(gc_weight)
#
# # Calculate average fusion weights for each class
# avg_weights = []
# for i in range(num_classes):
#     if class_weights[i]['scm']:
#         avg_scm = np.mean(class_weights[i]['scm'])
#         avg_gc = np.mean(class_weights[i]['gc'])
#         avg_weights.append((i, avg_scm, avg_gc))
#
# # Plot fusion weight distribution
# class_ids = [x[0] for x in avg_weights]
# scm_weights = [x[1] for x in avg_weights]
# gc_weights = [x[2] for x in avg_weights]
#
# plt.figure(figsize=(10, 5))
# x = np.arange(len(class_ids))
# width = 0.4  # 增加柱体宽度
#
# # 创建柱状图
# bars1 = plt.bar(x - width / 2, scm_weights, width, label='Spatial Causal Weight', color='#87CEEB')
# bars2 = plt.bar(x + width / 2, gc_weights, width, label='Granger Causal Weight', color='#FFB6C1')
#
# # 为占优势的权重添加斜线标记
# for i, (scm, gc) in enumerate(zip(scm_weights, gc_weights)):
#     if scm > gc:  # 结构因果权重占优势
#         bars1[i].set_hatch('///')
#     elif gc > scm:  # 格兰杰因果权重占优势
#         bars2[i].set_hatch('///')
#
# # 创建自定义图例，显示斜线标记的含义
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='#87CEEB', label='Spatial Causal Weight (SCW)'),
#     Patch(facecolor='#FFB6C1', label='Granger Causal Weight (GCW)'),
#     Patch(facecolor='#87CEEB', hatch='///', label='Dominant SCW'),
#     Patch(facecolor='#FFB6C1', hatch='///', label='Dominant GCW')
# ]
#
# plt.xlabel('Fault Category', fontsize=14)
# plt.ylabel('Fusion Weight', fontsize=14)
# plt.xticks(x, [f'Fault {i}' for i in class_ids], fontsize=14, rotation=45, ha='right')
# plt.legend(handles=legend_elements, fontsize=14, ncol=2, loc='lower left')
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.savefig('Fig.10-fusion_weight_distribution.png', dpi=1080, bbox_inches='tight')
#
# # 建议同时保存一份 PDF 格式（矢量图），无论放大多少倍都绝对清晰
# plt.savefig('fusion_weight_distribution.pdf', bbox_inches='tight')
# plt.show()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ============================================================
# Unified plotting settings
# ============================================================
FONT_SIZE = 16

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["hatch.linewidth"] = 0.8
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# Input and output paths
# ============================================================
INPUT_CSV = Path("fusion_gates_labels.csv")

# Keep the output path explicitly configurable.
OUTPUT_DIR = Path(r".")
OUTPUT_FILENAME = "TAES-Fig.10-fusion_weight_distribution.png"
OUTPUT_PNG = OUTPUT_DIR / OUTPUT_FILENAME

# ============================================================
# Plotting constants
# ============================================================
NUM_CLASSES = 16
BAR_WIDTH = 0.36
SCW_COLOR = "#87CEEB"
GCW_COLOR = "#FFB6C1"
EDGE_COLOR = "#4f4f4f"
HATCH_PATTERN = "////"


def load_gate_weights(csv_path: Path) -> pd.DataFrame:
    """Load the gate-weight CSV and validate the required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path.resolve()}")

    dataframe = pd.read_csv(csv_path)

    if "true_label" not in dataframe.columns:
        raise ValueError("The CSV must contain a 'true_label' column.")

    gate_columns = [
        column
        for column in dataframe.columns
        if column.startswith("gate_")
    ]

    if not gate_columns:
        raise ValueError(
            "No gate-weight columns were found. "
            "At least one column beginning with 'gate_' is required."
        )

    return dataframe


def calculate_class_weights(
    dataframe: pd.DataFrame,
    num_classes: int = NUM_CLASSES,
):
    """Calculate the mean spatial and Granger causal weights by class."""
    gate_columns = [
        column
        for column in dataframe.columns
        if column.startswith("gate_")
    ]

    class_weights = {
        class_id: {"scw": [], "gcw": []}
        for class_id in range(num_classes)
    }

    for _, row in dataframe.iterrows():
        class_id = int(row["true_label"])

        if class_id not in class_weights:
            continue

        gate_values = pd.to_numeric(
            row[gate_columns],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid_values = gate_values[np.isfinite(gate_values)]

        if valid_values.size == 0:
            continue

        scw = float(np.mean(valid_values))
        gcw = 1.0 - scw

        class_weights[class_id]["scw"].append(scw)
        class_weights[class_id]["gcw"].append(gcw)

    class_ids = []
    scw_values = []
    gcw_values = []

    for class_id in range(num_classes):
        if not class_weights[class_id]["scw"]:
            continue

        class_ids.append(class_id)
        scw_values.append(
            float(np.mean(class_weights[class_id]["scw"]))
        )
        gcw_values.append(
            float(np.mean(class_weights[class_id]["gcw"]))
        )

    if not class_ids:
        raise ValueError("No valid class-level gate weights could be calculated.")

    return class_ids, scw_values, gcw_values


def plot_fusion_weights(
    class_ids,
    scw_values,
    gcw_values,
    output_path: Path,
):
    """
    Plot Fig. 10 and save a high-resolution PNG.

    The legend is placed inside the lower part of the plotting area,
    directly below the main portions of the bars.
    """
    # This size is intended for insertion into one column of a two-column paper.
    # Saving at 1000 dpi preserves clarity after Word scales the image down.
    fig, ax = plt.subplots(
        figsize=(7.0, 4.8),
        constrained_layout=False,
    )

    x = np.arange(len(class_ids))

    bars_scw = ax.bar(
        x - BAR_WIDTH / 2,
        scw_values,
        width=BAR_WIDTH,
        color=SCW_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
        label="Spatial Causal Weight (SCW)",
        zorder=3,
    )

    bars_gcw = ax.bar(
        x + BAR_WIDTH / 2,
        gcw_values,
        width=BAR_WIDTH,
        color=GCW_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
        label="Granger Causal Weight (GCW)",
        zorder=3,
    )

    # Add hatching only to the dominant weight for each fault class.
    for index, (scw, gcw) in enumerate(zip(scw_values, gcw_values)):
        if scw > gcw:
            bars_scw[index].set_hatch(HATCH_PATTERN)
        elif gcw > scw:
            bars_gcw[index].set_hatch(HATCH_PATTERN)

    legend_handles = [
        Patch(
            facecolor=SCW_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            label="Spatial Causal Weight (SCW)",
        ),
        Patch(
            facecolor=GCW_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            label="Granger Causal Weight (GCW)",
        ),
        Patch(
            facecolor=SCW_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            hatch=HATCH_PATTERN,
            label="Dominant SCW",
        ),
        Patch(
            facecolor=GCW_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            hatch=HATCH_PATTERN,
            label="Dominant GCW",
        ),
    ]

    ax.set_xlabel(
        "Fault Category",
        fontsize=FONT_SIZE,
        labelpad=8,
    )
    ax.set_ylabel(
        "Fusion Weight",
        fontsize=FONT_SIZE,
        labelpad=7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Fault {class_id}" for class_id in class_ids],
        fontsize=FONT_SIZE,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_ylim(0.0, 0.67)
    ax.set_yticks(np.arange(0.0, 0.61, 0.2))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_SIZE,
        width=1.0,
        length=4,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        zorder=0,
    )

    # Place the legend inside the plotting area at the bottom-left,
    # i.e. in the region below the main bar heights.
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.025),
        ncol=2,
        fontsize=FONT_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        borderpad=0.25,
        columnspacing=0.65,
        handlelength=1.15,
        handleheight=0.85,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    # Manual margins are more stable than tight_layout for rotated labels.
    fig.subplots_adjust(
        left=0.13,
        right=0.995,
        top=0.97,
        bottom=0.27,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=1000,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
    )

    plt.show()
    plt.close(fig)


def main():
    dataframe = load_gate_weights(INPUT_CSV)

    class_ids, scw_values, gcw_values = calculate_class_weights(
        dataframe,
        num_classes=NUM_CLASSES,
    )

    plot_fusion_weights(
        class_ids,
        scw_values,
        gcw_values,
        output_path=OUTPUT_PNG,
    )

    print(f"Saved Fig. 10 to: {OUTPUT_PNG.resolve()}")


if __name__ == "__main__":
    main()