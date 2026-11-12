import pandas as pd
import numpy as np
import os

def load_confusion_matrix(csv_path):
    """加载无表头、无行索引的混淆矩阵CSV文件（16×16）。"""
    try:
        df = pd.read_csv(
            csv_path,
            header=None,      # 无表头
            index_col=None    # 无行索引
        )
        # 验证形状（16行×16列）和数据类型（整数）
        assert df.shape == (16, 16), f"混淆矩阵形状错误（期望16×16，实际{df.shape}）"
        assert df.dtypes.eq('int64').all(), "混淆矩阵必须为整数类型（样本计数）"
        # 设置行列名称（可选，便于理解）
        df.index.name = "True Class"
        df.columns.name = "Predicted Class"
        return df
    except Exception as e:
        raise ValueError(f"混淆矩阵加载失败：{e}")

def calculate_diagnostic_metrics(cm_df):
    """计算类别指标（Precision、Recall、F1-Score）和总体指标（Overall Accuracy、宏平均、加权平均）。"""
    cm = cm_df.values  # 转换为numpy数组（行=真实类别，列=预测类别）
    n_classes = cm.shape[0]
    total_samples = cm.sum()  # 总样本数

    # 初始化类别指标
    class_metrics = {
        "True Class": list(range(n_classes)),  # 真实类别（0~15）
        "Support": cm.sum(axis=1).tolist(),    # 每类总样本数（TP+FN）
        "TP": np.diag(cm).tolist(),            # 真正例（对角线元素）
        "FP": (cm.sum(axis=0) - np.diag(cm)).tolist(),  # 假正例（列和-对角线）
        "FN": (cm.sum(axis=1) - np.diag(cm)).tolist()   # 假反例（行和-对角线）
    }

    # 计算Precision、Recall、F1-Score（避免除以零）
    class_metrics["Precision"] = [
        tp / (tp + fp) if (tp + fp) > 0 else 0
        for tp, fp in zip(class_metrics["TP"], class_metrics["FP"])
    ]
    class_metrics["Recall"] = [
        tp / (tp + fn) if (tp + fn) > 0 else 0
        for tp, fn in zip(class_metrics["TP"], class_metrics["FN"])
    ]
    class_metrics["F1-Score"] = [
        2 * (p * r) / (p + r) if (p + r) > 0 else 0
        for p, r in zip(class_metrics["Precision"], class_metrics["Recall"])
    ]

    # 转换为DataFrame（保留4位小数）
    class_metrics_df = pd.DataFrame(class_metrics).round(4)

    # 计算总体指标
    overall_metrics = {
        "Overall Accuracy": np.diag(cm).sum() / total_samples if total_samples > 0 else 0,
        "Macro Precision": class_metrics_df["Precision"].mean(),
        "Macro Recall": class_metrics_df["Recall"].mean(),
        "Macro F1": class_metrics_df["F1-Score"].mean(),
        "Weighted Precision": np.average(class_metrics_df["Precision"], weights=class_metrics_df["Support"]),
        "Weighted Recall": np.average(class_metrics_df["Recall"], weights=class_metrics_df["Support"]),
        "Weighted F1": np.average(class_metrics_df["F1-Score"], weights=class_metrics_df["Support"])
    }
    # 总体指标保留4位小数
    overall_metrics = {k: round(v, 4) for k, v in overall_metrics.items()}

    return class_metrics_df, overall_metrics

def save_and_print_results(class_metrics, overall_metrics, input_csv):
    """保存结果到CSV（原文件同名加后缀"_metrics"），并打印全部结果。"""
    # 生成输出文件名（避免覆盖原文件）
    output_csv = os.path.splitext(input_csv)[0] + "_metrics.csv"

    # 构造空行分隔符（1行n列，匹配class_metrics的列数）
    empty_row = pd.DataFrame(
        [[pd.NA] * len(class_metrics.columns)],  # 2维列表，生成1行n列
        columns=class_metrics.columns
    )

    # 转换总体指标为DataFrame（添加"Overall"标识）
    overall_df = pd.DataFrame(overall_metrics, index=["Overall"]).round(4)

    # 合并类别指标、空行、总体指标
    result_df = pd.concat([
        class_metrics,
        empty_row,    # 空行分隔
        overall_df
    ], ignore_index=True)  # 忽略索引，避免重复

    # 保存结果到CSV
    result_df.to_csv(output_csv, index=False)

    # 打印结果（优化格式）
    print("\n" + "="*80)
    print(f"类别指标（全部{len(class_metrics)}类）:")
    print("="*80)
    # 打印关键列（调整顺序，更易读）
    print(class_metrics[["True Class", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"]])

    print("\n" + "="*80)
    print("总体指标:")
    print("="*80)
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value}")

    print(f"\n结果已保存到: {output_csv}")


# ====================== 主程序（入口） ======================
if __name__ == "__main__":
    # 1. 配置输入文件路径（替换为你的混淆矩阵文件路径）
    INPUT_CSV = r"evaluation_results_MAG.csv"

    # 2. 加载混淆矩阵
    try:
        cm_df = load_confusion_matrix(INPUT_CSV)
        print(f"混淆矩阵加载成功！（路径：{INPUT_CSV}）")
        print(f"混淆矩阵形状：{cm_df.shape}")
    except ValueError as e:
        print(f"错误：{e}")
        exit()

    # 3. 计算指标
    class_metrics, overall_metrics = calculate_diagnostic_metrics(cm_df)

    # 4. 保存并打印结果
    save_and_print_results(class_metrics, overall_metrics, INPUT_CSV)

