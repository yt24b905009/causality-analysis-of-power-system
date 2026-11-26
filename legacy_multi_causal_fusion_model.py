import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse, add_self_loops, to_undirected
from torch.distributions.normal import Normal

# 导入自定义模块
from legacy_data_preprocessing import PowerSystemDataset, CHANNEL_NAMES, DERIVED_FEATURES, ALL_FEATURE_NAMES
from legacy_structural_causal_model import SimpleCVAE, CVAE_GAT
from legacy_granger_causal_model import compute_granger_causality_matrix, create_fused_adjacency_matrix

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签宋体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
DATA_DIR = r'D:\Dataset_Real\Paper3-Satellite电源系统故障诊断'
NUM_CLASSES = 16  # 16个故障类别
WINDOW_SIZE = 100  # 时间窗口大小
STRIDE = 50  # 窗口滑动步长
PLOT_RESULTS = 1  # 控制是否绘制图表
LATENT_DIM = 32  # 潜在空间维度
EPSILON = 1e-8  # 避免除零的小量


class GatedFusion(nn.Module):
    """门控融合模块，用于融合结构因果图和格兰杰因果图的特征"""

    def __init__(self, feature_dim, hidden_dim=64, dropout=0.2):
        super(GatedFusion, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

        # 特征转换层，保证维度一致
        self.scm_transform = nn.Linear(feature_dim, feature_dim)
        self.granger_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, scm_features, granger_features):
        """
        门控融合两种因果特征
        Args:
            scm_features: 结构因果模型特征
            granger_features: 格兰杰因果模型特征
        """
        # 特征转换
        scm_trans = self.scm_transform(scm_features)
        granger_trans = self.granger_transform(granger_features)

        # 计算融合门控权重
        concat_features = torch.cat([scm_trans, granger_trans], dim=1)
        gate = self.gate_network(concat_features)

        # 门控融合
        fused_features = gate * scm_trans + (1 - gate) * granger_trans

        return fused_features, gate


class MultiCausalFusionModel(nn.Module):
    """多层次因果融合故障诊断模型"""

    def __init__(self, num_node_features, num_classes,
                 hidden_dim=64, heads=4, dropout=0.3, latent_dim=32,
                 edge_index_gc=None):  # 新增格兰杰因果图边索引参数
        super(MultiCausalFusionModel, self).__init__()

        # 存储格兰杰因果边索引
        self.edge_index_gc = edge_index_gc

        # 1. 结构因果图模型部分 - 基于CVAE重建特征
        self.cvae = SimpleCVAE(
            input_dim=num_node_features,
            latent_dim=latent_dim,
            num_classes=num_classes
        )

        # 结构因果图的GAT层
        self.scm_gat1 = GATConv(
            num_node_features,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.scm_gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # 2. 格兰杰因果图模型部分
        # 时序特征提取器
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 计算CNN输出维度
        self.cnn_output_dim = 128

        # 格兰杰因果图的GAT层
        self.gc_gat1 = GATConv(
            self.cnn_output_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.gc_gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # 3. 门控融合模块
        self.gated_fusion = GatedFusion(feature_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

        # 4. 融合后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        num_nodes = x.size(0)
        device = x.device

        # 保存原始输入特征
        original_x = x.clone()

        # 第一部分：结构因果图模型处理
        # 使用CVAE重建特征
        x_recon, mu, logvar = self.cvae(x, y)

        # 添加自环和转换为无向图
        edge_index_scm, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index_scm = to_undirected(edge_index_scm)

        # 结构因果图的GAT层
        x_scm = F.elu(self.scm_gat1(x_recon, edge_index_scm))
        x_scm = F.dropout(x_scm, p=0.3, training=self.training)
        x_scm = self.scm_gat2(x_scm, edge_index_scm)

        # 全局池化得到图级别特征
        x_scm_pooled = global_mean_pool(x_scm, batch)

        # 第二部分：格兰杰因果图模型处理
        # 处理时序特征
        x_seq = original_x.unsqueeze(1)  # 添加通道维度 (num_nodes, 1, seq_len)
        x_encoded = self.temporal_encoder(x_seq)
        x_gc = x_encoded.view(num_nodes, self.cnn_output_dim)  # (num_nodes, cnn_output_dim)

        # 使用格兰杰因果图边索引（如果有）
        if self.edge_index_gc is not None:
            # 确保边索引在正确的设备上
            edge_index_gc = self.edge_index_gc.to(device)
            # 限制索引在有效范围内
            edge_index_gc = torch.clamp(edge_index_gc, 0, num_nodes - 1)
        else:
            # 如果没有提供格兰杰因果边索引，使用默认边索引
            edge_index_gc, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            edge_index_gc = to_undirected(edge_index_gc)

        # 格兰杰因果图的GAT层
        x_gc = F.elu(self.gc_gat1(x_gc, edge_index_gc))
        x_gc = F.dropout(x_gc, p=0.3, training=self.training)
        x_gc = self.gc_gat2(x_gc, edge_index_gc)

        # 全局池化得到图级别特征
        x_gc_pooled = global_mean_pool(x_gc, batch)

        # 第三部分：门控融合
        fused_features, fusion_gate = self.gated_fusion(x_scm_pooled, x_gc_pooled)

        # 分类
        logits = self.classifier(fused_features)

        # 计算损失
        recon_loss = F.mse_loss(x_recon, original_x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / num_nodes

        return logits, recon_loss, kl_loss, fusion_gate


def train_fusion_model(model, train_loader, val_loader, device, num_epochs=50):
    """训练多层次因果融合模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'scm_weight': [], 'gc_weight': []}

    # 早停机制
    early_stopping_counter = 0
    patience = 15

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        epoch_scm_weight = 0.0
        epoch_gc_weight = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            outputs, recon_loss, kl_loss, fusion_gate = model(data)
            cls_loss = criterion(outputs, data.y)

            # 总损失 = 分类损失 + 重建损失 + KL损失
            total_loss = cls_loss + 0.5 * recon_loss + 0.1 * kl_loss

            total_loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += total_loss.item() * data.num_graphs
            running_correct += (preds == data.y).sum().item()
            total += data.num_graphs

            # 记录门控权重
            epoch_scm_weight += fusion_gate.mean().item() * data.num_graphs
            epoch_gc_weight += (1 - fusion_gate.mean().item()) * data.num_graphs

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        avg_scm_weight = epoch_scm_weight / total
        avg_gc_weight = epoch_gc_weight / total

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['scm_weight'].append(avg_scm_weight)
        history['gc_weight'].append(avg_gc_weight)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                outputs, recon_loss, kl_loss, _ = model(data)

                cls_loss = criterion(outputs, data.y)
                total_loss = cls_loss + 0.5 * recon_loss + 0.1 * kl_loss

                _, preds = torch.max(outputs, 1)
                val_loss += total_loss.item() * data.num_graphs
                val_correct += (preds == data.y).sum().item()
                val_total += data.num_graphs

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 更新学习率
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_fusion_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"早停于第{epoch + 1}轮")
                break

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"训练损失: {epoch_loss:.4f} | 训练准确率: {epoch_acc:.4f} | "
              f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
              f"SCM权重: {avg_scm_weight:.4f} | GC权重: {avg_gc_weight:.4f}")

    return history


def evaluate_fusion_model(model, data_loader, device, save_csv_path='evaluation_results.csv'):
    """评估多层次因果融合模型性能"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    fusion_gates = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs, recon_loss, kl_loss, fusion_gate = model(data)

            cls_loss = criterion(outputs, data.y)
            total_loss = cls_loss + 0.5 * recon_loss + 0.1 * kl_loss

            _, preds = torch.max(outputs, 1)
            running_loss += total_loss.item() * data.num_graphs
            running_correct += (preds == data.y).sum().item()
            total += data.num_graphs

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            fusion_gates.append(fusion_gate.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = running_correct / total
    fusion_gates = np.vstack(fusion_gates)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 将混淆矩阵保存到CSV文件
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(save_csv_path, index=True, header=True)

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), fusion_gates


def plot_fusion_history(history):
    """绘制多层次因果融合模型训练历史"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 融合权重曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['scm_weight'], label='结构因果权重')
    plt.plot(history['gc_weight'], label='格兰杰因果权重')
    plt.title('因果融合权重变化')
    plt.xlabel('Epoch')
    plt.ylabel('权重')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_fusion_weights(fusion_gates, labels, num_classes=NUM_CLASSES):
    """可视化不同故障类型的融合权重"""
    class_weights = {i: {'scm': [], 'gc': []} for i in range(num_classes)}

    for label, gate in zip(labels, fusion_gates):
        class_weights[label]['scm'].append(gate.mean())
        class_weights[label]['gc'].append(1 - gate.mean())

    # 计算每个类别的平均融合权重
    avg_weights = []
    for i in range(num_classes):
        if class_weights[i]['scm']:  # 确保类别有样本
            avg_scm = np.mean(class_weights[i]['scm'])
            avg_gc = np.mean(class_weights[i]['gc'])
            avg_weights.append((i, avg_scm, avg_gc))

    # 绘制融合权重分布
    class_ids = [x[0] for x in avg_weights]
    scm_weights = [x[1] for x in avg_weights]
    gc_weights = [x[2] for x in avg_weights]

    plt.figure(figsize=(12, 8))
    x = np.arange(len(class_ids))
    width = 0.35

    plt.bar(x - width / 2, scm_weights, width, label='结构因果权重')
    plt.bar(x + width / 2, gc_weights, width, label='格兰杰因果权重')

    plt.xlabel('故障类别')
    plt.ylabel('融合权重')
    plt.title('不同故障类型的因果融合权重分布')
    plt.xticks(x, [f'故障{i}' for i in class_ids])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


# 提取特征序列用于格兰杰因果分析
def extract_feature_sequences(dataset):
    """从数据集中提取特征序列用于格兰杰因果分析"""
    all_sequences = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        features = data.x.numpy()  # (num_features, window_size)
        all_sequences.append(features.T)  # (window_size, num_features)
    return np.array(all_sequences)  # (num_samples, window_size, num_features)

def analyze_results(dataset, all_preds, all_labels, indices):
    """分析测试结果（不保存详细报告）"""
    # 创建结果DataFrame
    results = pd.DataFrame({
        '样本索引': indices,
        '预测标签': all_preds,
        '真实标签': all_labels
    })

    # 添加来源文件信息
    sample_to_file = [dataset.sample_to_file[i] for i in indices]
    results['来源文件'] = sample_to_file

    # 添加故障类别
    results['故障类别'] = results['来源文件'].map(dataset.file_labels)

    # 计算每个故障类别的准确率
    class_acc = results.groupby('故障类别').apply(
        lambda x: (x['预测标签'] == x['真实标签']).mean()
    ).reset_index(name='准确率')

    return results, class_acc

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据集
    print("\n正在创建图数据集...")
    dataset = PowerSystemDataset(DATA_DIR, window_size=WINDOW_SIZE, stride=STRIDE, include_derived_features=False)

    # 验证数据集
    if len(dataset) == 0:
        print("错误：数据集为空！")
        return

    # 检查样本特征
    sample = dataset[0]
    print(f"样本特征形状: {sample.x.shape}")
    print(f"边索引形状: {sample.edge_index.shape}")
    print(f"标签: {sample.y.item()}")

    # 计算格兰杰因果矩阵
    print("\n正在计算格兰杰因果矩阵...")
    feature_sequences = extract_feature_sequences(dataset)
    gc_matrix = compute_granger_causality_matrix(feature_sequences)
    gc_adj_matrix = create_fused_adjacency_matrix(gc_matrix, CHANNEL_NAMES)

    # 将格兰杰因果邻接矩阵转换为边索引格式
    edge_index_gc, _ = dense_to_sparse(torch.tensor(gc_adj_matrix, dtype=torch.float32))
    print(f"格兰杰因果图边索引形状: {edge_index_gc.shape}")

    # 按样本划分训练集和测试集（分层抽样）
    indices = np.arange(len(dataset))
    labels = [graph.y.item() for graph in dataset]  # 提取标签

    # 按样本划分训练集和测试集（分层抽样）
    indices = np.arange(len(dataset))
    labels = [graph.y.item() for graph in dataset]  # 提取标签

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 创建训练集和测试集
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"\n数据集划分统计:")
    print(f"训练集大小: {len(train_dataset)} 个样本")
    print(f"测试集大小: {len(test_dataset)} 个样本")

    from torch_geometric.data import DataLoader as GeoDataLoader
    # 创建数据加载器
    train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = GeoDataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取输入特征维度
    num_node_features = dataset[0].x.shape[1]  # 窗口大小 WINDOW_SIZE

    # 初始化多层次因果融合模型，传入格兰杰因果边索引
    model = MultiCausalFusionModel(
        num_node_features=num_node_features,
        num_classes=NUM_CLASSES,
        hidden_dim=64,
        heads=4,
        dropout=0.3,
        latent_dim=LATENT_DIM,
        edge_index_gc=edge_index_gc  # 传入格兰杰因果边索引
    ).to(device)

    print("\n模型架构:")
    print(model)

    # 训练模型
    print("\n开始训练多层次因果融合模型...")
    # history = train_fusion_model(model, train_loader, test_loader, device, num_epochs=2)

    # 评估最佳模型
    model.load_state_dict(torch.load('best_fusion_model.pth'))
    test_loss, test_acc, all_preds, all_labels, fusion_gates = evaluate_fusion_model(model, test_loader, device,
                                                                save_csv_path='evaluation_results_proposed_method.csv')

    print(f"\n测试准确率: {test_acc:.4f}")

    # 生成分类报告
    class_names = [f"故障类别{i}" for i in range(NUM_CLASSES)]
    report = classification_report(all_labels, all_preds, digits=4, target_names=class_names)

    # 打印分类报告
    print("\n分类报告:")
    print(report)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 可选：绘制图表
    if PLOT_RESULTS:
        print("\n正在绘制图表...")
        # plot_fusion_history(history)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot_kws={"size": 12})
        plt.title('多层次因果融合故障诊断混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.tight_layout()
        plt.show()

        # 分析错误样本
        results, class_acc = analyze_results(dataset, all_preds, all_labels, test_idx)
        print("\n各故障类别准确率:")
        print(class_acc)

        # 可视化融合权重
        visualize_fusion_weights(fusion_gates, all_labels)
    else:
        print("\n跳过图表绘制（PLOT_RESULTS=False）")

    # 将模型文件和代码保存
    print("\n保存模型和结果...")
    torch.save(model.state_dict(), 'final_fusion_model.pth')

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
