import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, dense_to_sparse
import numpy as np
import os
import time


# 避免除零的小量
EPSILON = 1e-8
# 格兰杰因果显著性阈值
GC_THRESHOLD = 0.05


class SharedLSTMGC(nn.Module):
    """共享LSTM模型用于格兰杰因果分析"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SharedLSTMGC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 共享的LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 每个输出特征一个独立的线性层
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            ) for _ in range(output_size)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 每个特征独立预测
        outputs = []
        for i in range(self.output_size):
            outputs.append(self.output_layers[i](last_out))

        return torch.cat(outputs, dim=1)  # (batch_size, output_size)


class PowerSystemGAT(nn.Module):
    """电源系统图注意力网络模型"""

    def __init__(self, num_node_features, num_classes, hidden_dim=128, heads=4, dropout=0.3):
        super(PowerSystemGAT, self).__init__()

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

        # 第一层GAT - 捕获局部关系
        self.conv1 = GATConv(
            self.cnn_output_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        # 第二层GAT - 捕获全局关系
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 获取节点数量
        num_nodes = x.size(0)

        # 处理时序特征
        x_seq = x.unsqueeze(1)  # 添加通道维度 (num_nodes, 1, seq_len)
        x_encoded = self.temporal_encoder(x_seq)
        x = x_encoded.view(num_nodes, self.cnn_output_dim)  # (num_nodes, cnn_output_dim)

        # 确保边索引有效
        # 添加自环和转换为无向图
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = to_undirected(edge_index)

        # 第一层GAT
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)

        # 第二层GAT
        x = self.conv2(x, edge_index)

        # 全局平均池化
        x = global_mean_pool(x, batch)

        # 分类器
        logits = self.classifier(x)
        
        return logits, x


def compute_granger_causality_matrix(features_seq, lstm_hidden_size=128, num_layers=2, batch_size=32, num_epochs=20):
    """
    计算格兰杰因果矩阵
    Args:
        features_seq: 特征序列 (num_samples, seq_len, num_features)
        lstm_hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        batch_size: 批大小
        num_epochs: 训练轮数
    Returns:
        gc_matrix: 格兰杰因果矩阵 (num_features, num_features)
    """
    # 检查是否存在缓存
    cache_path = 'gc_matrix.npy'
    if os.path.exists(cache_path):
        gc_matrix = np.load(cache_path)
        print(f"加载保存的GC矩阵: {gc_matrix.shape}")
        return gc_matrix
        
    print("\n开始计算格兰杰因果矩阵...")
    start_time = time.time()
    
    # 转换为PyTorch张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_seq = torch.tensor(features_seq, dtype=torch.float32)
    num_samples, seq_len, num_features = X_seq.shape
    
    # 划分训练集和验证集
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(0.8 * num_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    X_train = X_seq[train_idx].to(device)
    X_val = X_seq[val_idx].to(device)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train)
    val_dataset = torch.utils.data.TensorDataset(X_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化LSTM模型
    model = SharedLSTMGC(
        input_size=num_features,
        hidden_size=lstm_hidden_size,
        num_layers=num_layers,
        output_size=num_features
    ).to(device)
    
    # 训练模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"训练共享LSTM模型 ({num_epochs} epochs)...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            inputs = batch[0]
            
            # 输入: 前n-1个时间步
            # 目标: 最后1个时间步
            inputs_seq = inputs[:, :-1, :]
            targets = inputs[:, -1, :]
            
            optimizer.zero_grad()
            outputs = model(inputs_seq)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                inputs_seq = inputs[:, :-1, :]
                targets = inputs[:, -1, :]
                
                outputs = model(inputs_seq)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # 在验证集上计算GC矩阵
    model.eval()
    base_errors = np.zeros((len(val_loader.dataset), num_features))
    perturb_errors = np.zeros((len(val_loader.dataset), num_features, num_features))
    
    # 计算基准误差
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch[0]
            inputs_seq = inputs[:, :-1, :]
            targets = inputs[:, -1, :]
            
            outputs = model(inputs_seq)
            errors = (outputs - targets).cpu().numpy()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(val_loader.dataset))
            base_errors[start_idx:end_idx] = errors
    
    # 计算每个特征扰动后的误差
    with torch.no_grad():
        for feature_idx in range(num_features):
            for i, batch in enumerate(val_loader):
                inputs = batch[0]
                inputs_seq = inputs[:, :-1, :].clone()
                
                # 扰动特定特征的历史数据
                inputs_seq[:, :, feature_idx] = 0.0
                
                targets = inputs[:, -1, :]
                
                outputs = model(inputs_seq)
                errors = (outputs - targets).cpu().numpy()
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(val_loader.dataset))
                perturb_errors[start_idx:end_idx, :, feature_idx] = errors
    
    # 计算GC矩阵
    gc_matrix = np.zeros((num_features, num_features))
    
    for i in range(num_features):
        for j in range(num_features):
            # 计算扰动特征j对预测特征i的影响
            base_mse = np.mean(base_errors[:, i] ** 2)
            perturb_mse = np.mean(perturb_errors[:, i, j] ** 2)
            
            # 格兰杰因果度量：误差增加的比例
            gc_matrix[i, j] = (perturb_mse - base_mse) / (base_mse + EPSILON)
    
    # 保存GC矩阵
    np.save(cache_path, gc_matrix)
    
    elapsed = time.time() - start_time
    print(f"GC矩阵计算完成! 耗时: {elapsed:.2f}秒")
    print(f"GC矩阵形状: {gc_matrix.shape}")
    print(f"GC矩阵统计 - 最小值: {gc_matrix.min():.4f}, 最大值: {gc_matrix.max():.4f}, 平均值: {gc_matrix.mean():.4f}")
    
    return gc_matrix


def create_fused_adjacency_matrix(gc_matrix, channel_names, derived_features=None):
    """
    创建融合物理拓扑和格兰杰因果的邻接矩阵
    Args:
        gc_matrix: 格兰杰因果矩阵
        channel_names: 原始通道名称列表
        derived_features: 派生特征名称列表（可选）
    Returns:
        adj_matrix: 融合后的邻接矩阵
    """
    # 确定节点数量
    num_raw_features = len(channel_names)
    if derived_features:
        num_nodes = num_raw_features + len(derived_features)
    else:
        num_nodes = num_raw_features
    
    # 创建空邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # 1. 添加物理拓扑关系
    
    # 蓄电池子系统
    battery_nodes = [0, 1, 2, 3, 5, 6]  # 原始特征中的索引
    for i in battery_nodes:
        for j in battery_nodes:
            if i != j:
                adj_matrix[i, j] = 1.0
    
    # 母线连接关系
    bus_node = 4
    bus_connected = [0, 1, 7, 8, 9, 10, 11]
    for node in bus_connected:
        adj_matrix[bus_node, node] = 1.0
        adj_matrix[node, bus_node] = 1.0
    
    # S3R控制设备连接
    s3r_nodes = [7, 8, 9, 10, 11]
    for i in s3r_nodes:
        for j in s3r_nodes:
            if i != j:
                adj_matrix[i, j] = 1
    
    # 工况模式连接
    mode_node = 12
    for node in [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]:
        adj_matrix[mode_node, node] = 1
        adj_matrix[node, mode_node] = 1
    
    # 目标识别率连接
    target_node = 13
    for node in [4, 7, 8, 9]:
        adj_matrix[target_node, node] = 1
        adj_matrix[node, target_node] = 1
    
    # 2. 如果有派生特征，添加派生特征与原始特征的关系
    if derived_features:
        # 派生特征索引从len(channel_names)=14开始
        dependencies = {
            14: [0, 4],  # Psolar: I_solar_out, main_bus_voltage
            15: [1, 6],  # Pcharge: VNA2, CHARG
            16: [1, 5],  # Pdischarge: VNA2, DISCHARG
            17: [4, 8],  # ηcharge: main_bus_voltage, BCR
            18: [4, 7],  # ηdischarge: main_bus_voltage, BDR
            19: [2, 3],  # ΔVcell: A1_out, A2_11_out
            20: [1, 5, 6],  # SOC_rate: VNA2, DISCHARG, CHARG
            21: [1, 19],  # Cell_imbalance: VNA2, ΔVcell
            22: [4, 9],  # Reg_error: main_bus_voltage, mea_S3R
            23: [10, 11]  # Bat_ctrl_eff: mea, BEA
        }
        
        for derived_idx, deps in dependencies.items():
            for dep in deps:
                if dep < adj_matrix.shape[0]:
                    adj_matrix[derived_idx, dep] = 1.0
                    adj_matrix[dep, derived_idx] = 1.0
    
    # 3. 融合GC矩阵
    if gc_matrix is not None:
        # 确保GC矩阵与邻接矩阵大小一致
        if gc_matrix.shape[0] != num_nodes or gc_matrix.shape[1] != num_nodes:
            # 如果大小不匹配，裁剪或填充
            print(f"警告: GC矩阵形状{gc_matrix.shape}与邻接矩阵形状{adj_matrix.shape}不匹配，进行调整")
            min_dim = min(gc_matrix.shape[0], num_nodes)
            gc_matrix_resized = np.zeros((num_nodes, num_nodes))
            gc_matrix_resized[:min_dim, :min_dim] = gc_matrix[:min_dim, :min_dim]
            gc_matrix = gc_matrix_resized
        
        # 取GC矩阵的绝对值
        gc_abs = np.abs(gc_matrix)
        
        # 归一化处理
        gc_min, gc_max = gc_abs.min(), gc_abs.max()
        if gc_max > gc_min:  # 避免除以零
            gc_norm = (gc_abs - gc_min) / (gc_max - gc_min + EPSILON)
        else:
            gc_norm = gc_abs
        
        # 融合策略：取物理拓扑和GC矩阵的最大值
        adj_matrix = np.maximum(adj_matrix, gc_norm)
        
        # 二值化处理：大于阈值的设为1
        adj_matrix = (adj_matrix > GC_THRESHOLD).astype(np.float32)
    
    # 添加自环
    np.fill_diagonal(adj_matrix, 1.0)
    
    return adj_matrix 