import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# 常量定义
WINDOW_SIZE = 100  # 时间窗口大小
STRIDE = 50  # 窗口滑动步长
EPSILON = 1e-8  # 避免除零的小量

# 遥测通道名称映射
CHANNEL_NAMES = [
    "蓄电池电流I_solar_out",
    "蓄电池电压VNA2",
    "蓄电池单体电压A1_out",
    "蓄电池单体电压A2_11_out",
    "母线电压main_bus_voltage",
    "蓄电池放电电流DISCHARG",
    "蓄电池充电电流CHARG",
    "BDR电流BDR",
    "BCR电流BCR",
    "mea_S3R电压mea_S3R",
    "MEA电压mea",
    "BEA电压BEA",
    "电源系统工况MODE",
    "负载系统目标识别率Target_Identification_Rate"
]

# 派生特征名称
DERIVED_FEATURES = [
    "Psolar", "Pcharge", "Pdischarge", "ηcharge", "ηdischarge",
    "ΔVcell", "SOC_rate", "Cell_imbalance", "Reg_error", "Bat_ctrl_eff"
]

# 所有特征名称
ALL_FEATURE_NAMES = CHANNEL_NAMES + DERIVED_FEATURES
NUM_ALL_FEATURES = len(ALL_FEATURE_NAMES)  # 总特征数


def create_power_system_topology():
    """基于卫星电源系统物理拓扑创建邻接矩阵"""
    # 创建空邻接矩阵
    num_nodes = len(CHANNEL_NAMES)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 定义电源系统拓扑关系
    # 1. 蓄电池子系统
    battery_nodes = [0, 1, 2, 3, 5, 6]  # 蓄电池相关通道
    for i in battery_nodes:
        for j in battery_nodes:
            if i != j:
                adj_matrix[i, j] = 1.0  # 蓄电池内部全连接

    # 2. 母线连接关系
    bus_node = 4  # 母线电压
    bus_connected = [0, 1, 7, 8, 9, 10, 11]  # 与母线直接连接的设备
    for node in bus_connected:
        adj_matrix[bus_node, node] = 1.0
        adj_matrix[node, bus_node] = 1.0

    # 3. S3R控制设备连接
    s3r_nodes = [7, 8, 9, 10, 11]  # BDR, BCR, S3R, MEA, BEA
    for i in s3r_nodes:
        for j in s3r_nodes:
            if i != j:
                adj_matrix[i, j] = 1  # S3R设备间中等强度连接

    # 4. 工况模式连接
    mode_node = 12
    # 工况模式影响所有关键设备
    for node in [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]:
        adj_matrix[mode_node, node] = 1
        adj_matrix[node, mode_node] = 1

    # 5. 目标识别率连接
    target_node = 13
    # 目标识别率与电源输出相关
    for node in [4, 7, 8, 9]:  # 母线电压、BDR、BCR、S3R
        adj_matrix[target_node, node] = 1
        adj_matrix[node, target_node] = 1

    # 转换为PyG边索引格式
    edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix))
    return edge_index, adj_matrix


def create_all_features_topology(adj_matrix_raw):
    """创建包含派生特征的邻接矩阵"""
    # 复制原始邻接矩阵到左上角
    num_all_features = len(ALL_FEATURE_NAMES)
    num_raw_features = len(CHANNEL_NAMES)
    
    adj_matrix = np.zeros((num_all_features, num_all_features), dtype=np.float32)
    adj_matrix[:num_raw_features, :num_raw_features] = adj_matrix_raw
    
    # 添加派生特征与原始特征的关系
    # 派生特征索引从len(CHANNEL_NAMES)=14开始
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
            adj_matrix[derived_idx, dep] = 1.0
            adj_matrix[dep, derived_idx] = 1.0
            
    return adj_matrix


def compute_derived_features(raw_data):
    """
    计算派生特征
    Args:
        raw_data: 原始数据，形状为(num_samples, num_raw_features)
    Returns:
        derived_data: 派生特征，形状为(num_samples, num_derived_features)
    """
    # 解析原始特征列索引
    I_solar_out = raw_data[:, 0]
    VNA2 = raw_data[:, 1]
    A1_out = raw_data[:, 2]
    A2_11_out = raw_data[:, 3]
    main_bus_voltage = raw_data[:, 4]
    DISCHARG = raw_data[:, 5]
    CHARG = raw_data[:, 6]
    BDR = raw_data[:, 7]
    BCR = raw_data[:, 8]
    mea_S3R = raw_data[:, 9]
    mea = raw_data[:, 10]
    BEA = raw_data[:, 11]
    
    # 计算派生特征
    Psolar = I_solar_out * main_bus_voltage
    Pcharge = VNA2 * CHARG
    Pdischarge = VNA2 * DISCHARG
    ηcharge = Pcharge / (main_bus_voltage * BCR + EPSILON)
    ηdischarge = (main_bus_voltage * BDR) / (Pdischarge + EPSILON)
    ΔVcell = np.abs(A1_out - A2_11_out)
    SOC_rate = (CHARG - DISCHARG) / (VNA2 + EPSILON)
    Cell_imbalance = ΔVcell / (VNA2 + EPSILON)
    Reg_error = mea_S3R - main_bus_voltage  # 假设k=1
    Bat_ctrl_eff = np.abs(BEA) / (np.abs(mea) + EPSILON)
    
    # 组合所有派生特征
    derived_features = np.column_stack((
        Psolar, Pcharge, Pdischarge, ηcharge, ηdischarge,
        ΔVcell, SOC_rate, Cell_imbalance, Reg_error, Bat_ctrl_eff
    ))
    
    return derived_features


def load_data(data_dir):
    """
    加载所有故障数据文件
    Args:
        data_dir: 数据目录路径
    Returns:
        file_labels: 文件到标签的映射字典
        all_data: 所有原始数据的列表
    """
    # 获取所有CSV文件
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    file_list.sort()  # 确保文件顺序一致
    
    # 为每个文件分配标签
    file_labels = {file: i for i, file in enumerate(file_list)}
    print("文件标签映射:")
    for file, label in file_labels.items():
        print(f"{file} -> 故障类别 {label}")
    
    # 处理每个文件
    all_data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)
        
        # 确保数据有15列（时间列+14个遥测通道）
        if df.shape[1] != 15:
            print(f"警告: 文件 {file_name} 有 {df.shape[1]} 列，预期为15列")
            continue
        
        # 剔除时间列（第一列）
        data = df.iloc[:, 1:].values
        all_data.append(data)
    
    return file_labels, all_data


class PowerSystemDataset(Dataset):
    """卫星电源系统故障诊断数据集类"""
    
    def __init__(self, data_dir, window_size=WINDOW_SIZE, stride=STRIDE, include_derived_features=True):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.include_derived_features = include_derived_features
        self.file_labels = {}  # 文件到标签的映射
        self.graphs = []  # 存储图数据对象
        self.labels = []  # 存储所有标签
        self.scaler = StandardScaler()
        self.sample_to_file = []  # 样本到文件的映射
        
        # 加载数据并处理
        self.process_data()
    
    def process_data(self):
        # 加载所有文件数据
        self.file_labels, all_data = load_data(self.data_dir)
        
        if not all_data:
            raise ValueError("没有找到有效数据文件")
        
        # 合并所有数据进行标准化
        all_data_concat = np.vstack(all_data)
        self.scaler.fit(all_data_concat)
        
        # 创建图结构的拓扑关系
        edge_index, adj_matrix = create_power_system_topology()
        self.edge_index = edge_index
        
        if self.include_derived_features:
            self.adj_matrix_all = create_all_features_topology(adj_matrix)
        
        # 为每个文件生成图样本
        for file_name, file_data in zip(self.file_labels.keys(), all_data):
            label = self.file_labels[file_name]
            
            # 标准化
            norm_data = self.scaler.transform(file_data)
            
            # 生成时间窗口样本
            num_windows = (len(norm_data) - self.window_size) // self.stride + 1
            for j in range(num_windows):
                start = j * self.stride
                end = start + self.window_size
                window = norm_data[start:end]
                
                # 提取节点特征 (window_size, num_features) -> (num_features, window_size)
                node_features = window.T
                
                if self.include_derived_features:
                    # 计算派生特征
                    derived_features = compute_derived_features(window)
                    # 转置为 (num_derived_features, window_size)
                    derived_features = derived_features.T
                    
                    # 合并原始特征和派生特征
                    all_features = np.vstack((node_features, derived_features))
                    
                    # 创建包含派生特征的图数据对象
                    graph = Data(
                        x=torch.tensor(all_features, dtype=torch.float32),
                        edge_index=None,  # 稍后设置
                        y=torch.tensor([label], dtype=torch.long)
                    )
                else:
                    # 创建只包含原始特征的图数据对象
                    graph = Data(
                        x=torch.tensor(node_features, dtype=torch.float32),
                        edge_index=self.edge_index,
                        y=torch.tensor([label], dtype=torch.long)
                    )
                
                self.graphs.append(graph)
                self.labels.append(label)
                self.sample_to_file.append(file_name)
        
        print(f"\n数据集统计:")
        print(f"总样本数: {len(self.graphs)}")
        if self.graphs:
            print(f"原始节点特征形状: {self.graphs[0].x.shape}")
        
        # 打印类别分布
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\n类别分布:")
        for cls, count in zip(unique, counts):
            print(f"故障类别 {cls}: {count} 个样本 ({count / len(self.graphs):.2%})")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if self.include_derived_features:
            # 为包含派生特征的图设置边索引
            edge_index_all, _ = dense_to_sparse(torch.tensor(self.adj_matrix_all))
            self.graphs[idx].edge_index = edge_index_all
        
        return self.graphs[idx]


# 测试代码
if __name__ == "__main__":
    data_dir = r'D:\Dataset_Real\Paper3-Satellite电源系统故障诊断'
    dataset = PowerSystemDataset(data_dir, include_derived_features=True)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 查看第一个样本
    sample = dataset[0]
    print(f"样本特征形状: {sample.x.shape}")
    print(f"边索引形状: {sample.edge_index.shape}")
    print(f"标签: {sample.y.item()}")
    
    # 检查是否包含派生特征
    if sample.x.shape[0] > len(CHANNEL_NAMES):
        print(f"包含派生特征，总特征数: {sample.x.shape[0]}")
        print(f"派生特征: {DERIVED_FEATURES}") 