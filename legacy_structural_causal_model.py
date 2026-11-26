import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected

# 潜在空间维度
LATENT_DIM = 32


class SimpleCVAE(nn.Module):
    """简化的条件变分自编码器 - 直接重建原始特征"""

    def __init__(self, input_dim, latent_dim, num_classes):
        super(SimpleCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 潜在空间参数
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  # 加入条件
            # nn.Linear(latent_dim + num_classes, 64),  # 加入条件
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x, y):
        """编码输入特征到潜在空间"""
        # 扩展标签以匹配节点数量
        y_expanded = y.repeat_interleave(x.size(0) // y.size(0))
        y_onehot = F.one_hot(y_expanded, num_classes=self.num_classes).float()

        # 拼接特征和条件标签
        x_cond = torch.cat([x, y_onehot], dim=-1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        """从潜在空间解码特征"""
        return self.decoder(z)

    # def decode(self, z, y):
    #     """从潜在空间解码特征"""
    #     # 扩展标签以匹配潜在变量数量
    #     y_expanded = y.repeat_interleave(z.size(0) // y.size(0))
    #     y_onehot = F.one_hot(y_expanded, num_classes=self.num_classes).float()
    #
    #     # 拼接潜在变量和条件标签
    #     z_cond = torch.cat([z, y_onehot], dim=-1)
    #     return self.decoder(z_cond)

    def forward(self, x, y):
        """前向传播"""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar


class CVAE_GAT(nn.Module):
    """基于CVAE重建的图注意力网络"""

    def __init__(self, num_node_features, num_classes, hidden_dim=64, heads=8, dropout=0.5, latent_dim=32):
        super(CVAE_GAT, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # CVAE模块 - 重建原始特征
        self.cvae = SimpleCVAE(
            input_dim=num_node_features,
            latent_dim=latent_dim,
            num_classes=num_classes
        )

        # GAT层
        self.conv1 = GATConv(
            num_node_features,  # 输入维度是原始特征维度
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

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

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        num_nodes = x.size(0)

        # 保存原始输入特征
        original_x = x.clone()

        # 使用CVAE重建特征
        x_recon, mu, logvar = self.cvae(x, y)

        # 添加自环和转换为无向图
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = to_undirected(edge_index)

        # GAT层处理重建特征
        x = F.elu(self.conv1(x_recon, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 全局池化和分类
        x = global_mean_pool(x, batch)
        out = self.classifier(x)

        # 计算重建损失
        recon_loss = F.mse_loss(x_recon, original_x, reduction='mean')

        # KL散度损失（使用节点数归一化）
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / num_nodes

        return out, recon_loss, kl_loss, x 