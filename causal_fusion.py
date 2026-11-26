"""Topology-constrained CVAE, directed temporal GAT and dual-scale gated fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool


def _batched_edges(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert [batch, source, target] adjacency to PyG source->target edges."""
    batch, source, target = adjacency.nonzero(as_tuple=True)
    width = adjacency.shape[1]
    return torch.stack((batch * width + source, batch * width + target))


class ConditionalVAE(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim + num_classes, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU())
        self.posterior_mu = nn.Linear(64, latent_dim)
        self.posterior_logvar = nn.Linear(64, latent_dim)
        self.prior = nn.Sequential(nn.Linear(num_classes, 64), nn.ReLU(), nn.Linear(64, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim + num_classes, 64), nn.ReLU(),
                                     nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, feature_dim))

    def forward(self, features: torch.Tensor, condition: torch.Tensor):
        # The condition is always the auxiliary *prediction*, including in training.
        posterior = self.encoder(torch.cat((features, condition), dim=-1))
        mu, logvar = self.posterior_mu(posterior), self.posterior_logvar(posterior).clamp(-10, 10)
        prior_mu, prior_logvar = self.prior(condition).chunk(2, dim=-1)
        prior_logvar = prior_logvar.clamp(-10, 10)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        reconstruction = self.decoder(torch.cat((latent, condition), dim=-1))
        kl = 0.5 * (prior_logvar - logvar +
                    (logvar.exp() + (mu - prior_mu).square()) / prior_logvar.exp() - 1)
        return reconstruction, F.mse_loss(reconstruction, features), kl.sum(-1).mean()


class CausalFusionModel(nn.Module):
    def __init__(self, raw_nodes: int, temporal_nodes: int, num_classes: int,
                 spatial_adjacency: torch.Tensor, window_size: int = 100,
                 hidden_dim: int = 64, heads: int = 4, latent_dim: int = 16,
                 dropout: float = 0.3):
        super().__init__()
        if spatial_adjacency.shape != (raw_nodes, raw_nodes) or temporal_nodes != raw_nodes + 9:
            raise ValueError("Expected D raw nodes and D+9 temporal nodes with a D x D topology")
        self.raw_nodes = raw_nodes
        self.temporal_nodes = temporal_nodes
        self.dropout = dropout
        self.register_buffer("spatial_adjacency", spatial_adjacency.float())
        self.spatial_encoder = GATConv(window_size, hidden_dim, heads=heads,
                                       concat=False, dropout=dropout)
        self.auxiliary = nn.Linear(hidden_dim, num_classes)
        self.cvae = ConditionalVAE(hidden_dim, num_classes, latent_dim)
        self.spatial_graph = GATConv(hidden_dim, hidden_dim, heads=heads,
                                     concat=False, dropout=dropout)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, hidden_dim), nn.ReLU())
        self.temporal_graph1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                                       concat=False, dropout=dropout)
        self.temporal_graph2 = GATConv(hidden_dim, hidden_dim, heads=heads,
                                       concat=False, dropout=dropout)
        self.edge_gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.feature_gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.fusion_graph = GATConv(hidden_dim, hidden_dim, heads=heads,
                                    concat=False, edge_dim=1, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, raw: torch.Tensor, temporal: torch.Tensor,
                gc_adjacency: torch.Tensor):
        batch_size, raw_nodes, window = raw.shape
        if raw_nodes != self.raw_nodes or temporal.shape != (batch_size, self.temporal_nodes, window):
            raise ValueError("Input dimensions do not match the configured model")
        if gc_adjacency.shape != (batch_size, self.temporal_nodes, self.temporal_nodes):
            raise ValueError("Temporal adjacency must be [batch, D+9, D+9]")
        batch_ids = torch.arange(batch_size, device=raw.device).repeat_interleave(raw_nodes)
        spatial_edges = _batched_edges(self.spatial_adjacency.unsqueeze(0).expand(batch_size, -1, -1))
        initial = F.elu(self.spatial_encoder(raw.reshape(-1, window), spatial_edges))
        aux_logits = self.auxiliary(global_mean_pool(initial, batch_ids))
        condition = aux_logits.softmax(-1)[batch_ids]
        reconstructed, recon_loss, kl_loss = self.cvae(initial, condition)
        spatial, (attention_edges, weights) = self.spatial_graph(
            reconstructed, spatial_edges, return_attention_weights=True)
        spatial = F.elu(spatial).reshape(batch_size, raw_nodes, -1)
        edge_batch = attention_edges[0] // raw_nodes
        attention = torch.zeros((batch_size, raw_nodes, raw_nodes), device=raw.device)
        attention = attention.index_put((edge_batch, attention_edges[0] % raw_nodes,
                                         attention_edges[1] % raw_nodes), weights.mean(-1))

        temporal_edges = _batched_edges(gc_adjacency)
        # PyG adds self-loops. Explicitly supplied directed edges are never symmetrized.
        temporal_encoded = self.temporal_encoder(temporal.reshape(-1, 1, window))
        temporal_hidden = F.elu(self.temporal_graph1(temporal_encoded, temporal_edges))
        temporal_hidden = F.elu(self.temporal_graph2(temporal_hidden, temporal_edges))
        temporal_aligned = temporal_hidden.reshape(batch_size, self.temporal_nodes, -1)[:, :raw_nodes]
        temporal_adj_aligned = gc_adjacency[:, :raw_nodes, :raw_nodes].float()

        candidates = ((attention > 0) | (temporal_adj_aligned > 0)).nonzero(as_tuple=False)
        graph_index, source, target = candidates.T
        pair = torch.cat((temporal_aligned[graph_index, source],
                          spatial[graph_index, target]), dim=-1)
        edge_gate = self.edge_gate(pair).squeeze(-1)
        fused_weights = edge_gate * temporal_adj_aligned[graph_index, source, target] + \
            (1 - edge_gate) * attention[graph_index, source, target]
        fused_adj = torch.zeros_like(attention).index_put(
            (graph_index, source, target), fused_weights)
        feature_gate = self.feature_gate(torch.cat((temporal_aligned, spatial), dim=-1))
        fused_nodes = feature_gate * temporal_aligned + (1 - feature_gate) * spatial
        fused_edges = _batched_edges(fused_adj)
        edge_attr = fused_adj[fused_adj > 0].unsqueeze(-1)
        aggregated = F.elu(self.fusion_graph(fused_nodes.reshape(-1, fused_nodes.shape[-1]),
                                              fused_edges, edge_attr=edge_attr))
        pooled = global_mean_pool(aggregated, batch_ids)
        return {"logits": self.classifier(pooled), "aux_logits": aux_logits,
                "recon_loss": recon_loss, "kl_loss": kl_loss,
                "feature_gate": feature_gate, "edge_gate": edge_gate,
                "fused_adjacency": fused_adj}
