"""Exact executable external-baseline architectures frozen for the V2 pilot."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINEConv, TransformerConv
from torch_geometric.nn import global_max_pool, global_mean_pool


def time_features(times: torch.Tensor) -> torch.Tensor:
    tau = (times.clamp_min(0.0) / 30.0).clamp(max=1.0)
    values = [tau, tau.square(), tau.sqrt()]
    for harmonic in (1.0, 2.0, 4.0, 8.0):
        phase = math.pi * harmonic * tau
        values.extend([torch.sin(phase), torch.cos(phase)])
    return torch.stack(values, dim=-1)


class DirectTrajectoryDecoder(nn.Module):
    """The frozen direct time-conditioned decoder, parameterized only by width."""
    def __init__(self, hidden: int, dropout: float = 0.05):
        super().__init__()
        self.device_encoder = nn.Sequential(
            nn.Linear(3 * hidden + 14, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(11, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, 2 * hidden)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1),
        )
        nn.init.normal_(self.decoder[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, h_node, batch, rows, data):
        batch_count, n_dev = rows.shape
        graph_mean = global_mean_pool(h_node, batch)
        graph_max = global_max_pool(h_node, batch)
        pair = torch.cat([graph_mean, graph_max], -1).unsqueeze(1).expand(-1, n_dev, -1)
        ptr = data.ptr[:-1].view(-1, 1)
        h_dev = h_node[rows.clamp_min(0) + ptr]
        token = self.device_encoder(torch.cat([h_dev, pair, data.controller_features], -1))
        gamma, beta = self.time_encoder(time_features(data.phys_rollout_t)).chunk(2, -1)
        z = token.unsqueeze(2) * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        pred = self.decoder(z).squeeze(-1)
        return torch.where(data.phys_gad_dev_mask.unsqueeze(-1), pred, torch.zeros_like(pred))


class DirectMessagePassing(nn.Module):
    def __init__(self, kind: str, node_dim: int, hidden: int, layers: int = 4, dropout: float = 0.05):
        super().__init__()
        self.kind = kind
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if kind == "gat":
                if hidden % 4:
                    raise ValueError("GAT width must be divisible by four")
                layer = GATConv(hidden, hidden // 4, heads=4, concat=True, dropout=dropout)
            elif kind == "gcn":
                layer = GCNConv(hidden, hidden)
            else:
                raise ValueError(kind)
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.decoder = DirectTrajectoryDecoder(hidden, dropout)

    def forward(self, data):
        x = self.node_encoder(data.x)
        for layer, norm in zip(self.layers, self.norms):
            update = layer(x, data.edge_index)
            x = self.dropout(torch.nn.functional.silu(norm(x + update)))
        return self.decoder(x, data.batch, data.phys_dev_node_row, data)


class GraphicalDeepONet(nn.Module):
    """Bus-only published encoder with a post-graph variable-device adapter."""
    def __init__(self, node_dim: int, hidden: int = 256, q: int = 256, dropout: float = 0.05):
        super().__init__()
        if hidden != 256 or q != 256:
            raise ValueError("V2 freezes hidden=q=256")
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.transformers = nn.ModuleList([
            TransformerConv(hidden, 64, heads=4, concat=True, edge_dim=3, dropout=dropout, beta=False)
            for _ in range(2)
        ])
        self.transformer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        gine_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gine = GINEConv(gine_mlp, edge_dim=3, train_eps=True)
        self.gine_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.device_query = nn.Sequential(
            nn.Linear(2 * hidden + 14, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, q),
        )
        self.trunk = nn.Sequential(
            nn.Linear(11, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, q)
        )
        self.q = q

    def forward(self, data):
        x = self.node_encoder(data.bus_x)
        for layer, norm in zip(self.transformers, self.transformer_norms):
            update = layer(x, data.bus_edge_index, data.bus_edge_attr)
            x = self.dropout(torch.nn.functional.relu(norm(x + update)))
        x = torch.nn.functional.relu(self.gine_norm(x + self.gine(x, data.bus_edge_index, data.bus_edge_attr)))
        graph_global = global_mean_pool(x, data.bus_batch)
        n_dev = data.terminal_bus_row.shape[1]
        global_device = graph_global.unsqueeze(1).expand(-1, n_dev, -1)
        terminal = x[data.terminal_bus_row.clamp_min(0)]
        coefficients = self.device_query(torch.cat([global_device, terminal, data.controller_features], -1))
        kernels = self.trunk(time_features(data.phys_rollout_t))
        pred = torch.einsum("bdq,btq->bdt", coefficients, kernels) / math.sqrt(self.q)
        return torch.where(data.phys_gad_dev_mask.unsqueeze(-1), pred, torch.zeros_like(pred))


class ComplexPolynomialLayer(nn.Module):
    def __init__(self, hidden: int, order: int = 2):
        super().__init__()
        self.order = order
        scale = 1.0 / math.sqrt(hidden)
        self.weight_real = nn.Parameter(torch.randn(order + 1, hidden, hidden) * scale)
        self.weight_imag = nn.Parameter(torch.randn(order + 1, hidden, hidden) * scale)
        self.bias_real = nn.Parameter(torch.zeros(hidden))
        self.bias_imag = nn.Parameter(torch.zeros(hidden))
        self.norm_real = nn.LayerNorm(hidden)
        self.norm_imag = nn.LayerNorm(hidden)

    def forward(self, real, imag, shift):
        out_r = self.bias_real.expand_as(real).clone()
        out_i = self.bias_imag.expand_as(imag).clone()
        power_r, power_i = real, imag
        sr, si = shift.real, shift.imag
        for k in range(self.order + 1):
            wr, wi = self.weight_real[k], self.weight_imag[k]
            out_r = out_r + power_r @ wr - power_i @ wi
            out_i = out_i + power_r @ wi + power_i @ wr
            if k < self.order:
                power_r, power_i = sr @ power_r - si @ power_i, sr @ power_i + si @ power_r
        return torch.nn.functional.silu(self.norm_real(out_r)), torch.nn.functional.silu(self.norm_imag(out_i))


class UGCNTraj(nn.Module):
    """T=1 prospective complex graph polynomial model with adaptive Np=4 pooling."""
    def __init__(self, node_dim: int, hidden: int = 256, order: int = 2, pooled_nodes: int = 4, dropout: float = 0.05):
        super().__init__()
        if (hidden, order, pooled_nodes) != (256, 2, 4):
            raise ValueError("V2 freezes hidden=256, K=2, Np=4")
        self.real_encoder = nn.Linear(node_dim, hidden)
        self.imag_encoder = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([ComplexPolynomialLayer(hidden, order) for _ in range(2)])
        self.assignment = nn.Linear(2 * hidden, pooled_nodes)
        self.real_projection = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.decoder = DirectTrajectoryDecoder(hidden, dropout)
        self.pooled_nodes = pooled_nodes

    @staticmethod
    def normalized_shift(gso):
        # Per-sample deterministic scaling; no target-derived fitted statistic.
        scale = torch.linalg.matrix_norm(gso, ord=2).clamp_min(1e-8)
        return gso / scale

    def forward(self, data):
        all_bus = []
        all_pool = []
        for graph in range(len(data.gso)):
            lo, hi = int(data.bus_ptr[graph]), int(data.bus_ptr[graph + 1])
            bx = data.bus_x[lo:hi]
            real, imag = self.real_encoder(bx), self.imag_encoder(bx)
            shift = self.normalized_shift(data.gso[graph])
            for layer in self.layers:
                update_r, update_i = layer(real, imag, shift)
                real, imag = real + update_r, imag + update_i
            joined = torch.cat([real, imag], -1)
            bus_real = self.real_projection(joined)
            assignment = torch.softmax(self.assignment(joined), dim=0)
            pooled = assignment.transpose(0, 1) @ bus_real
            all_bus.append(bus_real)
            all_pool.append(pooled)
        x = torch.cat(all_bus, 0)
        pooled = torch.stack(all_pool, 0)
        graph_mean = pooled.mean(1)
        graph_max = pooled.max(1).values
        n_dev = data.terminal_bus_row.shape[1]
        terminal = x[data.terminal_bus_row.clamp_min(0)]
        pair = torch.cat([graph_mean, graph_max], -1).unsqueeze(1).expand(-1, n_dev, -1)
        # Reuse the exact direct decoder modules but provide precomputed bus/pool context.
        token = self.decoder.device_encoder(torch.cat([terminal, pair, data.controller_features], -1))
        gamma, beta = self.decoder.time_encoder(time_features(data.phys_rollout_t)).chunk(2, -1)
        z = token.unsqueeze(2) * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        pred = self.decoder.decoder(z).squeeze(-1)
        return torch.where(data.phys_gad_dev_mask.unsqueeze(-1), pred, torch.zeros_like(pred))


def build_model(name: str, node_dim: int):
    if name == "graphical_deeponet":
        return GraphicalDeepONet(node_dim)
    if name == "ugcn":
        return UGCNTraj(node_dim)
    if name.startswith("gat_h"):
        return DirectMessagePassing("gat", node_dim, int(name.split("h")[-1]))
    if name.startswith("gcn_h"):
        return DirectMessagePassing("gcn", node_dim, int(name.split("h")[-1]))
    raise ValueError(name)


def parameter_inventory(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return trainable, frozen, buffers
