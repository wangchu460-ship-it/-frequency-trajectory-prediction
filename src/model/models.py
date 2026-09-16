from __future__ import annotations

import math
import torch
from torch import nn
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool

from .haag_layers import HAAGLayer


def laplacian_message(x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Compute l_i = sum_j w_ij (h_i - h_j) from directed edges j -> i."""
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    src = edge_index[0]
    dst = edge_index[1]
    weight = edge_weight.to(dtype=x.dtype, device=x.device).view(-1, 1)
    msg = weight * (x[dst] - x[src])
    out = torch.zeros_like(x)
    out.index_add_(0, dst, msg)
    return out


class LaplacianTransformerBlock(nn.Module):
    """Transformer message passing with an optional F-weighted Laplacian channel.

    ``laplacian_fusion="simple"`` uses the paper-friendly update
        h_i' = LN(h_i + m_i + sigma * W_l l_i)
    while ``"gated"`` keeps the older conservative gate for checkpoint
    compatibility and ablations.
    """

    def __init__(self, hidden_dim: int, heads: int, dropout: float, use_laplacian: bool,
                 laplacian_fusion: str = "gated", sigma_init: float = 0.1,
                 gate_bias_init: float = -3.0) -> None:
        super().__init__()
        self.use_laplacian = use_laplacian
        self.laplacian_fusion = str(laplacian_fusion)
        if self.laplacian_fusion not in {"gated", "simple"}:
            raise ValueError("laplacian_fusion must be 'gated' or 'simple'")
        self.conv = TransformerConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=hidden_dim,
            dropout=dropout,
            beta=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if use_laplacian:
            if self.laplacian_fusion == "simple":
                self.lap_mlp = nn.Linear(hidden_dim, hidden_dim)
            else:
                self.lap_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.gate_mlp = nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                # Initialise the gate so it is closed at the start. Only shift the
                # bias of the *last* Linear before the Sigmoid; leave the weights at
                # their default random init so gradients still flow through the
                # earlier layers of gate_mlp.
                for m in reversed(self.gate_mlp):
                    if isinstance(m, nn.Linear):
                        nn.init.constant_(m.bias, gate_bias_init)
                        break
            # Learnable scalar sigma for the whole Laplacian channel.
            self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, data) -> torch.Tensor:
        residual = x
        m = self.conv(x, edge_index, edge_attr=edge_attr)
        if self.use_laplacian and hasattr(data, "lap_edge_index") and hasattr(data, "lap_edge_weight") \
                and data.lap_edge_index.numel() > 0:
            lap = laplacian_message(x, data.lap_edge_index, data.lap_edge_weight)
            lap = self.lap_mlp(lap)
            if self.laplacian_fusion == "simple":
                x = self.norm(residual + m + self.sigma * lap)
            else:
                gate = self.gate_mlp(torch.cat([x, m, lap], dim=-1))
                x = self.norm(residual + m + self.sigma * gate * lap)
        else:
            x = self.norm(residual + m)
        x = torch.nn.functional.silu(x)
        return self.dropout(x)


class GraphLevelGNN(nn.Module):
    """GNN with optional prediction heads:
        - class_logits       : graph-level classification (3 classes)
        - regression         : graph-level scalar regression (5 targets)
        - node_regression    : per-node regression (Stage 1: nadir, max|RoCoF|)
        - q_theta            : per-node parameter corrections (Stage 3, 7 slots,
                                masked by device family). Bounded by tanh*scale.
        - r_theta            : per-node residual-strength scalar (Stage 3).
        - q_theta_raw        : tanh outputs BEFORE masking, for diagnostic only.

    Stage 3 adds q_theta and r_theta as auxiliary outputs that Stages 4/5 will
    consume in the descriptor state-space equation. The training signal for
    them comes from (i) sharing the GNN backbone with the main prediction
    heads, and (ii) optional L2 regularisation in train.py to keep them small
    (physics-first; residual where genuinely needed).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        num_classes: int = 3,
        num_regression_targets: int = 5,
        num_node_targets: int = 2,
        num_q_params: int = 7,
        num_r_params: int = 3,
        q_tanh_scale: float = 0.3,
        dropout: float = 0.1,
        use_laplacian: bool = False,
        laplacian_fusion: str = "gated",
        graph_mode: str = "laplacian",
        edge_dim_phy: int = 6,
        node_slow_dim: int | None = None,
        node_fast_dim: int | None = None,
        dyn_topk: int = 6,
        dyn_topk_mode: str = "fixed",
        dyn_topk_min: int = 6,
        dyn_topk_max: int = 12,
        dyn_topk_scale: float = 1.5,
        dyn_mass_threshold: float = 0.90,
        dyn_relation_mode: str = "global_topk",
        dyn_structured_hops: int = 2,
        dyn_mandatory_weight: float = 0.15,
        dyn_gate_init: float = -4.0,
        phy_operator: str = "message",
        dyn_typepair: bool = True,
        haag_fast_v2: bool = False,
        haag_fast_v21: bool = False,
        haag_split_readout: bool = False,
        haag_unified_dynamic: bool = False,
        haag_compact_dynamic: bool = False,
        unified_dyn_gate_init: float = 0.04,
        unified_dyn_gate_max: float = 0.15,
        support_topology: str = "shared",
        graph_context_dim: int = 0,
        use_global_context: bool = False,
        global_context_gate_init: float = 0.05,
        global_context_gate_max: float = 0.25,
        use_system_adapter: bool = True,
        enable_r: bool = False,
        q_aware_heads: bool = False,
        prediction_heads: bool = True,
        q_channel_scales: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        architecture_screen_mode: str = "legacy",
    ) -> None:
        super().__init__()
        self.q_tanh_scale = float(q_tanh_scale)
        self.laplacian_fusion = str(laplacian_fusion)
        self.graph_mode = str(graph_mode)
        if self.graph_mode not in {"laplacian", "phy_graph", "haag", "dyn_only"}:
            raise ValueError("graph_mode must be 'laplacian', 'phy_graph', 'haag', or 'dyn_only'")
        self.enable_r = bool(enable_r)
        self.num_r_params = int(num_r_params)
        self.node_slow_dim = int(node_slow_dim) if node_slow_dim is not None else int(node_dim)
        self.node_fast_dim = int(node_fast_dim) if node_fast_dim is not None else int(node_dim)
        self.dyn_warmup_scale = 1.0
        self.haag_fast_v2 = bool(haag_fast_v2)
        self.haag_fast_v21 = bool(haag_fast_v21)
        self.haag_split_readout = bool(haag_split_readout)
        self.haag_unified_dynamic = bool(haag_unified_dynamic)
        self.haag_compact_dynamic = bool(haag_compact_dynamic)
        self.use_global_context = bool(use_global_context)
        self.global_context_gate_max = float(global_context_gate_max)
        if sum((self.haag_fast_v2, self.haag_fast_v21, self.haag_split_readout, self.haag_unified_dynamic)) > 1:
            raise ValueError("HAAG fast-v2, fast-v21, split-readout, and unified-dynamic modes are mutually exclusive")
        self.q_aware_heads = bool(q_aware_heads)
        self.prediction_heads = bool(prediction_heads)
        self.architecture_screen_mode = str(architecture_screen_mode).lower()
        if self.architecture_screen_mode not in {"legacy", "e0_direct", "e2a_residual"}:
            raise ValueError("architecture_screen_mode must be legacy/e0_direct/e2a_residual")
        self.direct_trajectory = self.architecture_screen_mode != "legacy"
        if q_channel_scales is None:
            q_channel_scales_t = torch.ones(num_q_params, dtype=torch.float32)
        else:
            q_channel_scales_t = torch.as_tensor(q_channel_scales, dtype=torch.float32).flatten()
            if q_channel_scales_t.numel() != num_q_params:
                raise ValueError(f"q_channel_scales must have {num_q_params} values")
        self.register_buffer("q_channel_scales", q_channel_scales_t, persistent=False)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        if self.use_global_context:
            if int(graph_context_dim) <= 0 or self.global_context_gate_max <= 0.0:
                raise ValueError("global context requires positive feature dimension and gate maximum")
            self.global_context_encoder = nn.Sequential(
                nn.Linear(int(graph_context_dim), hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            ratio = min(max(float(global_context_gate_init) / self.global_context_gate_max, 1e-6), 1.0 - 1e-6)
            self.global_context_gate_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        else:
            self.global_context_encoder = None
            self.register_parameter("global_context_gate_logit", None)

        self.blocks = nn.ModuleList()
        if self.graph_mode == "laplacian":
            for _ in range(num_layers):
                self.blocks.append(
                    LaplacianTransformerBlock(
                        hidden_dim=hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        use_laplacian=use_laplacian,
                        laplacian_fusion=self.laplacian_fusion,
                    )
                )
        else:
            for _ in range(num_layers):
                self.blocks.append(
                    HAAGLayer(
                        hidden_dim=hidden_dim,
                        edge_dim_phy=edge_dim_phy,
                        node_feature_dim=node_dim,
                        node_feature_dim_slow=self.node_slow_dim,
                        node_feature_dim_fast=self.node_fast_dim,
                        dyn_topk=dyn_topk,
                        dyn_topk_mode=dyn_topk_mode,
                        dyn_topk_min=dyn_topk_min,
                        dyn_topk_max=dyn_topk_max,
                        dyn_topk_scale=dyn_topk_scale,
                        dyn_mass_threshold=dyn_mass_threshold,
                        dyn_relation_mode=dyn_relation_mode,
                        dyn_structured_hops=dyn_structured_hops,
                        dyn_mandatory_weight=dyn_mandatory_weight,
                        dyn_gate_init=dyn_gate_init,
                        phy_operator=phy_operator,
                        use_typepair_bias=bool(dyn_typepair),
                        fast_v2=self.haag_fast_v2,
                        fast_v21=self.haag_fast_v21,
                        split_readout=self.haag_split_readout,
                        unified_dynamic=self.haag_unified_dynamic,
                        compact_unified_dynamic=self.haag_compact_dynamic,
                        dyn_heads=heads,
                        unified_gate_init=unified_dyn_gate_init,
                        unified_gate_max=unified_dyn_gate_max,
                        dropout=dropout,
                        support_topology=support_topology,
                    )
                )

        self.dropout = nn.Dropout(dropout)
        # IEEE39 and NPCC140 share the physical q/r heads, but their network
        # scales and converter interactions are materially different.  A
        # zero-initialised low-rank system adapter lets the common backbone
        # specialise without changing the exact E0 output at initialisation.
        self.system_adapter = nn.Embedding(2, hidden_dim) if bool(use_system_adapter) else None
        if self.system_adapter is not None:
            nn.init.zeros_(self.system_adapter.weight)
        if self.prediction_heads:
            graph_in_dim = 2 * hidden_dim + (2 * num_q_params if self.q_aware_heads else 0)
            node_in_dim = hidden_dim + (num_q_params if self.q_aware_heads else 0)
            self.graph_head = nn.Sequential(
                nn.Linear(graph_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.regressor = nn.Linear(hidden_dim, num_regression_targets)
            # Per-node head: MLP applied independently to each node's hidden vector.
            self.node_head = nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_node_targets),
            )
        else:
            self.graph_head = None
            self.classifier = None
            self.regressor = None
            self.node_head = None
        def _make_param_head(out_dim: int) -> nn.Sequential:
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            nn.init.zeros_(head[-1].bias)
            nn.init.normal_(head[-1].weight, mean=0.0, std=0.01)
            return head

        # Split readout routes physical time scales before q/r are produced.
        # Q order: dD,dM_sg,dm_p_gfm,dw_pf,dm_p_gfl,dw_pll,dpll_zeta.
        if self.direct_trajectory:
            # Architecture screening removes the Q/R parameterisation rather
            # than merely forcing its outputs to zero.  E0 and E2A therefore
            # share exactly the same encoder and direct decoder parameters.
            self.q_head = None
            self.q_slow_head = None
            self.q_fast_head = None
        elif self.haag_split_readout:
            self.q_head = None
            self.q_slow_head = _make_param_head(num_q_params)
            self.q_fast_head = _make_param_head(num_q_params)
            q_slow_mask = torch.zeros(num_q_params, dtype=torch.float32)
            q_fast_mask = torch.zeros(num_q_params, dtype=torch.float32)
            q_slow_mask[[0, 1, 2, 4]] = 1.0
            q_fast_mask[[3, 5, 6]] = 1.0
            self.register_buffer("q_slow_readout_mask", q_slow_mask, persistent=False)
            self.register_buffer("q_fast_readout_mask", q_fast_mask, persistent=False)
        else:
            self.q_head = _make_param_head(num_q_params)
            self.q_slow_head = None
            self.q_fast_head = None
        # Stage 3: r_theta head -- per-node residual-strength scalar. It is
        # disabled by default until the physics residual equation is introduced.
        if self.direct_trajectory:
            self.r_head = None
            self.r_slow_head = None
            self.r_fast_head = None
        elif self.enable_r and self.haag_split_readout:
            self.r_head = None
            self.r_slow_head = _make_param_head(1)
            self.r_fast_head = _make_param_head(2)
        elif self.enable_r:
            # Three per-node residual coefficients:
            # [:,0] sg_swing for SG-only swing residual sin(dtheta)-dtheta;
            # [:,1] fast_gfm for GFM fast power-frequency/current-limit residual;
            # [:,2] fast_gfl for GFL PLL/control fast residual.
            self.r_head = _make_param_head(self.num_r_params)
            self.r_slow_head = None
            self.r_fast_head = None
        else:
            self.r_head = None
            self.r_slow_head = None
            self.r_fast_head = None

        if self.direct_trajectory:
            # Reuse the audited direct-H0/L decoder design, updated only from
            # its obsolete 14 s grid to the sealed 30 s G201 grid.  Device
            # token = host-node embedding + graph mean/max + eleven controller
            # values + three family flags.  E0/E2A share every parameter.
            direct_in = 3 * hidden_dim + 14
            self.direct_device_encoder = nn.Sequential(
                nn.Linear(direct_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.direct_time_encoder = nn.Sequential(
                nn.Linear(11, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
            self.direct_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.normal_(self.direct_decoder[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.direct_decoder[-1].bias)
        else:
            self.direct_device_encoder = None
            self.direct_time_encoder = None
            self.direct_decoder = None

    def set_dyn_warmup_scale(self, value: float) -> None:
        self.dyn_warmup_scale = float(value)

    def forward(self, data, return_cr1_taps: bool = False):
        x = self.node_encoder(data.x)
        h_initial = x
        graph_context = None
        if self.global_context_encoder is not None:
            graph_raw = getattr(data, "graph_level_feat", None)
            if graph_raw is None:
                raise ValueError("global context enabled but graph_level_feat is missing")
            graph_context = self.global_context_encoder(graph_raw)
            graph_gate = self.global_context_gate_max * torch.sigmoid(self.global_context_gate_logit)
            graph_context = graph_gate * graph_context
        haag_debug = None
        if self.graph_mode == "laplacian":
            edge_attr = self.edge_encoder(data.edge_attr)
            for block in self.blocks:
                x = block(x, data.edge_index, edge_attr, data)
        else:
            haag_debug = []
            edge_index_phy = getattr(data, "edge_index_phy", None)
            edge_index_line = getattr(data, "edge_index", None)
            event_node_mask = getattr(data, "event_node_mask", None)
            edge_attr_phy = getattr(data, "edge_attr_phy", None)
            node_type_id = getattr(
                data,
                "node_type_id",
                torch.full((x.size(0),), -1, dtype=torch.long, device=x.device),
            )
            batch = getattr(data, "batch", None)
            x_slow = getattr(data, "x_slow", None)
            x_fast = getattr(data, "x_fast", None)
            if x_slow is not None and x_slow.size(-1) != self.node_slow_dim:
                x_slow = None
            if x_fast is not None and x_fast.size(-1) != self.node_fast_dim:
                x_fast = None
            for block in self.blocks:
                x, dbg = block(
                    x,
                    x_node=data.x,
                    node_type_id=node_type_id,
                    h_initial=h_initial,
                    x_slow=x_slow,
                    x_fast=x_fast,
                    edge_index_phy=edge_index_phy,
                    edge_index_line=edge_index_line,
                    event_node_mask=event_node_mask,
                    edge_attr_phy=edge_attr_phy,
                    batch=batch,
                    mode=self.graph_mode,
                    dyn_scale=self.dyn_warmup_scale,
                    graph_level_feat=graph_context,
                )
                haag_debug.append(dbg)

        batch_index = getattr(
            data,
            "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device),
        )
        if self.system_adapter is not None:
            graph_sizes = torch.bincount(batch_index, minlength=int(batch_index.max().item()) + 1)
            system_id_by_graph = (graph_sizes >= 100).long()
            system_id_by_node = system_id_by_graph[batch_index]
            x = x + self.system_adapter(system_id_by_node).to(x.dtype)

        # Q/R exist only in the legacy recursive architecture.  Direct E0/E2A
        # do not materialise compatibility tensors: they have no correction
        # heads and no correction-related forward work.
        if self.direct_trajectory:
            out = {}
        elif self.haag_split_readout:
            if not haag_debug:
                raise RuntimeError("split readout requires HAAG debug representations")
            h_slow = haag_debug[-1]["_h_slow"]
            h_fast = haag_debug[-1]["_h_fast"]
            q_slow_logit = self.q_slow_head(h_slow)
            q_fast_logit = self.q_fast_head(h_fast)
            q_logit = (
                q_slow_logit * self.q_slow_readout_mask.view(1, -1)
                + q_fast_logit * self.q_fast_readout_mask.view(1, -1)
            )
            q_raw = torch.tanh(q_logit) * self.q_tanh_scale
            q_theta = q_raw * self.q_channel_scales.view(1, -1).to(q_raw.device, q_raw.dtype) * data.q_mask
        else:
            q_logit = self.q_head(x)
            q_raw = torch.tanh(q_logit) * self.q_tanh_scale
            q_theta = q_raw * self.q_channel_scales.view(1, -1).to(q_raw.device, q_raw.dtype) * data.q_mask
        if not self.direct_trajectory:
            if self.enable_r and self.haag_split_readout:
                r_theta = torch.cat([self.r_slow_head(h_slow), self.r_fast_head(h_fast)], dim=-1)
            elif self.r_head is None:
                r_theta = x.new_zeros(x.size(0), self.num_r_params)
            else:
                r_theta = self.r_head(x)
            out = {
                "q_theta": q_theta,
                "q_theta_raw": q_raw,
                "q_logit": q_logit,
                "r_theta": r_theta,
            }
        if self.direct_trajectory:
            out["node_embedding"] = x
            required = (
                "phys_dev_node_row", "phys_gad_dev_mask", "phys_rollout_t",
                "phys_rollout_event_time",
            )
            missing = [name for name in required if not hasattr(data, name)]
            if missing:
                raise RuntimeError(f"direct trajectory fields missing: {missing}")
            rows = data.phys_dev_node_row
            if rows.dim() == 3 and rows.size(1) == 1:
                rows = rows[:, 0]
            dev_mask = data.phys_gad_dev_mask.to(torch.bool)
            if dev_mask.dim() == 3 and dev_mask.size(1) == 1:
                dev_mask = dev_mask[:, 0]
            batch_count, n_dev = rows.shape
            ptr = data.ptr[:-1].view(-1, 1)
            h_dev = x[rows.clamp(min=0) + ptr]
            graph_mean = global_mean_pool(x, data.batch)
            graph_max = global_max_pool(x, data.batch)
            graph_pair = torch.cat([graph_mean, graph_max], dim=-1).unsqueeze(1).expand(-1, n_dev, -1)
            numeric_names = (
                "phys_gad_M", "phys_gad_D", "phys_gad_R", "phys_gad_T1",
                "phys_gad_T2", "phys_gad_T3", "phys_gad_Dt", "phys_gad_m_p",
                "phys_gad_w_pf", "phys_gad_w_pll", "phys_gad_pll_zeta",
            )
            flag_names = ("phys_gad_has_gov", "phys_gad_is_gfm", "phys_gad_is_gfl")
            numeric = torch.stack([getattr(data, name).to(x.dtype) for name in numeric_names], dim=-1)
            numeric = torch.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
            numeric = torch.sign(numeric) * torch.log1p(numeric.abs())
            flags = torch.stack([getattr(data, name).to(x.dtype) for name in flag_names], dim=-1)
            flags = torch.nan_to_num(flags, nan=0.0, posinf=1.0, neginf=0.0)
            token = self.direct_device_encoder(
                torch.cat([h_dev, graph_pair, torch.cat([numeric, flags], dim=-1)], dim=-1)
            )
            times = data.phys_rollout_t
            event_time = data.phys_rollout_event_time.view(-1, 1)
            tau = ((times - event_time).clamp_min(0.0) / 30.0).clamp(max=1.0)
            time_features = [tau, tau.square(), tau.sqrt()]
            for harmonic in (1.0, 2.0, 4.0, 8.0):
                phase = math.pi * harmonic * tau
                time_features.extend([torch.sin(phase), torch.cos(phase)])
            gamma_beta = self.direct_time_encoder(torch.stack(time_features, dim=-1))
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            z = token.unsqueeze(2) * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
            trajectory = self.direct_decoder(z).squeeze(-1)
            # No hard zero anchor: the observable at event time is supervised by its valid target.
            out["direct_trajectory"] = torch.where(
                dev_mask.unsqueeze(-1), trajectory, torch.zeros_like(trajectory)
            )
            out["direct_device_mask"] = dev_mask
            if return_cr1_taps:
                if not hasattr(data, "cr1_terminal_bus_row"):
                    raise RuntimeError("CR1 tap requested but cr1_terminal_bus_row is missing")
                terminal_rows = data.cr1_terminal_bus_row
                if terminal_rows.dim() == 3 and terminal_rows.size(1) == 1:
                    terminal_rows = terminal_rows[:, 0]
                if terminal_rows.shape != rows.shape:
                    raise RuntimeError(
                        f"CR1 terminal/device row shape mismatch: {tuple(terminal_rows.shape)} != {tuple(rows.shape)}"
                    )
                h_terminal = x[terminal_rows.clamp(min=0) + ptr]
                out["cr1_taps"] = {
                    "h_dev": h_dev,
                    "h_terminal": h_terminal,
                }
        if haag_debug is not None:
            out["haag_debug"] = haag_debug
        if self.prediction_heads:
            if self.direct_trajectory and self.q_aware_heads:
                raise RuntimeError("direct trajectory mode forbids q-aware prediction heads")
            # Optional Stage 4 coupling: prediction heads can read q_theta, so q is
            # no longer a side-channel supervised only by auxiliary physics losses.
            node_input = torch.cat([x, q_theta], dim=-1) if self.q_aware_heads else x
            node_pred = self.node_head(node_input)

            graph_mean = global_mean_pool(x, data.batch)
            graph_max = global_max_pool(x, data.batch)
            graph = torch.cat([graph_mean, graph_max], dim=-1)
            if self.q_aware_heads:
                q_mean = global_mean_pool(q_theta, data.batch)
                q_absmax = global_max_pool(q_theta.abs(), data.batch)
                graph = torch.cat([graph, q_mean, q_absmax], dim=-1)
            graph = self.graph_head(graph)
            out.update({
                "class_logits": self.classifier(graph),
                "regression": self.regressor(graph),
                "node_regression": node_pred,
            })
        return out
