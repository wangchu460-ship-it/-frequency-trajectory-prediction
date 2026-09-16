from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class _Entmax15Function(torch.autograd.Function):
    """Numerically stable exact 1.5-entmax with its analytic backward.

    Autograd through the closed-form support threshold contains a square-root
    derivative at zero.  At a sparse-support boundary that derivative is
    infinite and can turn an otherwise finite forward pass into NaN gradients.
    The analytic entmax Jacobian is finite on the active support and avoids
    differentiating through support selection.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int) -> torch.Tensor:
        finite = torch.isfinite(logits)
        safe = logits.masked_fill(~finite, -1e9)
        x = safe / 2.0
        x = x - x.max(dim=dim, keepdim=True).values
        x_sorted, _ = torch.sort(x, dim=dim, descending=True)
        size = x.size(dim)
        rho_shape = [1] * x.dim()
        rho_shape[dim] = size
        rho = torch.arange(
            1, size + 1, device=x.device, dtype=x.dtype
        ).view(rho_shape)
        mean = x_sorted.cpu().cumsum(dim).to(x_sorted.device) / rho
        mean_sq = x_sorted.square().cpu().cumsum(dim).to(x_sorted.device) / rho
        variance = torch.clamp(mean_sq - mean.square(), min=0.0)
        delta = torch.clamp((1.0 - rho * variance) / rho, min=0.0)
        tau = mean - torch.sqrt(delta)
        support = tau <= x_sorted
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau_star = torch.gather(tau, dim, support_size - 1)
        output = torch.clamp(x - tau_star, min=0.0).square()
        output = output * finite.to(dtype=output.dtype)
        output = output / output.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        ctx.dim = int(dim)
        ctx.save_for_backward(output, finite)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, finite = ctx.saved_tensors
        gppr = torch.sqrt(torch.clamp(output, min=0.0))
        grad = grad_output * gppr
        normalizer = gppr.sum(dim=ctx.dim, keepdim=True).clamp_min(1e-12)
        correction = grad.sum(dim=ctx.dim, keepdim=True) / normalizer
        grad = (grad - correction * gppr) * finite.to(dtype=grad.dtype)
        return grad, None


class HAAGLayer(nn.Module):
    """Heterogeneous Apparatus Adaptive Graph layer.

    This layer keeps the v14 structural constraint explicit:
      - A_phy is processed by a separate Physics/Edge-GCN branch -> R_phy.
      - A_dyn is generated as two heterogeneous dynamic views:
            slow-view HetGAT -> R_slow
            fast-view HetGAT -> R_fast
        then fused inside the dynamic channel -> R_dyn.
      - Fusion is only after both branches finish message passing:
            LN(H + gamma_phy * R_phy + gamma_dyn * R_dyn)

    No mixed graph A_phy + lambda*A_dyn is constructed.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim_phy: int,
        node_feature_dim: int | None = None,
        node_feature_dim_slow: int | None = None,
        node_feature_dim_fast: int | None = None,
        node_type_count: int = 3,
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
        phy_gate_init: float = 0.0,
        phy_operator: str = "message",
        use_typepair_bias: bool = True,
        fast_v2: bool = False,
        fast_v21: bool = False,
        split_readout: bool = False,
        unified_dynamic: bool = False,
        compact_unified_dynamic: bool = False,
        dyn_heads: int = 4,
        unified_gate_init: float = 0.04,
        unified_gate_max: float = 0.15,
        dropout: float = 0.1,
        support_topology: str = "shared",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim_phy = int(edge_dim_phy)
        self.node_feature_dim = None if node_feature_dim is None else int(node_feature_dim)
        self.node_feature_dim_slow = self.node_feature_dim if node_feature_dim_slow is None else int(node_feature_dim_slow)
        self.node_feature_dim_fast = self.node_feature_dim if node_feature_dim_fast is None else int(node_feature_dim_fast)
        self.node_type_count = int(node_type_count)
        self.dyn_topk = int(dyn_topk)
        self.dyn_topk_mode = str(dyn_topk_mode).strip().lower()
        self.dyn_topk_min = int(dyn_topk_min)
        self.dyn_topk_max = int(dyn_topk_max)
        self.dyn_topk_scale = float(dyn_topk_scale)
        self.dyn_mass_threshold = float(dyn_mass_threshold)
        self.dyn_relation_mode = str(dyn_relation_mode).strip().lower()
        self.dyn_structured_hops = int(dyn_structured_hops)
        self.dyn_mandatory_weight = float(dyn_mandatory_weight)
        if self.dyn_relation_mode not in {
            "global_topk", "structured_softmax", "structured_entmax", "structured_hybrid",
            "bus_dense_softmax", "bus_dense_entmax15", "all_node_entmax15",
            "hierarchical_bus_device", "hierarchical_bus_device_dd",
        }:
            raise ValueError("unsupported dyn_relation_mode")
        if self.dyn_structured_hops not in {1, 2}:
            raise ValueError("dyn_structured_hops must be 1 or 2")
        if not 0.0 <= self.dyn_mandatory_weight <= 1.0:
            raise ValueError("dyn_mandatory_weight must be in [0, 1]")
        if self.dyn_topk_mode not in {"fixed", "sqrt", "mass_budget", "true_mass", "fixed_ratio"}:
            raise ValueError(
                "dyn_topk_mode must be fixed/sqrt/mass_budget/true_mass/fixed_ratio"
            )
        if self.dyn_topk_min < 1 or self.dyn_topk_max < self.dyn_topk_min:
            raise ValueError("require 1 <= dyn_topk_min <= dyn_topk_max")
        if self.dyn_topk_scale <= 0.0:
            raise ValueError("dyn_topk_scale must be positive")
        if not 0.0 < self.dyn_mass_threshold <= 1.0:
            raise ValueError("dyn_mass_threshold must be in (0, 1]")
        self.phy_operator = str(phy_operator).strip().lower()
        if self.phy_operator not in {"message", "laplacian"}:
            raise ValueError("phy_operator must be 'message' or 'laplacian'")
        self.use_typepair_bias = bool(use_typepair_bias)
        self.fast_v2 = bool(fast_v2)
        self.fast_v21 = bool(fast_v21)
        self.split_readout = bool(split_readout)
        self.unified_dynamic = bool(unified_dynamic)
        self.compact_unified_dynamic = bool(compact_unified_dynamic)
        self.hierarchical_dynamic = self.dyn_relation_mode in {
            "hierarchical_bus_device", "hierarchical_bus_device_dd"
        }
        if self.compact_unified_dynamic and not self.unified_dynamic:
            raise ValueError("compact_unified_dynamic requires unified_dynamic")
        self.dyn_heads = int(dyn_heads)
        if self.hidden_dim % self.dyn_heads != 0:
            raise ValueError("hidden_dim must be divisible by dyn_heads")
        self.dyn_head_dim = self.hidden_dim // self.dyn_heads
        self.support_topology = str(support_topology).strip().lower()
        if self.support_topology not in {"shared", "per_head"}:
            raise ValueError("support_topology must be shared/per_head")
        if sum((self.fast_v2, self.fast_v21, self.split_readout, self.unified_dynamic)) > 1:
            raise ValueError("fast-v2, fast-v21, split-readout, and unified-dynamic are mutually exclusive")

        self.phy_edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_phy, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.phy_msg = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.phy_update = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Device type is allowed; bus ID is not used anywhere in this layer.
        # Slot 0 is passive/unknown, slots 1..node_type_count are SG/GFM/GFL.
        self.type_embedding = nn.Embedding(self.node_type_count + 1, self.hidden_dim)
        if self.node_feature_dim_slow is None and self.node_feature_dim_fast is None:
            dyn_in_dim = 2 * self.hidden_dim
            self.node_feature_proj_slow = None
            self.node_feature_proj_fast = None
        else:
            self.node_feature_proj_slow = nn.Sequential(
                nn.Linear(int(self.node_feature_dim_slow or 0), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
            ) if self.node_feature_dim_slow is not None else None
            self.node_feature_proj_fast = nn.Sequential(
                nn.Linear(int(self.node_feature_dim_fast or 0), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
            ) if self.node_feature_dim_fast is not None else None
            dyn_in_dim = 3 * self.hidden_dim

        self.dyn_slow_source = nn.Linear(dyn_in_dim, self.hidden_dim)
        self.dyn_slow_target = nn.Linear(dyn_in_dim, self.hidden_dim)
        self.dyn_slow_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dyn_slow_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.phy_laplacian_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.dyn_fast_source = nn.Linear(dyn_in_dim, self.hidden_dim)
        self.dyn_fast_target = nn.Linear(dyn_in_dim, self.hidden_dim)
        self.dyn_fast_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dyn_fast_feature_value = (
            nn.Linear(self.hidden_dim, self.hidden_dim) if (self.fast_v2 or self.fast_v21) else None
        )
        # v2.1 starts the newly added fast-value path small; the model can
        # open it only when it improves the rollout objective.
        self.gamma_fast_feature_logit = (
            nn.Parameter(torch.tensor(-2.2)) if self.fast_v21 else None
        )
        self.dyn_fast_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dyn_fuse = None if self.split_readout else nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Unified mode keeps the two feature families separately normalized so
        # neither wins attention merely through units, dimensionality, or
        # variance.  Their semantics are not routed: both projections are
        # concatenated before one shared multi-head relation graph is formed.
        if self.unified_dynamic:
            self.unified_slow_proj = None if self.compact_unified_dynamic else nn.Sequential(
                nn.Linear(self.node_feature_dim_slow, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
            )
            self.unified_fast_proj = None if self.compact_unified_dynamic else nn.Sequential(
                nn.Linear(self.node_feature_dim_fast, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
            )
            unified_in_dim = 4 * self.hidden_dim
            self.dyn_unified_source = nn.Linear(unified_in_dim, self.hidden_dim)
            self.dyn_unified_target = nn.Linear(unified_in_dim, self.hidden_dim)
            self.dyn_unified_value = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
            self.dyn_unified_out = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            if not 0.0 < float(unified_gate_init) < float(unified_gate_max):
                raise ValueError("unified_gate_init must be between 0 and unified_gate_max")
            self.unified_gate_max = float(unified_gate_max)
            gate_fraction = float(unified_gate_init) / float(unified_gate_max)
            gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))
            self.dyn_unified_gate = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.normal_(self.dyn_unified_gate[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.dyn_unified_gate[-1].bias, gate_bias)
            self.typepair_bias_unified = (
                nn.Parameter(torch.zeros(self.node_type_count + 1, self.node_type_count + 1))
                if self.use_typepair_bias else None
            )
            self.gamma_bus_device_logit = (
                nn.Parameter(torch.tensor(0.0)) if self.hierarchical_dynamic else None
            )
            self.gamma_device_device_logit = (
                nn.Parameter(torch.tensor(0.0))
                if self.dyn_relation_mode == "hierarchical_bus_device_dd" else None
            )
        else:
            self.unified_slow_proj = None
            self.unified_fast_proj = None
            self.dyn_unified_source = None
            self.dyn_unified_target = None
            self.dyn_unified_value = None
            self.dyn_unified_out = None
            self.dyn_unified_gate = None
            self.register_parameter("typepair_bias_unified", None)
            self.register_parameter("gamma_bus_device_logit", None)
            self.register_parameter("gamma_device_device_logit", None)
        if self.use_typepair_bias:
            self.typepair_bias_slow = nn.Parameter(torch.zeros(self.node_type_count + 1, self.node_type_count + 1))
            self.typepair_bias_fast = nn.Parameter(torch.zeros(self.node_type_count + 1, self.node_type_count + 1))
        else:
            self.register_parameter("typepair_bias_slow", None)
            self.register_parameter("typepair_bias_fast", None)

        self.gamma_phy_logit = nn.Parameter(torch.tensor(float(phy_gate_init)))
        self.gamma_dyn_logit = nn.Parameter(torch.tensor(float(dyn_gate_init)))
        self.gamma_slow_logit = nn.Parameter(torch.tensor(0.0))
        self.gamma_fast_logit = nn.Parameter(torch.tensor(0.0))
        split_gate_max = 0.15
        split_gate_init = 0.04
        split_gate_logit = math.log(split_gate_init / (split_gate_max - split_gate_init))
        self.split_gate_max = float(split_gate_max)
        self.gamma_slow_out_logit = (
            nn.Parameter(torch.tensor(split_gate_logit)) if self.split_readout else None
        )
        self.gamma_fast_out_logit = (
            nn.Parameter(torch.tensor(split_gate_logit)) if self.split_readout else None
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.split_base_norm = nn.LayerNorm(self.hidden_dim) if self.split_readout else None
        self.split_slow_norm = nn.LayerNorm(self.hidden_dim) if self.split_readout else None
        self.split_fast_norm = nn.LayerNorm(self.hidden_dim) if self.split_readout else None
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _type_slots(node_type_id: torch.Tensor, node_type_count: int) -> torch.Tensor:
        node_type_id = node_type_id.to(dtype=torch.long)
        slots = node_type_id + 1
        slots = torch.where(node_type_id < 0, torch.zeros_like(slots), slots)
        return slots.clamp_(0, node_type_count)

    def _dyn_input(
        self,
        h: torch.Tensor,
        x_node: torch.Tensor | None,
        x_slow: torch.Tensor | None,
        x_fast: torch.Tensor | None,
        node_type_id: torch.Tensor,
        view: str,
    ) -> torch.Tensor:
        slots = self._type_slots(node_type_id, self.node_type_count)
        type_h = self.type_embedding(slots.to(device=h.device))
        parts = [h, type_h]
        node_feature_proj = self.node_feature_proj_fast if view == "fast" else self.node_feature_proj_slow
        x_view = x_fast if view == "fast" else x_slow
        if x_view is None:
            x_view = x_node
        if node_feature_proj is not None:
            if x_view is None:
                node_part = torch.zeros_like(h)
            else:
                node_part = node_feature_proj(x_view.to(dtype=h.dtype, device=h.device))
            parts.append(node_part)
        return torch.cat(parts, dim=-1)

    def _physical_branch(
        self,
        h: torch.Tensor,
        edge_index_phy: torch.Tensor | None,
        edge_attr_phy: torch.Tensor | None,
        node_type_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index_phy is None or edge_attr_phy is None or edge_index_phy.numel() == 0:
            return torch.zeros_like(h)
        src = edge_index_phy[0].to(device=h.device, dtype=torch.long)
        dst = edge_index_phy[1].to(device=h.device, dtype=torch.long)
        valid = (src >= 0) & (src < h.size(0)) & (dst >= 0) & (dst < h.size(0))
        if node_type_id is not None:
            # The F matrix is defined on explicit SG/GFM/GFL apparatus nodes.
            # Never let a malformed mapping turn the physical branch into a
            # bus graph: passive physical buses have node_type_id == -1.
            node_type_id = node_type_id.to(device=h.device, dtype=torch.long)
            safe_src = src.clamp(0, max(h.size(0) - 1, 0))
            safe_dst = dst.clamp(0, max(h.size(0) - 1, 0))
            valid = valid & (node_type_id[safe_src] >= 0) & (node_type_id[safe_dst] >= 0)
        if not torch.any(valid):
            return torch.zeros_like(h)
        src = src[valid]
        dst = dst[valid]
        edge_attr = edge_attr_phy.to(dtype=h.dtype, device=h.device)[valid]
        # Column 3 is the normalized |F_ij| coupling from graph_builders.
        weight = (
            edge_attr[:, 3:4]
            if edge_attr.size(-1) >= 4
            else torch.ones(src.numel(), 1, dtype=h.dtype, device=h.device)
        )
        agg = torch.zeros_like(h)
        active = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        active[src] = True
        active[dst] = True
        if self.phy_operator == "laplacian":
            # Symmetric physical diffusion convention:
            #     ell_i = sum_j w_ij (h_j - h_i) = -(Lh)_i.
            # F_ii is deliberately excluded by graph construction; its self
            # contribution is represented by the degree term in h_i-h_j.
            # Therefore diagonal/self information is not counted twice.
            agg.index_add_(0, dst, weight * (h[src] - h[dst]))
            return self.phy_laplacian_proj(agg) * active.unsqueeze(-1).to(dtype=h.dtype)
        else:
            edge_h = self.phy_edge_encoder(edge_attr)
            msg = self.phy_msg(torch.cat([h[src], edge_h], dim=-1)) * weight
            agg.index_add_(0, dst, msg)
        return self.phy_update(torch.cat([h, agg], dim=-1)) * active.unsqueeze(-1).to(dtype=h.dtype)

    @staticmethod
    def _lift_bus_dynamic_to_devices(
        r_bus: torch.Tensor,
        slots: torch.Tensor,
        edge_index_line: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy each terminal-bus dynamic representation to its apparatus node.

        The dynamic relation itself remains strictly bus-to-bus. Terminal
        attachment edges are used only as a deterministic readout map, never as
        attention candidates. Bidirectional attachment edges are averaged, so
        this is invariant to whether the dataset stores one or both directions.
        """
        bus_mask = slots == 0
        result = r_bus * bus_mask.unsqueeze(-1).to(dtype=r_bus.dtype)
        if edge_index_line is None or edge_index_line.numel() == 0:
            return result, r_bus.new_tensor(0.0)
        line = edge_index_line.to(device=r_bus.device, dtype=torch.long)
        src, dst = line[0], line[1]
        valid = (src >= 0) & (src < r_bus.size(0)) & (dst >= 0) & (dst < r_bus.size(0))
        src, dst = src[valid], dst[valid]
        src_bus_dst_dev = bus_mask[src] & ~bus_mask[dst]
        src_dev_dst_bus = ~bus_mask[src] & bus_mask[dst]
        terminal_bus = torch.cat([src[src_bus_dst_dev], dst[src_dev_dst_bus]])
        apparatus = torch.cat([dst[src_bus_dst_dev], src[src_dev_dst_bus]])
        if apparatus.numel() == 0:
            return result, r_bus.new_tensor(0.0)
        lifted = torch.zeros_like(r_bus)
        counts = r_bus.new_zeros(r_bus.size(0), 1)
        lifted.index_add_(0, apparatus, r_bus[terminal_bus])
        counts.index_add_(0, apparatus, torch.ones(apparatus.numel(), 1, device=r_bus.device, dtype=r_bus.dtype))
        has_terminal = counts.squeeze(-1) > 0
        result[has_terminal] = lifted[has_terminal] / counts[has_terminal].clamp_min(1.0)
        return result, has_terminal.sum().to(dtype=r_bus.dtype)

    @staticmethod
    def _entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Exact alpha=1.5 entmax with stable support-boundary gradients."""
        return _Entmax15Function.apply(logits, dim)

    def _empty_dyn_debug(self, h: torch.Tensor, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_edge_density": h.new_tensor(0.0),
            f"{prefix}_entropy": h.new_tensor(0.0),
            f"{prefix}_typepair_mass": h.new_zeros(self.node_type_count + 1, self.node_type_count + 1),
            f"{prefix}_phy_overlap": h.new_tensor(0.0),
            f"{prefix}_sparse_loss": h.sum() * 0.0,
            f"{prefix}_dist_loss": h.sum() * 0.0,
            f"{prefix}_k_mean": h.new_tensor(0.0),
            f"{prefix}_k_max": h.new_tensor(0.0),
            f"{prefix}_retained_mass": h.new_tensor(0.0),
            f"{prefix}_budget_hit_rate": h.new_tensor(0.0),
            f"{prefix}_k_p50": h.new_tensor(0.0),
            f"{prefix}_k_p90": h.new_tensor(0.0),
            f"{prefix}_k_p95": h.new_tensor(0.0),
            f"{prefix}_candidate_count_mean": h.new_tensor(0.0),
            f"{prefix}_k_fraction_mean": h.new_tensor(0.0),
            f"{prefix}_retained_mass_p10": h.new_tensor(0.0),
            f"{prefix}_retained_mass_min": h.new_tensor(0.0),
            f"{prefix}_mass_achieved_rate": h.new_tensor(0.0),
            "gamma_bus_device": h.new_tensor(0.0),
            "gamma_device_device": h.new_tensor(0.0),
            "R_bus_device_norm": h.new_tensor(0.0),
            "R_device_device_norm": h.new_tensor(0.0),
            "bus_device_edge_count": h.new_tensor(0.0),
            "device_device_edge_count": h.new_tensor(0.0),
        }

    def _dynamic_support_diagnostics(
        self,
        selected_k: torch.Tensor,
        retained_mass: torch.Tensor,
        candidate_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Summarise actual support, not merely its configured upper bound."""
        k = selected_k.detach().float()
        mass = retained_mass.detach().float()
        candidates = candidate_count.detach().float().clamp_min(1.0)
        return {
            "dyn_k_p50": torch.quantile(k, 0.50),
            "dyn_k_p90": torch.quantile(k, 0.90),
            "dyn_k_p95": torch.quantile(k, 0.95),
            "dyn_candidate_count_mean": candidates.mean(),
            "dyn_k_fraction_mean": (k / candidates).mean(),
            "dyn_retained_mass_p10": torch.quantile(mass, 0.10),
            "dyn_retained_mass_min": mass.min(),
            "dyn_mass_achieved_rate": (
                mass >= (float(self.dyn_mass_threshold) - 1e-6)
            ).float().mean(),
        }

    def _dynamic_budget(self, n_candidates: int) -> int:
        """Return the per-target edge budget for the current graph.

        ``fixed`` preserves the historical meaning of ``dyn_topk``.  The
        ``sqrt`` and ``mass_budget`` use a bounded square-root budget.
        ``true_mass`` deliberately exposes the complete candidate set so the
        requested retained mass, rather than a hidden K ceiling, determines
        support size. ``fixed_ratio`` uses ceil(scale * candidate_count).
        """
        n_candidates = max(int(n_candidates), 0)
        if n_candidates <= 0:
            return 0
        if self.dyn_topk_mode == "fixed":
            return min(max(self.dyn_topk, 0), n_candidates)
        if self.dyn_topk_mode == "true_mass":
            return n_candidates
        if self.dyn_topk_mode == "fixed_ratio":
            return min(max(int(math.ceil(self.dyn_topk_scale * n_candidates)), 1), n_candidates)
        scaled = int(math.ceil(self.dyn_topk_scale * math.sqrt(float(n_candidates))))
        return min(max(scaled, self.dyn_topk_min), self.dyn_topk_max, n_candidates)

    def _adaptive_mass_reduce(
        self,
        alpha_h: torch.Tensor,
        values: torch.Tensor,
        k_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce a globally eligible relation by target-specific attention mass.

        All nodes are scored first.  The actual message support is the smallest
        ranked prefix reaching ``dyn_mass_threshold``, subject to a sqrt(N)
        safety budget for ``mass_budget`` or the full candidate set for
        ``true_mass``. Thus K is learned independently for every target.
        """
        importance = alpha_h.mean(dim=0)
        selected_mass, idx = torch.topk(importance, k=k_budget, dim=1)
        cumulative_mass = selected_mass.cumsum(dim=1)
        if self.dyn_topk_mode == "fixed_ratio":
            k_per_target = torch.full(
                (selected_mass.size(0),), k_budget, dtype=torch.long,
                device=selected_mass.device,
            )
        else:
            k_per_target = (cumulative_mass < self.dyn_mass_threshold).sum(dim=1) + 1
            k_per_target = k_per_target.clamp(min=min(self.dyn_topk_min, k_budget), max=k_budget)
        rank = torch.arange(k_budget, device=alpha_h.device).view(1, -1)
        keep = rank < k_per_target.view(-1, 1)
        selected_alpha = torch.gather(
            alpha_h, 2, idx.unsqueeze(0).expand(alpha_h.size(0), -1, -1)
        )
        selected_alpha = selected_alpha * keep.unsqueeze(0).to(dtype=selected_alpha.dtype)
        selected_alpha = selected_alpha / selected_alpha.sum(dim=2, keepdim=True).clamp_min(1e-12)
        selected_values = values[idx]
        aggregate = (
            selected_values * selected_alpha.permute(1, 2, 0).unsqueeze(-1)
        ).sum(dim=1)
        retained_mass = (selected_mass * keep).sum(dim=1)
        budget_hit = (k_per_target >= k_budget).to(dtype=importance.dtype)
        return aggregate, idx, keep, k_per_target, retained_mass, budget_hit

    def _per_head_size_cap_reduce(
        self,
        alpha_h: torch.Tensor,
        values: torch.Tensor,
        k_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """P5-4 H1: independent fixed-size support for every head and target."""
        selected_mass, idx = torch.topk(alpha_h, k=k_budget, dim=2)
        selected_alpha = selected_mass / selected_mass.sum(dim=2, keepdim=True).clamp_min(1e-12)
        values_h = values[idx]  # [head, target, K, head, dim]
        head_index = torch.arange(alpha_h.size(0), device=alpha_h.device).view(-1, 1, 1)
        target_index = torch.arange(alpha_h.size(1), device=alpha_h.device).view(1, -1, 1)
        rank_index = torch.arange(k_budget, device=alpha_h.device).view(1, 1, -1)
        selected_values = values_h[head_index, target_index, rank_index, head_index]
        aggregate = (selected_values * selected_alpha.unsqueeze(-1)).sum(dim=2).permute(1, 0, 2)
        keep = torch.ones_like(idx, dtype=torch.bool)
        k_per_target = torch.full((alpha_h.size(0), alpha_h.size(1)), k_budget, dtype=torch.long, device=alpha_h.device)
        retained_mass = selected_mass.sum(dim=2)
        budget_hit = torch.ones_like(retained_mass)
        return aggregate, idx, keep, k_per_target, retained_mass, budget_hit

    def _hierarchical_local_reduce(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        graph_nodes: torch.Tensor,
        slots: torch.Tensor,
        edge_index_line: torch.Tensor | None,
        include_device_device: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Local, physically anchored apparatus relations for the hierarchy.

        A device communicates only with its terminal bus and that bus's direct
        bus neighbours. Optional device-device edges require a shared terminal
        bus. These candidates are deliberately small and are all normalized;
        they never compete with the global bus retained-mass pool.
        """
        bd_agg = torch.zeros_like(v)
        dd_agg = torch.zeros_like(v)
        bd_typepair = v.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
        dd_typepair = v.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
        zero_count = v.new_tensor(0.0)
        if edge_index_line is None or edge_index_line.numel() == 0:
            return bd_agg, dd_agg, bd_typepair, dd_typepair, zero_count, zero_count

        n = int(graph_nodes.numel())
        local_of_global = torch.full(
            (v.size(0),), -1, dtype=torch.long, device=v.device
        )
        local_of_global[graph_nodes] = torch.arange(n, device=v.device)
        line = edge_index_line.to(device=v.device, dtype=torch.long)
        src_local = local_of_global[line[0]]
        dst_local = local_of_global[line[1]]
        valid = (src_local >= 0) & (dst_local >= 0)
        adjacency = torch.zeros((n, n), dtype=torch.bool, device=v.device)
        adjacency[src_local[valid], dst_local[valid]] = True
        adjacency[dst_local[valid], src_local[valid]] = True
        local_slots = slots[graph_nodes]
        bus_idx = torch.nonzero(local_slots == 0, as_tuple=False).view(-1)
        dev_idx = torch.nonzero(local_slots > 0, as_tuple=False).view(-1)
        if bus_idx.numel() == 0 or dev_idx.numel() == 0:
            return bd_agg, dd_agg, bd_typepair, dd_typepair, zero_count, zero_count

        attach = adjacency[bus_idx][:, dev_idx]
        bus_adj = adjacency[bus_idx][:, bus_idx]
        bus_neighbourhood = bus_adj | torch.eye(
            bus_idx.numel(), dtype=torch.bool, device=v.device
        )
        reachable = (
            bus_neighbourhood.to(torch.float32) @ attach.to(torch.float32)
        ) > 0
        candidate_bd = torch.zeros((n, n), dtype=torch.bool, device=v.device)
        candidate_bd[bus_idx[:, None], dev_idx[None, :]] = reachable
        candidate_bd[dev_idx[:, None], bus_idx[None, :]] = reachable.t()

        candidate_dd = torch.zeros_like(candidate_bd)
        if include_device_device:
            shared_terminal = (
                attach.t().to(torch.float32) @ attach.to(torch.float32)
            ) > 0
            shared_terminal.fill_diagonal_(False)
            candidate_dd[dev_idx[:, None], dev_idx[None, :]] = shared_terminal

        def reduce_candidate(
            candidate: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            update = torch.zeros_like(v)
            typepair = v.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
            # Vectorize all local targets and sources in one GPU operation.
            # The former target loop invoked entmax once per node and called
            # .item() inside that loop, forcing many device-host syncs.
            q_g = q[graph_nodes]
            k_g = k[graph_nodes]
            v_g = v[graph_nodes]
            logits = torch.einsum("ihd,jhd->hij", q_g, k_g)
            logits = logits / math.sqrt(float(self.dyn_head_dim))
            if self.typepair_bias_unified is not None:
                logits = logits + self.typepair_bias_unified[
                    local_slots[:, None], local_slots[None, :]
                ].unsqueeze(0)
            logits = logits.masked_fill(~candidate.unsqueeze(0), float("-inf"))
            alpha_h = self._entmax15(logits.float(), dim=2).to(dtype=v.dtype)
            update[graph_nodes] = torch.einsum("hij,jhd->ihd", alpha_h, v_g)

            alpha = alpha_h.mean(dim=0)
            active = candidate & (alpha > 1e-8)
            dst_types = local_slots[:, None].expand(n, n)[active]
            src_types = local_slots[None, :].expand(n, n)[active]
            typepair.index_put_(
                (dst_types, src_types), alpha[active].detach(), accumulate=True
            )
            edge_count = active.sum().to(dtype=v.dtype)
            return update, typepair, edge_count

        bd_agg, bd_typepair, bd_edges = reduce_candidate(candidate_bd)
        dd_agg, dd_typepair, dd_edges = reduce_candidate(candidate_dd)
        return bd_agg, dd_agg, bd_typepair, dd_typepair, bd_edges, dd_edges

    def _dynamic_subbranch(
        self,
        h: torch.Tensor,
        x_node: torch.Tensor | None,
        x_slow: torch.Tensor | None,
        x_fast: torch.Tensor | None,
        node_type_id: torch.Tensor,
        batch: torch.Tensor | None,
        edge_index_phy: torch.Tensor | None,
        view: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        else:
            batch = batch.to(device=h.device, dtype=torch.long)
        dyn_in = self._dyn_input(h, x_node, x_slow, x_fast, node_type_id, view=view)
        if view == "fast":
            q = self.dyn_fast_target(dyn_in)
            k = self.dyn_fast_source(dyn_in)
            v = self.dyn_fast_value(h)
            fast_feature_gate = h.new_tensor(0.0)
            if self.dyn_fast_feature_value is not None and self.node_feature_proj_fast is not None:
                x_view = x_fast if x_fast is not None else x_node
                if x_view is not None:
                    fast_h = self.node_feature_proj_fast(x_view.to(dtype=h.dtype, device=h.device))
                    fast_feature_gate = (
                        torch.sigmoid(self.gamma_fast_feature_logit)
                        if self.gamma_fast_feature_logit is not None else h.new_tensor(1.0)
                    )
                    v = v + fast_feature_gate * self.dyn_fast_feature_value(fast_h)
            out_layer = self.dyn_fast_out
            typepair_bias = self.typepair_bias_fast
            prefix = "dyn_fast"
        else:
            q = self.dyn_slow_target(dyn_in)
            k = self.dyn_slow_source(dyn_in)
            v = self.dyn_slow_value(h)
            out_layer = self.dyn_slow_out
            typepair_bias = self.typepair_bias_slow
            prefix = "dyn_slow"
            fast_feature_gate = h.new_tensor(0.0)
        slots = self._type_slots(node_type_id.to(device=h.device), self.node_type_count)

        src_all: list[torch.Tensor] = []
        dst_all: list[torch.Tensor] = []
        alpha_all: list[torch.Tensor] = []
        type_src_all: list[torch.Tensor] = []
        type_dst_all: list[torch.Tensor] = []
        for graph_id in torch.unique(batch, sorted=True):
            nodes = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
            n = int(nodes.numel())
            if n <= 1:
                continue
            q_g = q[nodes]
            k_g = k[nodes]
            score = (q_g @ k_g.t()) / math.sqrt(float(self.hidden_dim))
            score.fill_diagonal_(float("-inf"))
            if typepair_bias is not None:
                t = slots[nodes]
                score = score + typepair_bias[t[:, None], t[None, :]]
            if view == "fast" and self.fast_v21:
                # Keep fast relations heterogeneous without discarding the
                # passive/event relays that carry contingency location. Each
                # device target receives same-family and cross-device slots,
                # then two passive relay slots; remaining slots use score.
                local_slots = slots[nodes]
                k_total = min(max(self.dyn_topk, 0), n - 1)
                if k_total <= 0:
                    continue
                for local_dst in range(n):
                    used = torch.zeros(n, dtype=torch.bool, device=h.device)
                    used[local_dst] = True
                    chosen: list[torch.Tensor] = []

                    def take(mask: torch.Tensor, count: int) -> None:
                        nonlocal used
                        if count <= 0:
                            return
                        eligible = mask & (~used)
                        n_eligible = int(eligible.sum().item())
                        if n_eligible <= 0:
                            return
                        n_take = min(count, n_eligible)
                        masked = score[local_dst].masked_fill(~eligible, float("-inf"))
                        _, picked = torch.topk(masked, k=n_take)
                        chosen.append(picked)
                        used[picked] = True

                    dst_slot = int(local_slots[local_dst].item())
                    device_mask = local_slots > 0
                    passive_mask = local_slots == 0
                    if dst_slot > 0:
                        take(device_mask & (local_slots == dst_slot), 2)
                        if dst_slot in (2, 3):
                            other_converter = device_mask & (local_slots >= 2) & (local_slots != dst_slot)
                            take(other_converter, 1)
                        take(device_mask, 4 - int(used.sum().item() - 1))
                    else:
                        # Passive relay nodes retain one candidate from each
                        # apparatus family where available.
                        for family_slot in (1, 2, 3):
                            take(local_slots == family_slot, 1)
                        take(device_mask, 4 - int(used.sum().item() - 1))
                    take(passive_mask, 2)
                    take(torch.ones(n, dtype=torch.bool, device=h.device), k_total - int(used.sum().item() - 1))
                    if not chosen:
                        continue
                    picked = torch.cat(chosen)[:k_total]
                    vals = score[local_dst, picked]
                    alpha = torch.softmax(vals, dim=0)
                    dst = nodes.new_full((picked.numel(),), int(nodes[local_dst].item()))
                    src = nodes[picked]
                    src_all.append(src)
                    dst_all.append(dst)
                    alpha_all.append(alpha)
                    type_src_all.append(slots[src])
                    type_dst_all.append(slots[dst])
                continue
            if view == "fast" and self.fast_v2:
                # Fast converter relations should be device-centric. Passive
                # buses remain destinations/relays, but cannot consume all
                # top-k source slots merely because they are more numerous.
                device_source = slots[nodes] > 0
                n_device = int(device_source.sum().item())
                if n_device > 1:
                    score[:, ~device_source] = float("-inf")
                    k_keep = min(max(self.dyn_topk, 0), n_device - 1)
                else:
                    k_keep = min(max(self.dyn_topk, 0), n - 1)
            else:
                k_keep = min(max(self.dyn_topk, 0), n - 1)
            if k_keep <= 0:
                continue
            vals, idx = torch.topk(score, k=k_keep, dim=1)
            alpha = torch.softmax(vals, dim=1)
            dst = nodes[:, None].expand(-1, k_keep).reshape(-1)
            src = nodes[idx.reshape(-1)]
            src_all.append(src)
            dst_all.append(dst)
            alpha_all.append(alpha.reshape(-1))
            type_src_all.append(slots[src])
            type_dst_all.append(slots[dst])

        if not src_all:
            return torch.zeros_like(h), self._empty_dyn_debug(h, prefix)

        src = torch.cat(src_all)
        dst = torch.cat(dst_all)
        alpha = torch.cat(alpha_all).to(dtype=h.dtype)
        msg = alpha.view(-1, 1) * v[src]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        r_dyn = out_layer(agg)

        n_graph_edges = 0
        for graph_id in torch.unique(batch, sorted=True):
            n = int(torch.sum(batch == graph_id).item())
            n_graph_edges += n * max(n - 1, 0)
        density = float(src.numel()) / max(float(n_graph_edges), 1.0)
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-12))).sum() / max(float(alpha.numel()), 1.0)
        typepair = h.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
        t_src = torch.cat(type_src_all)
        t_dst = torch.cat(type_dst_all)
        typepair.index_put_((t_dst, t_src), alpha.detach(), accumulate=True)
        typepair = typepair / typepair.sum().clamp_min(1e-12)
        overlap = h.new_tensor(0.0)
        if edge_index_phy is not None and edge_index_phy.numel() > 0:
            dyn_pairs = set(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()))
            phy_src = edge_index_phy[0].detach().cpu().tolist()
            phy_dst = edge_index_phy[1].detach().cpu().tolist()
            phy_pairs = set(zip(phy_src, phy_dst))
            if dyn_pairs:
                overlap = h.new_tensor(len(dyn_pairs & phy_pairs) / max(len(dyn_pairs), 1))
        debug = {
            f"{prefix}_edge_density": h.new_tensor(density),
            f"{prefix}_entropy": entropy.detach(),
            f"{prefix}_typepair_mass": typepair,
            f"{prefix}_phy_overlap": overlap,
            f"{prefix}_sparse_loss": alpha.abs().mean(),
            # Electrical distance is reserved for future data schemas. Current
            # IEEE39 samples do not expose a device-device distance matrix, so
            # do not invent one here.
            f"{prefix}_dist_loss": alpha.sum() * 0.0,
        }
        if view == "fast":
            debug["gamma_fast_feature"] = fast_feature_gate.detach()
        return r_dyn, debug

    def _unified_dynamic_branch(
        self,
        h: torch.Tensor,
        h_initial: torch.Tensor | None,
        x_slow: torch.Tensor | None,
        x_fast: torch.Tensor | None,
        node_type_id: torch.Tensor,
        batch: torch.Tensor | None,
        edge_index_phy: torch.Tensor | None,
        edge_index_line: torch.Tensor | None,
        event_node_mask: torch.Tensor | None,
        graph_level_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build one shared top-k relation graph with unlabeled attention heads."""
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        else:
            batch = batch.to(device=h.device, dtype=torch.long)
        slots = self._type_slots(node_type_id.to(device=h.device), self.node_type_count)
        type_h = self.type_embedding(slots)
        if self.compact_unified_dynamic:
            h0 = h if h_initial is None else h_initial.to(dtype=h.dtype, device=h.device)
            if h0.shape != h.shape:
                raise ValueError("h_initial must have the same shape as h")
            if graph_level_feat is not None:
                context = graph_level_feat.to(dtype=h.dtype, device=h.device)
                graph_context = context if context.size(0) == h.size(0) else context[batch]
            else:
                graph_context = torch.zeros_like(h0)
                for graph_id in torch.unique(batch, sorted=True):
                    nodes = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
                    if nodes.numel() == 0:
                        continue
                    context = 0.5 * (h0[nodes].mean(dim=0) + h0[nodes].amax(dim=0))
                    graph_context[nodes] = context
            relation_input = torch.cat([h, h0, type_h, graph_context], dim=-1)
            value_input = torch.cat([h, h0, graph_context], dim=-1)
        else:
            if x_slow is None:
                slow_h = torch.zeros_like(h)
            else:
                slow_h = self.unified_slow_proj(x_slow.to(dtype=h.dtype, device=h.device))
            if x_fast is None:
                fast_h = torch.zeros_like(h)
            else:
                fast_h = self.unified_fast_proj(x_fast.to(dtype=h.dtype, device=h.device))
            relation_input = torch.cat([h, type_h, slow_h, fast_h], dim=-1)
            value_input = torch.cat([h, slow_h, fast_h], dim=-1)
        q = self.dyn_unified_target(relation_input).view(-1, self.dyn_heads, self.dyn_head_dim)
        k = self.dyn_unified_source(relation_input).view(-1, self.dyn_heads, self.dyn_head_dim)
        v = self.dyn_unified_value(value_input).view(
            -1, self.dyn_heads, self.dyn_head_dim
        )
        agg = torch.zeros_like(v)
        src_all: list[torch.Tensor] = []
        dst_all: list[torch.Tensor] = []
        alpha_all: list[torch.Tensor] = []
        type_src_all: list[torch.Tensor] = []
        type_dst_all: list[torch.Tensor] = []
        retained_mass_all: list[torch.Tensor] = []
        k_all: list[torch.Tensor] = []
        budget_hit_all: list[torch.Tensor] = []
        candidate_count_all: list[torch.Tensor] = []
        dense_entropy_terms: list[torch.Tensor] = []
        dense_alpha_counts = 0
        hierarchical = self.hierarchical_dynamic
        bus_dense = self.dyn_relation_mode in {
            "bus_dense_softmax", "bus_dense_entmax15",
            "hierarchical_bus_device", "hierarchical_bus_device_dd",
        }
        bd_agg_all = torch.zeros_like(v)
        dd_agg_all = torch.zeros_like(v)
        bd_typepair_all = h.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
        dd_typepair_all = h.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
        bd_edge_count = h.new_tensor(0.0)
        dd_edge_count = h.new_tensor(0.0)
        bus_target_count = h.new_tensor(0.0)
        p5_4_support_records: list[torch.Tensor] = []
        p5_4_detailed_records: list[dict[str, torch.Tensor]] = []
        for graph_id in torch.unique(batch, sorted=True):
            graph_nodes = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
            # v15.10 channel contract: the learned dynamic relation is formed
            # only among physical buses. Explicit apparatus nodes receive the
            # terminal-bus result later through a deterministic lift.
            nodes = graph_nodes[slots[graph_nodes] == 0] if bus_dense else graph_nodes
            n = int(nodes.numel())
            k_budget = (
                self._dynamic_budget(n - 1)
                if self.dyn_topk_mode in {"mass_budget", "true_mass", "fixed_ratio"}
                else ((n - 1) if bus_dense or self.dyn_relation_mode == "all_node_entmax15"
                      else self._dynamic_budget(n - 1))
            )
            if k_budget <= 0:
                continue
            candidate_count_all.append(
                torch.full((n,), n - 1, dtype=h.dtype, device=h.device)
            )
            q_g, k_g, v_g = q[nodes], k[nodes], v[nodes]
            # [head, target, source].  Heads share the topology selected from
            # their mean score, while retaining independent attention weights.
            score = torch.einsum("ihd,jhd->hij", q_g, k_g) / math.sqrt(float(self.dyn_head_dim))
            diagonal = torch.arange(n, device=h.device)
            score[:, diagonal, diagonal] = float("-inf")
            if self.typepair_bias_unified is not None:
                t = slots[nodes]
                score = score + self.typepair_bias_unified[t[:, None], t[None, :]].unsqueeze(0)
            if bus_dense:
                if self.dyn_relation_mode == "bus_dense_softmax":
                    alpha_h = torch.softmax(score, dim=2)
                else:
                    alpha_h = self._entmax15(score.float(), dim=2).to(dtype=score.dtype)
                if self.dyn_topk_mode in {"mass_budget", "true_mass", "fixed_ratio"}:
                    if self.support_topology == "per_head":
                        reduced, _idx, keep, k_per_target_h, retained_mass_h, budget_hit_h = (
                            self._per_head_size_cap_reduce(alpha_h, v_g, k_budget)
                        )
                        k_per_target = k_per_target_h.to(dtype=h.dtype).mean(dim=0)
                        retained_mass = retained_mass_h.mean(dim=0)
                        budget_hit = budget_hit_h.mean(dim=0)
                        alpha_kept = torch.gather(alpha_h, 2, _idx)[keep]
                    else:
                        reduced, _idx, keep, k_per_target, retained_mass, budget_hit = (
                            self._adaptive_mass_reduce(alpha_h, v_g, k_budget)
                        )
                        alpha_kept = torch.gather(alpha_h.mean(dim=0), 1, _idx)[keep]
                    agg[nodes] = reduced.to(dtype=agg.dtype)
                    k_all.append(k_per_target.to(dtype=h.dtype))
                    retained_mass_all.append(retained_mass)
                    budget_hit_all.append(budget_hit.to(dtype=h.dtype))
                    p5_4_support_records.append(_idx.detach().cpu())
                    if self.support_topology == "per_head":
                        p5_4_detailed_records.append({
                            "graph_id": graph_id.detach().cpu(),
                            "node_ids": nodes.detach().cpu(),
                            "support_idx": _idx.detach().cpu(),
                            "keep": keep.detach().cpu(),
                            "alpha_h": alpha_h.detach().cpu(),
                            "k_per_target_h": k_per_target_h.detach().cpu(),
                            "retained_mass_h": retained_mass_h.detach().cpu(),
                            "budget_hit_h": budget_hit_h.detach().cpu(),
                        })
                else:
                    agg[nodes] = torch.einsum("hij,jhd->ihd", alpha_h, v_g).to(dtype=agg.dtype)
                    alpha_mean = alpha_h.mean(dim=0)
                    keep = alpha_mean > 1e-8
                    k_per_target = keep.sum(dim=1).to(dtype=h.dtype)
                    k_all.append(k_per_target)
                    retained_mass_all.append(torch.ones(n, dtype=h.dtype, device=h.device))
                    budget_hit_all.append((k_per_target >= (n - 1)).to(dtype=h.dtype))
                    alpha_kept = alpha_mean[keep]
                dense_entropy_terms.append(
                    -(alpha_kept * torch.log(alpha_kept.clamp_min(1e-12))).sum().detach()
                )
                dense_alpha_counts += int(alpha_kept.numel())
                if hierarchical:
                    bd_update, dd_update, bd_tp, dd_tp, bd_edges, dd_edges = (
                        self._hierarchical_local_reduce(
                            q, k, v, graph_nodes, slots, edge_index_line,
                            include_device_device=(
                                self.dyn_relation_mode == "hierarchical_bus_device_dd"
                            ),
                        )
                    )
                    bd_agg_all = bd_agg_all + bd_update
                    dd_agg_all = dd_agg_all + dd_update
                    bd_typepair_all = bd_typepair_all + bd_tp
                    dd_typepair_all = dd_typepair_all + dd_tp
                    bd_edge_count = bd_edge_count + bd_edges
                    dd_edge_count = dd_edge_count + dd_edges
                    bus_target_count = bus_target_count + n
                continue
            # Sparse modes need full probabilities to measure retained mass.
            # The bus-dense branch above already uses its normalized relation
            # directly, so computing an additional dense softmax there was a
            # pure diagnostic duplicate.
            with torch.no_grad():
                full_prob_h = torch.softmax(score.float(), dim=2)
                importance = full_prob_h.mean(dim=0)
            if self.dyn_relation_mode != "global_topk":
                # Candidate set: original-line neighbourhood (one or two hops),
                # every explicit SG/GFM/GFL apparatus node, and the event bus
                # plus its direct line neighbours.  This operates on the real
                # augmented graph (39+devices / 140+devices), not system labels.
                adjacency = torch.zeros((n, n), dtype=torch.bool, device=h.device)
                if edge_index_line is not None and edge_index_line.numel() > 0:
                    local_of_global = torch.full(
                        (h.size(0),), -1, dtype=torch.long, device=h.device
                    )
                    local_of_global[nodes] = torch.arange(n, device=h.device)
                    line = edge_index_line.to(device=h.device, dtype=torch.long)
                    line_src = local_of_global[line[0]]
                    line_dst = local_of_global[line[1]]
                    valid = (line_src >= 0) & (line_dst >= 0)
                    adjacency[line_dst[valid], line_src[valid]] = True
                    adjacency[line_src[valid], line_dst[valid]] = True
                local_relation = adjacency
                if self.dyn_structured_hops == 2:
                    local_relation = local_relation | (
                        adjacency.to(dtype=torch.float32) @ adjacency.to(dtype=torch.float32) > 0
                    )
                event_local = torch.zeros(n, dtype=torch.bool, device=h.device)
                if event_node_mask is not None and event_node_mask.numel() == h.size(0):
                    event_local = event_node_mask.to(device=h.device, dtype=torch.bool)[nodes]
                event_plus = event_local | (adjacency[:, event_local].any(dim=1) if event_local.any() else event_local)
                device_source = slots[nodes] > 0
                if self.dyn_relation_mode == "all_node_entmax15":
                    # Paper-facing adaptive relation: every physical bus and
                    # every explicit SG/GFM/GFL apparatus node is eligible.
                    # Entmax chooses a target-specific sparse support, so the
                    # effective neighbour count is learned rather than fixed K.
                    candidate = torch.ones((n, n), dtype=torch.bool, device=h.device)
                    candidate.fill_diagonal_(False)
                    mandatory = torch.zeros_like(candidate)
                else:
                    candidate = local_relation | device_source.view(1, -1) | event_plus.view(1, -1)
                    candidate.fill_diagonal_(False)
                    mandatory = adjacency | event_plus.view(1, -1)
                    mandatory.fill_diagonal_(False)
                # A connected grid should always provide a candidate.  Keep a
                # deterministic fallback for malformed/isolated nodes.
                empty = ~candidate.any(dim=1)
                if empty.any():
                    fallback = torch.ones((n, n), dtype=torch.bool, device=h.device)
                    fallback.fill_diagonal_(False)
                    candidate[empty] = fallback[empty]

                masked_score = score.masked_fill(~candidate.unsqueeze(0), float("-inf"))
                if self.dyn_relation_mode == "structured_softmax":
                    alpha_h = torch.softmax(masked_score, dim=2)
                else:
                    alpha_h = self._entmax15(masked_score.float(), dim=2).to(dtype=score.dtype)
                    if self.dyn_relation_mode == "structured_hybrid" and self.dyn_mandatory_weight > 0.0:
                        mandatory_score = score.masked_fill(~mandatory.unsqueeze(0), float("-inf"))
                        mandatory_alpha = torch.softmax(mandatory_score, dim=2)
                        has_mandatory = mandatory.any(dim=1).view(1, n, 1)
                        mandatory_alpha = torch.where(
                            has_mandatory, mandatory_alpha, torch.zeros_like(mandatory_alpha)
                        )
                        w = self.dyn_mandatory_weight * has_mandatory.to(dtype=alpha_h.dtype)
                        alpha_h = (1.0 - w) * alpha_h + w * mandatory_alpha

                alpha_mean = alpha_h.mean(dim=0)
                if self.dyn_topk_mode in {"mass_budget", "true_mass", "fixed_ratio"}:
                    reduced, idx, keep, k_per_target, retained_mass, budget_hit = (
                        self._adaptive_mass_reduce(alpha_h, v_g, k_budget)
                    )
                    agg[nodes] = reduced.to(dtype=agg.dtype)
                    dst_grid = nodes[:, None].expand(-1, k_budget)
                    src = nodes[idx[keep]]
                    dst = dst_grid[keep]
                    alpha = torch.gather(alpha_mean, 1, idx)[keep]
                    retained_mass_all.append(retained_mass)
                    k_all.append(k_per_target.to(dtype=h.dtype))
                    budget_hit_all.append(budget_hit.to(dtype=h.dtype))
                else:
                    agg[nodes] = torch.einsum("hij,jhd->ihd", alpha_h, v_g).to(dtype=agg.dtype)
                    keep = alpha_mean > 1e-8
                    dst_grid = nodes[:, None].expand(-1, n)
                    src_grid = nodes[None, :].expand(n, -1)
                    src = src_grid[keep]
                    dst = dst_grid[keep]
                    alpha = alpha_mean[keep]
                    retained_mass_all.append((importance * candidate).sum(dim=1))
                    k_per_target = keep.sum(dim=1).to(dtype=h.dtype)
                    k_all.append(k_per_target)
                    budget_hit_all.append(torch.zeros_like(k_per_target))
                src_all.append(src)
                dst_all.append(dst)
                alpha_all.append(alpha)
                type_src_all.append(slots[src])
                type_dst_all.append(slots[dst])
                continue
            selection_score = (
                importance
                if self.dyn_topk_mode in {"mass_budget", "true_mass", "fixed_ratio"}
                else score.mean(dim=0)
            )
            _, idx = torch.topk(selection_score, k=k_budget, dim=1)
            selected_mass = torch.gather(importance, 1, idx)
            if self.dyn_topk_mode in {"mass_budget", "true_mass", "fixed_ratio"}:
                cumulative_mass = selected_mass.cumsum(dim=1)
                k_per_target = (cumulative_mass < self.dyn_mass_threshold).sum(dim=1) + 1
                k_per_target = k_per_target.clamp(min=min(self.dyn_topk_min, k_budget), max=k_budget)
            else:
                k_per_target = torch.full(
                    (n,), k_budget, dtype=torch.long, device=h.device
                )
            rank = torch.arange(k_budget, device=h.device).view(1, -1)
            keep = rank < k_per_target.view(-1, 1)
            selected = torch.gather(
                score,
                2,
                idx.unsqueeze(0).expand(self.dyn_heads, -1, -1),
            )
            selected = selected.masked_fill(~keep.unsqueeze(0), float("-inf"))
            alpha_h = torch.softmax(selected, dim=2)
            values = v_g[idx]  # [target, budget, head, head_dim]
            # AMP compatibility: a float32 type-pair bias promotes the
            # attention weights while v remains fp16. Cast the reduction back
            # to the destination dtype before indexed assignment.
            agg[nodes] = (
                values * alpha_h.permute(1, 2, 0).unsqueeze(-1)
            ).sum(dim=1).to(dtype=agg.dtype)

            dst_grid = nodes[:, None].expand(-1, k_budget)
            src = nodes[idx[keep]]
            dst = dst_grid[keep]
            alpha = alpha_h.mean(dim=0)[keep]
            src_all.append(src)
            dst_all.append(dst)
            alpha_all.append(alpha)
            type_src_all.append(slots[src])
            type_dst_all.append(slots[dst])
            retained_mass_all.append((selected_mass * keep).sum(dim=1))
            k_all.append(k_per_target.to(dtype=h.dtype))
            budget_hit_all.append((k_per_target >= k_budget).to(dtype=h.dtype))

        if bus_dense:
            gamma_bd = (
                torch.sigmoid(self.gamma_bus_device_logit)
                if hierarchical else h.new_tensor(0.0)
            )
            gamma_dd = (
                torch.sigmoid(self.gamma_device_device_logit)
                if hierarchical and self.gamma_device_device_logit is not None
                else h.new_tensor(0.0)
            )
            combined_agg = agg + gamma_bd * bd_agg_all + gamma_dd * dd_agg_all
            if hierarchical:
                # Preserve H0 exactly as the backbone of H1/H2: devices still
                # receive the deterministic terminal-bus lift, while learned
                # local messages are added through the combined relation path.
                # Thus the hierarchy is an additive ablation, not a silent
                # replacement of the old bus-to-device readout.
                r_bus_only = self.dyn_unified_out(
                    agg.reshape(h.size(0), self.hidden_dim)
                )
                r_base, lifted_device_count = self._lift_bus_dynamic_to_devices(
                    r_bus_only, slots, edge_index_line
                )
                r_dyn = self.dyn_unified_out(
                    combined_agg.reshape(h.size(0), self.hidden_dim)
                )
                device_mask = (slots > 0).unsqueeze(-1).to(dtype=r_dyn.dtype)
                r_dyn = r_dyn + r_base * device_mask
            else:
                r_dyn = self.dyn_unified_out(
                    combined_agg.reshape(h.size(0), self.hidden_dim)
                )
                r_dyn, lifted_device_count = self._lift_bus_dynamic_to_devices(
                    r_dyn, slots, edge_index_line
                )
            selected_k = torch.cat(k_all) if k_all else h.new_zeros(1)
            retained_mass = torch.cat(retained_mass_all) if retained_mass_all else h.new_zeros(1)
            candidate_count = (
                torch.cat(candidate_count_all) if candidate_count_all else h.new_ones(1)
            )
            budget_hit = torch.cat(budget_hit_all) if budget_hit_all else h.new_zeros(1)
            possible_edges = 0
            for graph_id in torch.unique(batch, sorted=True):
                graph_nodes = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
                n = (
                    int(graph_nodes.numel()) if hierarchical
                    else int(torch.sum(slots[graph_nodes] == 0).item())
                )
                possible_edges += n * max(n - 1, 0)
            density = (
                selected_k.sum().detach() + bd_edge_count.detach() + dd_edge_count.detach()
            ) / max(float(possible_edges), 1.0)
            entropy = (
                torch.stack(dense_entropy_terms).sum() / max(float(dense_alpha_counts), 1.0)
                if dense_entropy_terms else h.new_tensor(0.0)
            )
            # Every relation endpoint is a bus in this mode, therefore the
            # type-pair distribution is known without materialising O(N^2)
            # edge lists or copying them to CPU Python sets.
            typepair = h.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
            if hierarchical:
                typepair[0, 0] = bus_target_count
                typepair = typepair + gamma_bd.detach() * bd_typepair_all
                typepair = typepair + gamma_dd.detach() * dd_typepair_all
                typepair = typepair / typepair.sum().clamp_min(1e-12)
            else:
                typepair[0, 0] = 1.0
            zero = h.sum() * 0.0
            return r_dyn, {
                "dyn_edge_density": density,
                "dyn_entropy": entropy,
                "dyn_typepair_mass": typepair,
                "dyn_phy_overlap": h.new_tensor(0.0),
                "dyn_sparse_loss": zero,
                "dyn_dist_loss": zero,
                "dyn_k_mean": selected_k.mean().detach(),
                "dyn_k_max": selected_k.max().detach(),
                "dyn_retained_mass": retained_mass.mean().detach(),
                "dyn_budget_hit_rate": budget_hit.mean().detach(),
                **self._dynamic_support_diagnostics(
                    selected_k, retained_mass, candidate_count
                ),
                "dyn_bus_only": h.new_tensor(float(not hierarchical)),
                "dyn_bus_count": (slots == 0).sum().to(dtype=h.dtype).detach(),
                "dyn_device_lift_count": lifted_device_count.detach(),
                "p5_4_support_topology": self.support_topology,
                "p5_4_support_records": p5_4_support_records,
                "p5_4_detailed_records": p5_4_detailed_records,
                "gamma_bus_device": gamma_bd.detach(),
                "gamma_device_device": gamma_dd.detach(),
                "R_bus_device_norm": (
                    bd_agg_all.detach().norm() / math.sqrt(max(bd_agg_all.numel(), 1))
                ),
                "R_device_device_norm": (
                    dd_agg_all.detach().norm() / math.sqrt(max(dd_agg_all.numel(), 1))
                ),
                "bus_device_edge_count": bd_edge_count.detach(),
                "device_device_edge_count": dd_edge_count.detach(),
            }

        if not src_all:
            return torch.zeros_like(h), self._empty_dyn_debug(h, "dyn")

        r_dyn = self.dyn_unified_out(agg.reshape(h.size(0), self.hidden_dim))
        lifted_device_count = h.new_tensor(0.0)
        if bus_dense:
            r_dyn, lifted_device_count = self._lift_bus_dynamic_to_devices(
                r_dyn, slots, edge_index_line
            )
        src = torch.cat(src_all)
        dst = torch.cat(dst_all)
        alpha = torch.cat(alpha_all).to(dtype=h.dtype)
        possible_edges = 0
        for graph_id in torch.unique(batch, sorted=True):
            graph_nodes = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
            n = int(torch.sum(slots[graph_nodes] == 0).item()) if bus_dense else int(graph_nodes.numel())
            possible_edges += n * max(n - 1, 0)
        density = float(src.numel()) / max(float(possible_edges), 1.0)
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-12))).sum() / max(float(alpha.numel()), 1.0)
        typepair = h.new_zeros(self.node_type_count + 1, self.node_type_count + 1)
        typepair.index_put_(
            (torch.cat(type_dst_all), torch.cat(type_src_all)), alpha.detach(), accumulate=True
        )
        typepair = typepair / typepair.sum().clamp_min(1e-12)
        overlap = h.new_tensor(0.0)
        if edge_index_phy is not None and edge_index_phy.numel() > 0:
            dyn_pairs = set(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()))
            phy_pairs = set(zip(
                edge_index_phy[0].detach().cpu().tolist(),
                edge_index_phy[1].detach().cpu().tolist(),
            ))
            if dyn_pairs:
                overlap = h.new_tensor(len(dyn_pairs & phy_pairs) / len(dyn_pairs))
        retained_mass = torch.cat(retained_mass_all)
        selected_k = torch.cat(k_all)
        candidate_count = torch.cat(candidate_count_all)
        budget_hit = torch.cat(budget_hit_all)
        return r_dyn, {
            "dyn_edge_density": h.new_tensor(density),
            "dyn_entropy": entropy.detach(),
            "dyn_typepair_mass": typepair,
            "dyn_phy_overlap": overlap,
            "dyn_sparse_loss": alpha.abs().mean(),
            "dyn_dist_loss": alpha.sum() * 0.0,
            "dyn_k_mean": selected_k.mean().detach(),
            "dyn_k_max": selected_k.max().detach(),
            "dyn_retained_mass": retained_mass.mean().detach(),
            "dyn_budget_hit_rate": budget_hit.mean().detach(),
            **self._dynamic_support_diagnostics(
                selected_k, retained_mass, candidate_count
            ),
            "dyn_bus_only": h.new_tensor(float(bus_dense)),
            "dyn_bus_count": (slots == 0).sum().to(dtype=h.dtype).detach(),
            "dyn_device_lift_count": lifted_device_count.detach(),
        }

    def _dynamic_branch(
        self,
        h: torch.Tensor,
        h_initial: torch.Tensor | None,
        x_node: torch.Tensor | None,
        x_slow: torch.Tensor | None,
        x_fast: torch.Tensor | None,
        node_type_id: torch.Tensor,
        batch: torch.Tensor | None,
        edge_index_phy: torch.Tensor | None,
        edge_index_line: torch.Tensor | None,
        event_node_mask: torch.Tensor | None,
        graph_level_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.unified_dynamic:
            return self._unified_dynamic_branch(
                h, h_initial, x_slow, x_fast, node_type_id, batch, edge_index_phy,
                edge_index_line, event_node_mask,
                graph_level_feat=graph_level_feat,
            )
        r_slow, slow_debug = self._dynamic_subbranch(h, x_node, x_slow, x_fast, node_type_id, batch, edge_index_phy, view="slow")
        r_fast, fast_debug = self._dynamic_subbranch(h, x_node, x_slow, x_fast, node_type_id, batch, edge_index_phy, view="fast")
        gamma_slow = torch.sigmoid(self.gamma_slow_logit)
        gamma_fast = torch.sigmoid(self.gamma_fast_logit)
        r_dyn = (
            0.5 * (r_slow + r_fast)
            if self.split_readout else
            self.dyn_fuse(torch.cat([gamma_slow * r_slow, gamma_fast * r_fast], dim=-1))
        )

        dyn_typepair = 0.5 * (slow_debug["dyn_slow_typepair_mass"] + fast_debug["dyn_fast_typepair_mass"])
        debug = {
            **slow_debug,
            **fast_debug,
            "R_slow_norm": r_slow.detach().norm() / math.sqrt(max(r_slow.numel(), 1)),
            "R_fast_norm": r_fast.detach().norm() / math.sqrt(max(r_fast.numel(), 1)),
            "gamma_slow": gamma_slow.detach(),
            "gamma_fast": gamma_fast.detach(),
            "dyn_edge_density": 0.5 * (slow_debug["dyn_slow_edge_density"] + fast_debug["dyn_fast_edge_density"]),
            "dyn_entropy": 0.5 * (slow_debug["dyn_slow_entropy"] + fast_debug["dyn_fast_entropy"]),
            "dyn_phy_overlap": 0.5 * (slow_debug["dyn_slow_phy_overlap"] + fast_debug["dyn_fast_phy_overlap"]),
            "dyn_typepair_mass": dyn_typepair,
            "dyn_sparse_loss": 0.5 * (slow_debug["dyn_slow_sparse_loss"] + fast_debug["dyn_fast_sparse_loss"]),
            "dyn_dist_loss": 0.5 * (slow_debug["dyn_slow_dist_loss"] + fast_debug["dyn_fast_dist_loss"]),
            "_r_slow": r_slow,
            "_r_fast": r_fast,
        }
        return r_dyn, debug

    def forward(
        self,
        h: torch.Tensor,
        x_node: torch.Tensor | None,
        node_type_id: torch.Tensor,
        h_initial: torch.Tensor | None = None,
        x_slow: torch.Tensor | None = None,
        x_fast: torch.Tensor | None = None,
        edge_index_phy: torch.Tensor | None = None,
        edge_index_line: torch.Tensor | None = None,
        event_node_mask: torch.Tensor | None = None,
        edge_attr_phy: torch.Tensor | None = None,
        graph_level_feat: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        mode: str = "haag",
        dyn_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mode = str(mode)
        if mode not in {"phy_graph", "haag", "dyn_only"}:
            raise ValueError("HAAGLayer mode must be 'phy_graph', 'haag', or 'dyn_only'")
        r_phy = self._physical_branch(
            h, edge_index_phy, edge_attr_phy, node_type_id=node_type_id
        ) if mode in {"phy_graph", "haag"} else torch.zeros_like(h)
        dyn_debug = None
        if mode in {"haag", "dyn_only"}:
            r_dyn, dyn_debug = self._dynamic_branch(
                h, h_initial, x_node, x_slow, x_fast, node_type_id, batch, edge_index_phy,
                edge_index_line, event_node_mask,
                graph_level_feat=graph_level_feat,
            )
        else:
            r_dyn = torch.zeros_like(h)

        gamma_phy = torch.sigmoid(self.gamma_phy_logit)
        gamma_dyn_raw = torch.sigmoid(self.gamma_dyn_logit)
        dyn_scale_t = torch.as_tensor(dyn_scale, dtype=h.dtype, device=h.device)
        if self.unified_dynamic and dyn_debug is not None:
            gamma_dyn_node = self.unified_gate_max * torch.sigmoid(
                self.dyn_unified_gate(torch.cat([h, r_dyn], dim=-1))
            ) * dyn_scale_t
            gamma_dyn = gamma_dyn_node.mean()
        else:
            gamma_dyn_node = None
            gamma_dyn = gamma_dyn_raw * dyn_scale_t
        fused = h
        if mode in {"phy_graph", "haag"}:
            fused = fused + gamma_phy * r_phy
        if self.split_readout and mode in {"haag", "dyn_only"} and dyn_debug is not None:
            r_slow = dyn_debug["_r_slow"]
            r_fast = dyn_debug["_r_fast"]
            gamma_slow_out = self.split_gate_max * torch.sigmoid(self.gamma_slow_out_logit) * dyn_scale_t
            gamma_fast_out = self.split_gate_max * torch.sigmoid(self.gamma_fast_out_logit) * dyn_scale_t
            h_base = self.dropout(F.silu(self.split_base_norm(fused)))
            h_slow = self.dropout(F.silu(self.split_slow_norm(h_base + gamma_slow_out * r_slow)))
            h_fast = self.dropout(F.silu(self.split_fast_norm(h_base + gamma_fast_out * r_fast)))
            h_out = self.norm(h_base + gamma_slow_out * r_slow + gamma_fast_out * r_fast)
            h_out = self.dropout(F.silu(h_out))
            slow_effective = gamma_slow_out.detach() * dyn_debug["R_slow_norm"]
            fast_effective = gamma_fast_out.detach() * dyn_debug["R_fast_norm"]
            dyn_effective = slow_effective + fast_effective
            phy_effective = gamma_phy.detach() * (
                r_phy.detach().norm() / math.sqrt(max(r_phy.numel(), 1))
            )
            dyn_to_phy = (
                (slow_effective + fast_effective) / phy_effective.clamp_min(1e-8)
                if mode in {"phy_graph", "haag"} else h.new_tensor(0.0)
            )
        elif mode in {"haag", "dyn_only"}:
            fused = fused + (
                gamma_dyn_node * r_dyn if gamma_dyn_node is not None else gamma_dyn * r_dyn
            )
            h_out = self.norm(fused)
            h_out = self.dropout(F.silu(h_out))
            gamma_slow_out = h.new_tensor(0.0)
            gamma_fast_out = h.new_tensor(0.0)
            slow_effective = h.new_tensor(0.0)
            fast_effective = h.new_tensor(0.0)
            if gamma_dyn_node is not None:
                dyn_effective = (gamma_dyn_node.detach() * r_dyn.detach()).norm() / math.sqrt(
                    max(r_dyn.numel(), 1)
                )
                phy_effective = gamma_phy.detach() * (
                    r_phy.detach().norm() / math.sqrt(max(r_phy.numel(), 1))
                )
                dyn_to_phy = (
                    dyn_effective / phy_effective.clamp_min(1e-8)
                    if mode == "haag" else h.new_tensor(0.0)
                )
            else:
                dyn_effective = gamma_dyn.detach() * (
                    r_dyn.detach().norm() / math.sqrt(max(r_dyn.numel(), 1))
                )
                dyn_to_phy = h.new_tensor(0.0)
            h_base = h_slow = h_fast = h_out
        else:
            h_out = self.norm(fused)
            h_out = self.dropout(F.silu(h_out))
            gamma_slow_out = h.new_tensor(0.0)
            gamma_fast_out = h.new_tensor(0.0)
            slow_effective = h.new_tensor(0.0)
            fast_effective = h.new_tensor(0.0)
            dyn_effective = h.new_tensor(0.0)
            dyn_to_phy = h.new_tensor(0.0)
            h_base = h_slow = h_fast = h_out
        # Differentiable training-only ratio. The physical denominator is
        # detached deliberately: balance regularization should reduce the
        # dynamic correction, not inflate the physical residual to game the
        # ratio. Public diagnostics remain detached below.
        dyn_balance_ratio_live = h.sum() * 0.0
        if mode == "haag":
            if self.split_readout and dyn_debug is not None:
                dynamic_live = gamma_slow_out * dyn_debug["_r_slow"] + gamma_fast_out * dyn_debug["_r_fast"]
            elif gamma_dyn_node is not None:
                dynamic_live = gamma_dyn_node * r_dyn
            else:
                dynamic_live = gamma_dyn * r_dyn
            dyn_rms_live = torch.sqrt(dynamic_live.square().mean() + 1e-12)
            phy_rms_ref = torch.sqrt((gamma_phy * r_phy).detach().square().mean() + 1e-12)
            dyn_balance_ratio_live = dyn_rms_live / phy_rms_ref.clamp_min(1e-8)
        debug = {
            "R_phy_norm": r_phy.detach().norm() / math.sqrt(max(r_phy.numel(), 1)),
            "R_dyn_norm": r_dyn.detach().norm() / math.sqrt(max(r_dyn.numel(), 1)),
            "gamma_phy": gamma_phy.detach(),
            "gamma_dyn": (
                (gamma_slow_out + gamma_fast_out).detach()
                if self.split_readout and mode in {"haag", "dyn_only"} else gamma_dyn.detach()
            ),
            "gamma_dyn_raw": gamma_dyn_raw.detach(),
            "dyn_warmup_scale": dyn_scale_t.detach(),
            "dyn_debug": dyn_debug,
            "gamma_slow_out": gamma_slow_out.detach(),
            "gamma_fast_out": gamma_fast_out.detach(),
            "slow_effective_norm": slow_effective.detach(),
            "fast_effective_norm": fast_effective.detach(),
            "dyn_to_phy_effective_ratio": dyn_to_phy.detach(),
            "_dyn_to_phy_effective_ratio_live": dyn_balance_ratio_live,
            "dyn_effective_norm": dyn_effective.detach(),
            "gamma_dyn_node_std": (
                gamma_dyn_node.detach().std(unbiased=False) if gamma_dyn_node is not None else h.new_tensor(0.0)
            ),
            "gamma_dyn_node_min": (
                gamma_dyn_node.detach().min() if gamma_dyn_node is not None else gamma_dyn.detach()
            ),
            "gamma_dyn_node_max": (
                gamma_dyn_node.detach().max() if gamma_dyn_node is not None else gamma_dyn.detach()
            ),
            "_h_base": h_base,
            "_h_slow": h_slow,
            "_h_fast": h_fast,
        }
        if dyn_debug is not None:
            debug.update(dyn_debug)
        return h_out, debug
