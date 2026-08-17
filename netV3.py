import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import degree
from torch_scatter import scatter_mean

from modules.conv import DynConv
from modules.basic import MLPLinear

# ---------------------------------------------------------------------------
# netV3: point-wise sketch segmentation model, architecture follows the paper
# SketchMeSeg (Sketch Segmentation via Geometric and Sequential Processing).
#
# Two parallel branches are built on the raw point sequence:
#   * Geometric branch   : 4 stacked DynGCN blocks (DGCNN-style dynamic
#                          graph convolution, k=8) whose outputs are
#                          concatenated into F_geo of size 128.
#   * Sequence branch    : 1-layer Bi-GRU (hidden 32 -> output 64) followed by
#                          4 Transformer Encoder layers (d_model=64, 4 heads,
#                          FFN width 192, dropout 0.1). The multi-head attention
#                          is biased by the T-Grd encoding:
#                            central  : z^-(deg-) + z^+(deg+) added to tokens
#                            geometry : b_ij = w_b^T [phi_local, phi_global]
#                            stroke   : c_ij = (1/M) sum_m e_m^T w^E
# Fusion: cat(F_geo, S) -> MLP(192 -> 256). A stroke-level pooling aggregates
# the fused features, two LBR blocks + a linear head classify each stroke, and
# the per-stroke logits are broadcast back to the original points.
# ---------------------------------------------------------------------------

D_GRU   = 64     # Bi-GRU output width per point (also d_model of Transformer)
D_PROJ  = 64     # geometric branch input projection width
D_BLOCK = 32     # output width of a single DynGCN block
D_G     = D_BLOCK * 4   # concatenated geometric feature width (128)
D_FUSE  = 256    # fused feature width before stroke pooling
N_HEADS = 4
HID_FFN = 192
N_TF_LAYERS  = 4
N_GCN_LAYERS = 4
K_NEIGH      = 8
MAX_DEG      = 3    # centrality degrees are clamped into [0, MAX_DEG)
D_EDGE       = 32   # stroke encoding edge feature width (d_E)


class TGrdBias(nn.Module):
    """Computes the geometry and stroke attention biases of the T-Grd encoding.

    For every ordered point pair (i, j):
      phi_local(i, j)  : average turning curvature along the shortest path
                         (within the same stroke) between i and j,
                         normalized to [0, 1]
      phi_global(i, j) : shortest-path length in the stroke graph, -1 when i,j
                         are not connected
      b_ij             : w_b[0] * phi_local + w_b[1] * phi_global
      c_ij             : (1/M) sum_m e_m^T w^E over the edges e_m of the path,
                         -1 when i,j are not connected

    The stroke graph is a forest of paths, so shortest paths only exist between
    points sharing a stroke, and the path is the chain segment between them.
    The bias matrices only depend on geometry / structure, therefore they are
    computed once per forward and shared by all Transformer layers.
    """

    def __init__(self, d_edge=D_EDGE):
        super(TGrdBias, self).__init__()
        self.d_edge = d_edge
        self.wb = nn.Parameter(torch.randn(2))
        # shared linear projection of edge features (endpoint coordinate diff)
        self.edge_proj = nn.Linear(2, d_edge, bias=False)
        self.edge_score = nn.Parameter(torch.randn(d_edge))

    @staticmethod
    def _per_sample_bias(p, sl, wb, edge_proj, edge_score, device):
        # p  : (N, 2) normalized coordinates of one sketch
        # sl : (N,)   stroke id of every point (drawing order)
        N = p.size(0)
        ar = torch.arange(N, device=device)

        # stroke block geometry
        K = int(sl.max().item()) + 1
        is_start = torch.zeros(N, dtype=torch.bool, device=device)
        is_start[0] = True
        is_start[1:] = sl[1:] != sl[:-1]
        starts = ar[is_start]                     # start index of each stroke
        li = ar - starts[sl]                      # within-stroke position
        lens = torch.bincount(sl, minlength=K)
        slen = lens[sl]                           # stroke length per point

        same = sl[:, None] == sl[None, :]         # (N, N) same-stroke mask
        lo = torch.minimum(li[:, None], li[None, :])
        hi = torch.maximum(li[:, None], li[None, :])
        M = (hi - lo).float()                     # path length (edges)
        M_safe = M.clamp(min=1.0)

        # --- local curvature at each point --------------------------------
        has_prev = li > 0
        has_next = li < slen - 1
        d_in = torch.zeros(N, 2, device=device)
        d_out = torch.zeros(N, 2, device=device)
        p_prev = torch.cat([p[:1], p[:-1]], dim=0)   # p[i-1]
        p_next = torch.cat([p[1:], p[-1:]], dim=0)   # p[i+1]
        d_in[has_prev] = (p - p_prev)[has_prev]
        d_out[has_next] = (p_next - p)[has_next]
        denom = d_in.norm(dim=1) * d_out.norm(dim=1).clamp(min=1e-6)
        cosv = ((d_in * d_out).sum(dim=1) / denom.clamp(min=1e-6)).clamp(-1, 1)
        curv = torch.where(has_prev & has_next,
                           torch.acos(cosv) / math.pi, torch.zeros_like(cosv))

        # path-average curvature = average curv over the chain lo..hi
        gcs = torch.cat([torch.zeros(1, device=device), curv.cumsum(0)])
        curv_sum = gcs[hi + 1] - gcs[lo]          # includes both endpoints
        phi_local = torch.where(same, curv_sum / M_safe, torch.zeros_like(M))

        # --- global shortest-path length ----------------------------------
        d_global = torch.where(same, (hi - lo).float(),
                               torch.full_like(M, -1.0))

        b = wb[0] * phi_local + wb[1] * d_global

        # --- stroke encoding from path edge features ----------------------
        e = torch.zeros(N, 2, device=device)      # e_m = p[i+1] - p[i]
        e[has_next] = (p_next - p)[has_next]
        ecs = torch.cat([torch.zeros(1, 2, device=device), e.cumsum(0)])
        sum_e = ecs[hi] - ecs[lo]                 # (N, N, 2) path edge sum
        score = (edge_proj(sum_e) * edge_score).sum(dim=-1)
        c = torch.where(same, score / M_safe, torch.full_like(score, -1.0))

        return b + c

    def forward(self, pos, stroke_idx, batch):
        # pos        : (B*N, 2) normalized coordinates
        # stroke_idx : (B*N,) stroke id per point
        # batch      : (B*N,) sample id per point
        B = int(batch.max().item()) + 1
        bias_list = []
        pos = pos.view(B, -1, 2)
        sl = stroke_idx.view(B, -1)
        for i in range(B):
            bias_list.append(
                self._per_sample_bias(pos[i], sl[i], self.wb,
                                      self.edge_proj, self.edge_score,
                                      pos.device))
        return torch.stack(bias_list, dim=0)      # (B, N, N)


class CentralityEncoding(nn.Module):
    """Adds learnable in/out-degree embeddings to the node tokens."""

    def __init__(self, max_degree=MAX_DEG, node_dim=D_GRU):
        super(CentralityEncoding, self).__init__()
        self.max_degree = max_degree
        self.emb_in = nn.Embedding(max_degree, node_dim)
        self.emb_out = nn.Embedding(max_degree, node_dim)

    @staticmethod
    def _clamp_degree(deg):
        # keep values in [0, max_degree - 1]
        return deg.clamp(min=0, max=MAX_DEG - 1)

    def forward(self, x, edge_index, num_nodes):
        # x : (B*N, d)
        deg_in = degree(edge_index[1], num_nodes=num_nodes).long()
        deg_out = degree(edge_index[0], num_nodes=num_nodes).long()
        x = x + self.emb_in(self._clamp_degree(deg_in))
        x = x + self.emb_out(self._clamp_degree(deg_out))
        return x


class TGrdTransformerLayer(nn.Module):
    """Transformer encoder layer whose attention scores include the
    geometric + stroke biases (T-Grd)."""

    def __init__(self, d_model=D_GRU, n_heads=N_HEADS, hidden=HID_FFN,
                 dropout=0.1):
        super(TGrdTransformerLayer, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, bias):
        # x    : (B, N, d_model)
        # bias : (B, N, N) additive attention bias (T-Grd)
        B, N, _ = x.size()
        h = self.norm1(x)
        q = self.wq(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head ** -0.5)
        scores = scores + bias.unsqueeze(1)       # (B, H, N, N)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)               # (B, H, N, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, self.d_model)
        h = x + self.drop(self.wo(ctx))

        h2 = self.norm2(h)
        h = h + self.drop(self.ffn(h2))
        return h


class LBR(nn.Module):
    """Linear + BatchNorm + ReLU block."""

    def __init__(self, in_dim, out_dim):
        super(LBR, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DynGCNBlock(nn.Module):
    """One DynGCN block: dynamic EdgeConv (static stroke edges fused with
    k-NN edges) followed by batch norm + residual connection."""

    def __init__(self, in_ch, out_ch, opt):
        super(DynGCNBlock, self).__init__()
        self.conv = DynConv(in_ch, out_ch, 1, opt)   # k = 8, dilation = 1
        self.bn = nn.BatchNorm1d(out_ch)
        self.use_res = in_ch == out_ch

    def forward(self, x, edge_index, data):
        # x : (B*N, in_ch)
        h = self.conv(x, edge_index, data)
        h = F.relu(self.bn(h))
        if self.use_res:
            h = h + x
        return h


class CADSketch_Seg(nn.Module):
    """netV3 point-wise sketch segmentation network (paper SketchMeSeg)."""

    def __init__(self, opt):
        super(CADSketch_Seg, self).__init__()
        self.points_num = opt.points_num
        self.in_feature = opt.in_feature
        self.out_segment = opt.out_segment

        # ---- sequence branch ---------------------------------------------
        self.gru = nn.GRU(3, D_GRU // 2, bidirectional=True, num_layers=1,
                          batch_first=True)
        self.centrality = CentralityEncoding()
        self.t_grd = TGrdBias()
        self.tf_layers = nn.ModuleList(
            [TGrdTransformerLayer() for _ in range(N_TF_LAYERS)])

        # ---- geometric branch --------------------------------------------
        self.g_proj = MLPLinear([3, D_PROJ], act_type='relu', norm_type='batch')
        gcn_in = D_PROJ
        self.gcn_blocks = nn.ModuleList()
        for _ in range(N_GCN_LAYERS):
            self.gcn_blocks.append(DynGCNBlock(gcn_in, D_BLOCK, opt))
            gcn_in = D_BLOCK

        # ---- fusion -------------------------------------------------------
        self.fusion = MLPLinear([D_G + D_GRU, D_FUSE], act_type='relu',
                                norm_type='batch')

        # ---- stroke-level classifier --------------------------------------
        self.dec1 = LBR(D_FUSE, D_FUSE // 2)     # 256 -> 128
        self.dec2 = LBR(D_FUSE // 2, D_FUSE // 2)
        self.cls = nn.Linear(D_FUSE // 2, self.out_segment)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _build_stroke_mask(self, sl):
        # sl : (B, N) stroke id per point; returns a binary boundary signal
        B, N = sl.size()
        mask = torch.zeros((B, N), dtype=torch.float, device=sl.device)
        mask[:, 0] = 1.0
        mask[:, 1:] = (sl[:, 1:] != sl[:, :-1]).float()
        return mask

    def _stroke_key(self, sl, B, N):
        # globally unique stroke id across the batch: key = local stroke +
        # per-sample offset, in sample-major order
        stroke_counts = sl.max(dim=1).values + 1
        offset = stroke_counts.cumsum(dim=0) - stroke_counts
        gst = (sl + offset[:, None]).view(-1)
        return gst, int(gst.max().item()) + 1

    def forward(self, x, edge_index, stroke_data):
        # x           : (B*N, in_feature) normalized point coordinates
        # edge_index  : (2, E) stroke edges (consecutive points, both ways)
        # stroke_data : dict with stroke_idx / batch / pool_edge_index / pos
        num_nodes = x.size(0)
        B = int(stroke_data['batch'].max().item()) + 1
        N = self.points_num
        device = x.device

        stroke_idx = stroke_data['stroke_idx']
        pos = stroke_data['pos'] if stroke_data.get('pos') is not None else x

        # static stroke edges with self loops, used by the graph convolutions
        edge_all, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_all = torch.unique(edge_all, dim=1)

        x_b = x.view(B, N, self.in_feature)
        sl = stroke_idx.view(B, N)
        s_mask = self._build_stroke_mask(sl)
        seq_in = torch.cat([x_b, s_mask.unsqueeze(2)], dim=2)   # (B, N, 3)

        # ---- sequence branch ---------------------------------------------
        seq, _ = self.gru(seq_in)                      # (B, N, D_GRU)
        seq = seq.contiguous().view(-1, D_GRU)
        seq = self.centrality(seq, edge_index, num_nodes)

        t_grd_bias = self.t_grd(pos, stroke_idx, stroke_data['batch'])  # (B,N,N)
        seq = seq.view(B, N, D_GRU)
        for layer in self.tf_layers:
            seq = layer(seq, t_grd_bias)
        S = seq.contiguous().view(-1, D_GRU)           # (B*N, 64)

        # ---- geometric branch --------------------------------------------
        g_in = torch.cat([x_b, s_mask.unsqueeze(2)], dim=2)
        g_in = g_in.contiguous().view(-1, 3)
        g = self.g_proj(g_in)                          # (B*N, D_PROJ)
        geo_outs = []
        for block in self.gcn_blocks:
            g = block(g, edge_all, stroke_data)
            geo_outs.append(g)
        F_geo = torch.cat(geo_outs, dim=1)             # (B*N, D_G = 128)

        # ---- fusion -------------------------------------------------------
        Z = self.fusion(torch.cat([F_geo, S], dim=1))  # (B*N, D_FUSE)

        # ---- stroke-level pooling + decode -------------------------------
        gst, n_strokes = self._stroke_key(sl, B, N)
        Zs = scatter_mean(Z, gst, dim=0, dim_size=n_strokes)
        h = self.dec1(Zs)
        h = self.dec2(h)
        stroke_logits = self.cls(h)                    # (n_strokes, C)

        # broadcast stroke logits back to the points
        out = stroke_logits[gst]                       # (B*N, C)
        out = self.log_softmax(out)
        return out