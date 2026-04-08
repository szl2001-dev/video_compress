# 关键帧引导的层次化 Token 压缩 Projector
#
# 放置位置：复制到
#   VideoChat-Flash/llava-train_videochat/llava/model/multimodal_projector/
#
# 原理：
#   - 视频 clip = 4 帧，每帧 196 tokens（14×14 patches，224×224 分辨率）
#   - 第一帧（关键帧）做帧内去冗余：196 → 147
#   - 后三帧（增量帧，588 tokens）做跨帧迭代合并：588 → 49
#   - 最终输出 147 + 49 = 196 tokens/clip
#
# 与 tome16_mlp_hd64 的关系：
#   - MLP 结构完全相同（可直接加载 stage2 checkpoint 的 MLP 权重）
#   - compress=False（图片）时行为与原版相同（复用 ToMe 逻辑压缩到 64 tokens）
#   - compress=True（视频）时使用新的关键帧引导压缩

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple


# ============================================================
# ToMe 工具函数（用于 compress=False 的图片路径，保持与原版一致）
# ============================================================

def bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable]:
    """二分图软匹配（ToMe），用于图片路径的 token 压缩。"""
    protected = 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None):
    """加权平均合并（ToMe），用于图片路径。"""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


# ============================================================
# 主类：KFGuided_MLP
# ============================================================

class KFGuided_MLP(nn.Module):
    """
    关键帧引导层次化 Token 压缩 + MLP Projector。

    视频路径（compress=True）：
        输入  (B * local_num_frames, 196, mm_hidden_size)
        压缩后 (B, 196, hidden_size)  ← 关键帧引导算法
        每 clip 平均每帧 49 tokens

    图片路径（compress=False）：
        输入  (1, 196, mm_hidden_size)
        压缩后 (1, 64, hidden_size)   ← 复用 ToMe，与原版一致
    """

    # 关键帧 token 数量（第一帧保留）
    KF_KEEP = 147       # 196 - 49
    # 增量帧最终保留数量
    INCR_KEEP = 49
    # 图片路径压缩目标（与 tome16_mlp_hd64 保持一致）
    IMAGE_TOKENS = 64

    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.mm_hidden_size = config.mm_hidden_size
        self.hw = vision_cfg.image_size // vision_cfg.patch_size  # 14
        self.num_attention_heads = vision_cfg.num_attention_heads

        # MLP 与 tome16_mlp_hd64 完全相同 → stage2 权重可直接加载
        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    # ----------------------------------------------------------
    # 图片路径：ToMe 压缩到 64 tokens（与原版行为一致）
    # ----------------------------------------------------------

    def _tome_merge(self, x: torch.Tensor, target_num_token: int) -> torch.Tensor:
        """迭代 ToMe 合并，直到 token 数降至 target_num_token。"""
        size = None
        b, p, c = x.shape
        r_merge_list = []
        tmp_p = p
        assert tmp_p > target_num_token

        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)

        head = self.num_attention_heads
        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2)
            merge, _ = bipartite_soft_matching(metric, r)
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        return x

    # ----------------------------------------------------------
    # 视频路径：关键帧引导层次化压缩
    # ----------------------------------------------------------

    def _kf_guided_compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        关键帧引导层次化 Token 压缩。

        参数：
            x : (B, 4*196, D) = (B, 784, D)，B = clip 数量

        返回：
            out : (B, 196, D)，其中前 147 来自关键帧，后 49 来自增量帧
        """
        B, _, D = x.shape
        KF_TOTAL = self.hw * self.hw        # 196
        KF_KEEP  = self.KF_KEEP             # 147
        INCR_KEEP = self.INCR_KEEP          # 49

        # ── 拆分关键帧 / 增量帧 ──────────────────────────────────
        kf   = x[:, :KF_TOTAL, :].clone()  # (B, 196, D)
        incr = x[:, KF_TOTAL:, :].clone()  # (B, 588, D)

        # ── Stage 1：关键帧内部去冗余 ────────────────────────────
        # 计算 196×196 余弦相似度矩阵
        kf_norm = F.normalize(kf, dim=-1)                    # (B, 196, D)
        sim_kf  = torch.bmm(kf_norm, kf_norm.transpose(1, 2))  # (B, 196, 196)

        # 冗余度 = 与所有其他 token 的相似度之和（减去自身 1.0）
        redundancy = sim_kf.sum(dim=-1) - 1.0                # (B, 196)

        # 按冗余度升序排列（最独特在前），保留前 KF_KEEP 个
        sorted_idx  = redundancy.argsort(dim=-1, descending=False)  # (B, 196)
        keep_kf_idx = sorted_idx[:, :KF_KEEP].sort(dim=-1)[0]       # (B, 147)，保持原始顺序

        # 聚合选中的关键帧 token
        expand_idx = keep_kf_idx.unsqueeze(-1).expand(-1, -1, D)     # (B, 147, D)
        kf_selected = kf.gather(1, expand_idx)                       # (B, 147, D)

        # 用"和 + 计数"方式跟踪加权平均
        kf_sum   = kf_selected.clone()                               # (B, 147, D)
        kf_count = torch.ones(B, KF_KEEP, 1, device=x.device, dtype=x.dtype)  # (B, 147, 1)

        # ── Stage 2：增量帧跨帧对齐压缩（迭代） ─────────────────
        # 目标：从 588 个增量帧 token 中保留 INCR_KEEP = 49 个
        n_incr = incr.shape[1]  # 初始 588

        while n_incr > INCR_KEEP:
            # 计算当前关键帧均值（加权平均）
            kf_avg  = kf_sum / kf_count                              # (B, 147, D)
            kf_n    = F.normalize(kf_avg, dim=-1)                    # (B, 147, D)
            incr_n  = F.normalize(incr, dim=-1)                      # (B, n_incr, D)

            # 跨帧相似度矩阵：(B, 147, n_incr)
            sim_cross = torch.bmm(kf_n, incr_n.transpose(1, 2))     # (B, 147, n_incr)

            # 每个增量 token 与关键帧的最大相似度 → 找到最合适的"合并目标"
            max_sim, best_kf = sim_cross.max(dim=1)                  # 均为 (B, n_incr)

            # 本轮合并数量：全局 top-1/3，但不超过达到目标所需的数量
            n_to_merge = min(n_incr // 3, n_incr - INCR_KEEP)
            if n_to_merge <= 0:
                break

            # 按最大相似度降序排列，取前 n_to_merge 个增量 token 合并
            rank          = max_sim.argsort(dim=-1, descending=True)          # (B, n_incr)
            merge_idx     = rank[:, :n_to_merge]                              # (B, n_to_merge)
            keep_idx_incr = rank[:, n_to_merge:].sort(dim=-1)[0]             # (B, n_incr - n_to_merge)

            # 确定每个待合并 token 的目标关键帧
            target_kf = best_kf.gather(1, merge_idx)                         # (B, n_to_merge)

            # 取出待合并的增量帧 token 特征
            merge_feats = incr.gather(
                1, merge_idx.unsqueeze(-1).expand(-1, -1, D)
            )                                                                  # (B, n_to_merge, D)

            # 累加到对应的关键帧 token（加权平均分子）
            kf_sum = kf_sum.scatter_add(
                1,
                target_kf.unsqueeze(-1).expand(-1, -1, D),
                merge_feats,
            )
            # 更新计数（加权平均分母）
            kf_count = kf_count.scatter_add(
                1,
                target_kf.unsqueeze(-1),
                torch.ones(B, n_to_merge, 1, device=x.device, dtype=x.dtype),
            )

            # 保留未被合并的增量帧 token
            incr = incr.gather(
                1, keep_idx_incr.unsqueeze(-1).expand(-1, -1, D)
            )
            n_incr -= n_to_merge

        # 最终关键帧 token = 加权平均（融合了相似的增量 token）
        kf_final = kf_sum / kf_count   # (B, 147, D)

        # 拼接：147 个关键帧 token + 49 个最具差异性的增量 token
        return torch.cat([kf_final, incr], dim=1)   # (B, 196, D)

    # ----------------------------------------------------------
    # forward
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor, compress: bool = False, local_num_frames: int = -1) -> torch.Tensor:
        """
        参数：
            x              : 视觉编码器输出，形状 (F, N, D)
                             F = 总帧数（或 batch * num_frames）
                             N = 每帧 token 数（= hw * hw = 196）
                             D = mm_hidden_size
            compress       : True → 视频路径（关键帧引导压缩）
                             False → 图片路径（ToMe 压缩至 64 tokens）
            local_num_frames: 视频路径时每个 clip 的帧数（应为 4）

        返回：
            视频路径 : (n_clips, 196, hidden_size)
            图片路径 : (1, 64, hidden_size)
        """
        height = width = self.hw
        assert height * width == x.shape[1], (
            f"Token 数 {x.shape[1]} 与期望的 {height * width} 不匹配"
        )

        if compress:
            # ── 视频路径 ───────────────────────────────────────
            assert local_num_frames > 1, "视频路径要求 local_num_frames > 1"
            # reshape：(F, 196, D) → (n_clips, local_num_frames*196, D)
            # 其中 F = n_clips * local_num_frames
            n_clips = x.shape[0] // local_num_frames
            x = x.reshape(n_clips, local_num_frames * height * width, x.shape[-1])
            # 关键帧引导压缩：→ (n_clips, 196, D)
            x = self._kf_guided_compress(x)
        else:
            # ── 图片路径：ToMe 压缩到 64 tokens ────────────────
            # 注意：不需要 reshape，直接按 batch 压缩
            # 输入 (B, 196, D) → 输出 (B, 64, D)，与 tome16_mlp_hd64 行为一致
            x = self._tome_merge(x, target_num_token=self.IMAGE_TOKENS)

        x = self.mlp(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "kf_guided_mlp"}
