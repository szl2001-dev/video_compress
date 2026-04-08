# Video Token Compression: Keyframe-Guided MLP Projector

基于 [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash) 的视频 Token 压缩研究，提出关键帧引导的层次化 Token 压缩方法（`kf_guided_mlp`），在保持视觉信息的同时大幅压缩视频 token 数量。

## 核心思路

原版 VideoChat-Flash 使用 ToMe（Token Merging）对所有帧均匀压缩（`tome16_mlp_hd64`，每帧保留 16 tokens）。本工作提出不同策略：**对关键帧和增量帧区别对待**。

### 压缩算法（`kf_guided_mlp`）

每个 clip 包含 4 帧，每帧 196 tokens（14×14 patches，224×224 分辨率）：

| 组成 | 输入 tokens | 输出 tokens | 方法 |
|------|------------|------------|------|
| 关键帧（第 1 帧） | 196 | 147 | 帧内余弦相似度去冗余，保留最独特的 token |
| 增量帧（后 3 帧） | 588 | 49 | 跨帧迭代合并，与关键帧相似的 token 被吸收 |
| **合计** | **784** | **196** | 4:1 压缩比，等效每帧 49 tokens |

图片路径（`compress=False`）：复用 ToMe 压缩至 64 tokens，与原版行为完全一致。

### 两阶段压缩细节

**Stage 1 - 关键帧内部去冗余：**
- 计算 196×196 余弦相似度矩阵
- 按冗余度（与所有 token 的相似度之和）升序排列
- 保留最独特的 147 个 token

**Stage 2 - 增量帧跨帧对齐压缩（迭代）：**
- 以关键帧均值为锚点，计算增量帧 token 与关键帧的相似度
- 每轮合并相似度最高的 1/3 增量 token（融合入对应关键帧）
- 迭代至增量帧仅剩 49 个最具差异性的 token

### 与 `tome16_mlp_hd64` 的关系

- MLP 结构完全相同（Linear → GELU → Linear），stage2 权重可直接复用
- 仅 `config.json` 中的 `mm_projector_type` 不同
- checkpoint 转换只需秒级完成（无需重新训练 MLP）

## 目录结构

```
video_compress/
├── llava/
│   ├── model/
│   │   ├── multimodal_projector/
│   │   │   ├── keyframe_guided_mlp.py   # 核心：kf_guided_mlp projector
│   │   │   └── tome16_mlp_hd64.py       # 原版 ToMe projector（基线）
│   │   ├── multimodal_encoder/
│   │   │   └── umt_encoder.py           # UMT-Large 视觉编码器
│   │   └── llava_arch.py
│   └── train/
│       ├── train_mem.py                 # 训练入口
│       └── llava_trainer.py
├── lmms_eval/                           # 评估框架
├── scripts/
│   ├── zero1.json                       # DeepSpeed ZeRO-1 配置
│   └── zero2.json                       # DeepSpeed ZeRO-2 配置
├── data/                                # 数据配置 yaml 文件
├── checkpoints/                         # 模型 checkpoint
├── convert_checkpoint.py                # checkpoint 格式转换
├── train_kf_guided.sh                   # 快速验证训练（8 GPU，clevrer 82K）
├── train_kf_guided_subset.sh            # 单卡子集验证（clevrer 20K）
├── finetune_kf_guided.sh                # 完整 SFT 微调（Slurm）
├── eval_mvbench.sh                      # MVBench 评估
└── eval_videomme.sh                     # Video-MME 评估
```

## 使用流程

### 1. 环境准备

```bash
conda activate video_compress
```

### 2. 转换 Checkpoint

将原版 `tome16_mlp_hd64` checkpoint 转换为 `kf_guided_mlp` 初始化点（秒级完成，无需加载模型）：

```bash
python convert_checkpoint.py \
    --stage2_path ./checkpoints/stage2-UMT-Qwen2-7B-tome16_mlp \
    --output_path ./checkpoints/stage2-kf_guided-init
```

### 3. 快速验证（单卡，clevrer 20K）

```bash
bash train_kf_guided_subset.sh
```

- 单 GPU，clevrer_qa_20k 数据
- 只训练 `mm_mlp_adapter`，冻结 LLM 和视觉编码器
- 适合快速验证 projector 适配情况

### 4. 多卡训练（8 GPU，clevrer 82K）

```bash
bash train_kf_guided.sh
```

- 8 GPU，clevrer_qa + clevrer_mc（~82K 条）
- 只训练 `mm_mlp_adapter`
- 使用 DeepSpeed ZeRO-2

### 5. 完整 SFT 微调

```bash
bash finetune_kf_guided.sh
```

- 训练 `mm_mlp_adapter` + `mm_language_model`（可配置）
- 需要 stage3 SFT 数据（`data/stage3_short-long_mix_sft.yaml`）
- 支持 Slurm 和 torchrun 两种启动方式

### 6. 评估

**MVBench**（短视频，20 个子任务多选题）：

```bash
# 修改 eval_mvbench.sh 中的 CKPT_PATH 后运行
bash eval_mvbench.sh
```

**Video-MME**（Short / Medium / Long，有/无字幕）：

```bash
# 修改 eval_videomme.sh 中的 CKPT_PATH 为微调后的 checkpoint
bash eval_videomme.sh
```

## 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--mm_projector_type` | `kf_guided_mlp` | 使用关键帧引导压缩 |
| `--local_num_frames` | 4 | 每个 clip 帧数 |
| `--mm_local_num_frames` | 4 | projector 内部 clip 大小 |
| `--frames_upbound` | 64 / 512 | 最大采样帧数 |
| `--sample_type` | `dynamic_fps1` | 动态帧率采样 |
| `--vision_encode_type` | `video_image` | 视频+图片混合编码 |
| `--mm_tunable_parts` | `mm_mlp_adapter` | 仅训练 projector（验证阶段） |

## 基础模型

- 视觉编码器：UMT-Large
- 语言模型：Qwen2-7B
- 基座：VideoChat-Flash (stage2 checkpoint)
