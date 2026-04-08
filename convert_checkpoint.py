"""
convert_checkpoint.py
=====================
将 stage2 checkpoint（tome16_mlp_hd64）转换为使用关键帧引导压缩（kf_guided_mlp）的 checkpoint。

做法：
  1. 把 stage2 所有文件复制到输出目录
  2. 修改 config.json 中的 mm_projector_type
  3. MLP 权重结构完全相同，无需任何权重修改

不需要加载模型，秒级完成。

运行位置：video_compress/ 根目录
用法：
    python convert_checkpoint.py \
        --stage2_path ./checkpoints/stage2-UMT-Qwen2-7B-tome16_mlp \
        --output_path ./checkpoints/stage2-kf_guided-init
"""

import argparse
import os
import shutil
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[1/3] 复制 checkpoint: {args.stage2_path} → {args.output_path}")
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    shutil.copytree(args.stage2_path, args.output_path)

    print("[2/3] 修改 config.json: mm_projector_type → kf_guided_mlp")
    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    old_type = config.get("mm_projector_type")
    assert old_type == "tome16_mlp_hd64", f"期望 tome16_mlp_hd64，实际是 {old_type}"
    config["mm_projector_type"] = "kf_guided_mlp"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("[3/3] 验证")
    with open(config_path) as f:
        saved = json.load(f)
    assert saved["mm_projector_type"] == "kf_guided_mlp"

    print("转换完成！")
    print(f"  mm_projector_type: {old_type} → kf_guided_mlp")
    print(f"  MLP 权重无需修改（结构与 tome16_mlp_hd64 完全相同）")
    print(f"  输出路径: {args.output_path}")


if __name__ == "__main__":
    main()
