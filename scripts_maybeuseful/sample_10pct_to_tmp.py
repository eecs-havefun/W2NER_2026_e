#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 procnet/sidecar/ 中抽取约 10% 的文档，替换 procnet/tmp_sidecar/ 下的文件
"""

import json
import random
from pathlib import Path

# 配置
SOURCE_DIR = Path("procnet/sidecar")
TARGET_DIR = Path("procnet/tmp_sidecar")
SPLITS = ["train", "dev", "test"]
SAMPLE_RATIO = 0.10  # 10%
SEED = 42  # 固定随机种子，保证可复现


def load_jsonl(path: Path) -> list:
    """加载 JSONL 文件"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    return docs


def save_jsonl(path: Path, docs: list):
    """保存为 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def sample_and_replace(split: str) -> tuple:
    """
    从 source 抽取 10% 文档，替换 target
    返回：(抽取数，原文档数)
    """
    source_file = SOURCE_DIR / f"{split}_doc_typed_entities.jsonl"
    target_file = TARGET_DIR / f"{split}_doc_typed_entities.jsonl"
    
    print(f"\n处理 {split}:")
    print(f"  源文件：{source_file}")
    print(f"  目标文件：{target_file}")
    
    # 加载源数据
    source_docs = load_jsonl(source_file)
    print(f"  源文档数：{len(source_docs)}")
    
    # 随机抽样 10%
    random.seed(SEED)
    sample_size = max(1, int(len(source_docs) * SAMPLE_RATIO))
    sampled_docs = random.sample(source_docs, sample_size)
    print(f"  抽取 10%: {sample_size} 篇")
    
    # 替换目标文件
    save_jsonl(target_file, sampled_docs)
    print(f"  已替换：{target_file}")
    
    return sample_size, len(source_docs)


def main():
    print("=" * 60)
    print("从 sidecar 抽取 10% 文档替换 tmp_sidecar")
    print(f"随机种子：{SEED}")
    print("=" * 60)
    
    stats = []
    for split in SPLITS:
        sampled, total = sample_and_replace(split)
        stats.append((split, sampled, total))
    
    print("\n" + "=" * 60)
    print("完成!")
    print("-" * 60)
    print(f"{'数据集':<10} {'抽取数':>10} {'源文档数':>10} {'比例':>10}")
    print("-" * 60)
    for split, sampled, total in stats:
        ratio = sampled / total * 100
        print(f"{split:<10} {sampled:>10} {total:>10} {ratio:>9.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
