#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 5 个数据集运行 W2NER smoke test
"""

import subprocess
import json
from pathlib import Path

W2NER_DIR = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER")
CONFIG_PATH = W2NER_DIR / "config" / "data_w2ner_folded.json"

DATASETS = [
    "flight_orders",
    "hotel_orders",
    "id_cards",
    "mixed_data",
    "train_orders",
]


def update_config(dataset: str):
    """更新配置文件"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    config["dataset"] = dataset
    config["save_path"] = f"{dataset}_model.pt"
    config["predict_path"] = f"{dataset}_output.json"
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def run_smoke_test(dataset: str):
    """运行单个数据集的 smoke test"""
    print(f"\n{'='*70}")
    print(f"运行：{dataset}")
    print("=" * 70)
    
    # 更新配置
    update_config(dataset)
    
    # 运行训练
    cmd = ["python", "main.py", "--config", "config/data_w2ner_folded.json", "--device", "0"]
    result = subprocess.run(cmd, cwd=W2NER_DIR, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✅ {dataset} 完成")
    else:
        print(f"\n❌ {dataset} 失败，返回码：{result.returncode}")
    
    return result.returncode


def main():
    print("=" * 70)
    print("W2NER Smoke Test - 5 个数据集")
    print("=" * 70)
    
    results = {}
    
    for dataset in DATASETS:
        results[dataset] = run_smoke_test(dataset)
    
    print("\n" + "=" * 70)
    print("Smoke Test 总结")
    print("=" * 70)
    print(f"{'数据集':<20} {'状态':>10}")
    print("-" * 70)
    
    for dataset, code in results.items():
        status = "✅ 成功" if code == 0 else f"❌ 失败 ({code})"
        print(f"{dataset:<20} {status:>10}")
    
    print("=" * 70)
    
    # 统计
    success_count = sum(1 for code in results.values() if code == 0)
    print(f"成功：{success_count}/{len(DATASETS)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
