
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 train/dev/test 分割比例是否为 7:1.5:1.5
"""

import json
from pathlib import Path

DATA_ROOT = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("验证 train/dev/test 分割比例 (目标：7:1.5:1.5)")
    print("=" * 80)
    
    for dataset in DATASETS:
        print(f"\n【{dataset}】")
        print("-" * 80)
        
        train_path = DATA_ROOT / dataset / "train.json"
        dev_path = DATA_ROOT / dataset / "dev.json"
        test_path = DATA_ROOT / dataset / "test.json"
        
        train_data = load_json(train_path)
        dev_data = load_json(dev_path)
        test_data = load_json(test_path)
        
        train_count = len(train_data)
        dev_count = len(dev_data)
        test_count = len(test_data)
        total = train_count + dev_count + test_count
        
        # 计算实际比例
        train_ratio = train_count / total * 100
        dev_ratio = dev_count / total * 100
        test_ratio = test_count / total * 100
        
        # 目标比例：7:1.5:1.5 = 70%:15%:15%
        target_train_ratio = 70
        target_dev_ratio = 15
        target_test_ratio = 15
        
        print(f"  train: {train_count} 样本 ({train_ratio:.1f}%)")
        print(f"  dev:   {dev_count} 样本 ({dev_ratio:.1f}%)")
        print(f"  test:  {test_count} 样本 ({test_ratio:.1f}%)")
        print(f"  总计：{total} 样本")
        
        # 检查比例是否接近目标
        train_ok = abs(train_ratio - target_train_ratio) < 2
        dev_ok = abs(dev_ratio - target_dev_ratio) < 2
        test_ok = abs(test_ratio - target_test_ratio) < 2
        
        if train_ok and dev_ok and test_ok:
            print(f"  ✅ 分割比例符合 7:1.5:1.5 (允许±2% 误差)")
        else:
            print(f"  ⚠️  分割比例与 7:1.5:1.5 有偏差")
            if not train_ok:
                print(f"      train: 实际{train_ratio:.1f}%, 目标{target_train_ratio}%")
            if not dev_ok:
                print(f"      dev: 实际{dev_ratio:.1f}%, 目标{target_dev_ratio}%")
            if not test_ok:
                print(f"      test: 实际{test_ratio:.1f}%, 目标{target_test_ratio}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
