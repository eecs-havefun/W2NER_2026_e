
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格检查数据对齐：
1. 检查 dev 和 test 是否完全不同（没有重复样本）
2. 检查 train 和 dev/test 是否完全不同
3. 检查各个数据集的样本 ID 是否唯一
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


def get_sample_ids(data):
    """获取所有样本 ID"""
    ids = set()
    for sample in data:
        # 优先使用 sample_id
        if "sample_id" in sample:
            ids.add(sample["sample_id"])
        elif "doc_id" in sample and "sent_id" in sample:
            ids.add(f"{sample['doc_id']}__sent_{sample['sent_id']}")
        else:
            # 使用 text 作为 fallback
            ids.add(hash(sample.get("text", "")))
    return ids


def check_intersection(set1, set2, name1, name2):
    """检查两个集合的交集"""
    intersection = set1 & set2
    if intersection:
        print(f"  ❌ {name1} ∩ {name2} = {len(intersection)} 个重复样本!")
        if len(intersection) <= 5:
            for item in list(intersection)[:5]:
                print(f"     - {item}")
        return len(intersection)
    else:
        print(f"  ✓ {name1} 和 {name2} 无交集")
        return 0


def main():
    print("=" * 70)
    print("数据对齐严格检查")
    print("=" * 70)
    
    total_issues = 0
    
    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"数据集：{dataset}")
        print("-" * 70)
        
        train_path = DATA_ROOT / dataset / "train.json"
        dev_path = DATA_ROOT / dataset / "dev.json"
        test_path = DATA_ROOT / dataset / "test.json"
        
        if not train_path.exists():
            print(f"  ⚠️  {train_path} 不存在，跳过")
            continue
        
        # 加载数据
        train_data = load_json(train_path)
        dev_data = load_json(dev_path)
        test_data = load_json(test_path)
        
        print(f"  样本数：train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
        
        # 获取样本 ID
        train_ids = get_sample_ids(train_data)
        dev_ids = get_sample_ids(dev_data)
        test_ids = get_sample_ids(test_data)
        
        print(f"  唯一 ID: train={len(train_ids)}, dev={len(dev_ids)}, test={len(test_ids)}")
        
        # 检查交集
        issues = 0
        issues += check_intersection(train_ids, dev_ids, "train", "dev")
        issues += check_intersection(train_ids, test_ids, "train", "test")
        issues += check_intersection(dev_ids, test_ids, "dev", "test")
        
        # 检查 dev 和 test 的内容是否完全相同
        dev_texts = set(s.get("text", "") for s in dev_data)
        test_texts = set(s.get("text", "") for s in test_data)
        
        if dev_texts == test_texts and len(dev_texts) > 0:
            print(f"  ❌ dev 和 test 的文本内容完全相同!")
            issues += 1
        elif len(dev_texts & test_texts) > 0:
            overlap = len(dev_texts & test_texts)
            print(f"  ⚠️  dev 和 test 有 {overlap}/{len(dev_texts)} 条文本重复")
            issues += 1
        else:
            print(f"  ✓ dev 和 test 的文本内容完全不同")
        
        if issues == 0:
            print(f"\n  ✅ {dataset}: 数据对齐检查通过")
        else:
            print(f"\n  ❌ {dataset}: 发现 {issues} 个问题")
            total_issues += issues
    
    print("\n" + "=" * 70)
    if total_issues == 0:
        print("✅ 所有数据集对齐检查通过!")
    else:
        print(f"❌ 共发现 {total_issues} 个问题")
    print("=" * 70)


if __name__ == "__main__":
    main()
