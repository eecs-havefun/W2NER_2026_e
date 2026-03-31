
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终数据对齐检查报告
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

DATA_ROOT = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]


def md5_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sample_ids(data):
    ids = set()
    for sample in data:
        if "sample_id" in sample:
            ids.add(sample["sample_id"])
        elif "doc_id" in sample and "sent_id" in sample:
            ids.add(f"{sample['doc_id']}__sent_{sample['sent_id']}")
    return ids


def main():
    print("=" * 80)
    print(" " * 25 + "最终数据对齐检查报告")
    print("=" * 80)
    
    overall_status = "✅ PASS"
    issues = []
    
    for dataset in DATASETS:
        print(f"\n【{dataset}】")
        print("-" * 80)
        
        train_path = DATA_ROOT / dataset / "train.json"
        dev_path = DATA_ROOT / dataset / "dev.json"
        test_path = DATA_ROOT / dataset / "test.json"
        
        # 1. 文件存在性检查
        if not all(p.exists() for p in [train_path, dev_path, test_path]):
            print(f"  ❌ 文件不完整")
            overall_status = "❌ FAIL"
            issues.append(f"{dataset}: 文件不完整")
            continue
        
        # 2. MD5 检查（dev != test）
        dev_md5 = md5_file(dev_path)
        test_md5 = md5_file(test_path)
        
        if dev_md5 == test_md5:
            print(f"  ❌ dev.json 和 test.json 完全相同 (MD5: {dev_md5[:16]}...)")
            overall_status = "❌ FAIL"
            issues.append(f"{dataset}: dev/test 文件完全相同")
        else:
            print(f"  ✓ dev.json 和 test.json 不同")
        
        # 3. 样本数统计
        train_data = load_json(train_path)
        dev_data = load_json(dev_path)
        test_data = load_json(test_path)
        
        print(f"  样本数：train={len(train_data)}, dev={len(dev_data)}, test={len(test_data)}")
        
        # 4. 样本 ID 交集检查
        train_ids = get_sample_ids(train_data)
        dev_ids = get_sample_ids(dev_data)
        test_ids = get_sample_ids(test_data)
        
        train_dev_overlap = len(train_ids & dev_ids)
        train_test_overlap = len(train_ids & test_ids)
        dev_test_overlap = len(dev_ids & test_ids)
        
        if train_dev_overlap > 0:
            print(f"  ❌ train ∩ dev = {train_dev_overlap} 个重复样本")
            overall_status = "❌ FAIL"
            issues.append(f"{dataset}: train/dev 样本重复")
        else:
            print(f"  ✓ train ∩ dev = ∅")
        
        if train_test_overlap > 0:
            print(f"  ❌ train ∩ test = {train_test_overlap} 个重复样本")
            overall_status = "❌ FAIL"
            issues.append(f"{dataset}: train/test 样本重复")
        else:
            print(f"  ✓ train ∩ test = ∅")
        
        if dev_test_overlap > 0:
            print(f"  ❌ dev ∩ test = {dev_test_overlap} 个重复样本")
            overall_status = "❌ FAIL"
            issues.append(f"{dataset}: dev/test 样本重复")
        else:
            print(f"  ✓ dev ∩ test = ∅")
        
        # 5. 文本重复检查（仅警告，不影响整体）
        dev_texts = set(s.get("text", "") for s in dev_data)
        test_texts = set(s.get("text", "") for s in test_data)
        text_overlap = len(dev_texts & test_texts)
        
        # 检查重复文本是否有实体
        text_overlap_with_entities = 0
        for text in dev_texts & test_texts:
            dev_samples = [s for s in dev_data if s.get("text") == text]
            test_samples = [s for s in test_data if s.get("text") == text]
            has_entity = any(s.get("ner", []) for s in dev_samples + test_samples)
            if has_entity:
                text_overlap_with_entities += 1
        
        if text_overlap_with_entities > 0:
            print(f"  ⚠️  dev/test 有 {text_overlap_with_entities} 条带实体的文本重复")
            issues.append(f"{dataset}: dev/test 有实体文本重复")
        elif text_overlap > 0:
            print(f"  ℹ️  dev/test 有 {text_overlap} 条无实体文本重复（模板短信，可接受）")
        else:
            print(f"  ✓ dev/test 文本内容完全不同")
    
    # 总结
    print("\n" + "=" * 80)
    print(f"总体状态：{overall_status}")
    print("=" * 80)
    
    if issues:
        print("\n问题列表:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 所有检查通过！数据可以用于训练。")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
