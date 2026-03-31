#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入检查 dev/test 文本重复的原因：
- 检查是否来自同一文档的不同句子（正常）
- 检查是否是真正的数据泄露
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("深入检查 dev/test 文本重复原因")
    print("=" * 70)
    
    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"数据集：{dataset}")
        print("-" * 70)
        
        dev_path = DATA_ROOT / dataset / "dev.json"
        test_path = DATA_ROOT / dataset / "test.json"
        
        dev_data = load_json(dev_path)
        test_data = load_json(test_path)
        
        # 按文本分组
        dev_by_text = defaultdict(list)
        test_by_text = defaultdict(list)
        
        for i, sample in enumerate(dev_data):
            text = sample.get("text", "")
            doc_id = sample.get("doc_id", "unknown")
            dev_by_text[text].append((i, doc_id, sample.get("sent_id", -1)))
        
        for i, sample in enumerate(test_data):
            text = sample.get("text", "")
            doc_id = sample.get("doc_id", "unknown")
            test_by_text[text].append((i, doc_id, sample.get("sent_id", -1)))
        
        # 找出重复文本
        repeated_texts = set(dev_by_text.keys()) & set(test_by_text.keys())
        
        if not repeated_texts:
            print("  ✓ 无重复文本")
            continue
        
        print(f"  发现 {len(repeated_texts)} 条重复文本")
        
        # 分析重复原因
        same_doc_count = 0
        diff_doc_count = 0
        
        for text in list(repeated_texts)[:10]:  # 只分析前 10 个
            dev_samples = dev_by_text[text]
            test_samples = test_by_text[text]
            
            dev_docs = set(doc_id for _, doc_id, _ in dev_samples)
            test_docs = set(doc_id for _, doc_id, _ in test_samples)
            
            if dev_docs & test_docs:
                same_doc_count += 1
                print(f"  [同文档] 文本：{text[:50]}...")
                print(f"         dev docs: {dev_docs}, test docs: {test_docs}")
            else:
                diff_doc_count += 1
                print(f"  [跨文档] 文本：{text[:50]}...")
                print(f"         dev docs: {dev_docs}, test docs: {test_docs}")
        
        if len(repeated_texts) > 10:
            print(f"  ... 还有 {len(repeated_texts) - 10} 条未显示")
        
        print(f"\n  总结：同文档重复={same_doc_count}, 跨文档重复={diff_doc_count}")
        
        if diff_doc_count > 0:
            print(f"  ⚠️  发现跨文档数据泄露!")
        else:
            print(f"  ✓ 重复文本均来自同一文档的不同句子（正常）")


if __name__ == "__main__":
    main()
