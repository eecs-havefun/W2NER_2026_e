
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证：同一个 doc_id 的所有句子都在同一集合中（train/dev/test）
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

W2NER_PATH = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

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
    print("验证：同一个 doc_id 的句子是否在同一集合中")
    print("=" * 80)
    
    all_pass = True
    
    for dataset in DATASETS:
        print(f"\n【{dataset}】")
        print("-" * 60)
        
        # 收集所有 doc_id 及其出现的集合
        doc_to_splits = defaultdict(list)
        
        for split in ["train", "dev", "test"]:
            path = W2NER_PATH / dataset / f"{split}.json"
            if not path.exists():
                continue
            
            data = load_json(path)
            
            # 统计每个 doc_id 有多少句子
            doc_sent_count = defaultdict(int)
            for sample in data:
                doc_id = sample.get("doc_id")
                doc_sent_count[doc_id] += 1
            
            # 记录 doc_id 出现在哪个集合
            for doc_id in doc_sent_count:
                doc_to_splits[doc_id].append((split, doc_sent_count[doc_id]))
        
        # 检查是否有 doc_id 出现在多个集合
        cross_split_docs = {doc: splits for doc, splits in doc_to_splits.items() if len(splits) > 1}
        
        if cross_split_docs:
            print(f"  ❌ 发现 {len(cross_split_docs)} 个 doc_id 跨越多个集合!")
            all_pass = False
            
            # 显示前 5 个例子
            for doc_id, splits in list(cross_split_docs.items())[:5]:
                print(f"    {doc_id}:")
                for split, count in splits:
                    print(f"      {split}: {count} 句子")
        else:
            print(f"  ✅ 所有 doc_id 都只在一个集合中")
        
        # 统计每个集合的 doc_id 数量
        print(f"\n  各集合 doc_id 数量:")
        for split in ["train", "dev", "test"]:
            path = W2NER_PATH / dataset / f"{split}.json"
            if path.exists():
                data = load_json(path)
                unique_docs = len(set(s.get("doc_id") for s in data))
                total_sents = len(data)
                print(f"    {split}: {unique_docs} 个文档，{total_sents} 个句子")
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ 验证通过：没有 doc_id 被拆分到多个集合")
    else:
        print("❌ 验证失败：有 doc_id 被拆分到多个集合")
    print("=" * 80)


if __name__ == "__main__":
    main()
