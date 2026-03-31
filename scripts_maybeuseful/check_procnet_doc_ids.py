#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 ProcNet 原始数据中的 doc_id 分配
"""

import json
from pathlib import Path

PROCNET_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format/flight_orders_with_queries")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("检查 ProcNet 原始数据\n")
    
    for split in ["train", "dev", "test"]:
        path = PROCNET_PATH / f"{split}.json"
        if not path.exists():
            continue
        
        data = load_json(path)
        
        print(f"=== {split}.json ===")
        print(f"文档数：{len(data)}")
        
        # 检查 doc_id 是否唯一
        doc_ids = [doc[0] for doc in data]
        unique_doc_ids = set(doc_ids)
        
        print(f"唯一 doc_id 数：{len(unique_doc_ids)}")
        
        if len(doc_ids) != len(unique_doc_ids):
            from collections import Counter
            dup_doc_ids = {k: v for k, v in Counter(doc_ids).items() if v > 1}
            print(f"重复的 doc_id: {len(dup_doc_ids)} 个")
            for doc_id, count in list(dup_doc_ids.items())[:5]:
                print(f"  - {doc_id} (出现 {count} 次)")
        
        # 检查每个文档的句子数
        sent_counts = [len(doc[1].get("sentences", [])) for doc in data]
        print(f"句子数统计：min={min(sent_counts)}, max={max(sent_counts)}, avg={sum(sent_counts)/len(sent_counts):.1f}")
        
        # 显示前 3 个文档
        print(f"\n前 3 个文档:")
        for i, doc in enumerate(data[:3]):
            doc_id = doc[0]
            sentences = doc[1].get("sentences", [])
            print(f"  {i+1}. {doc_id}: {len(sentences)} 句")
            for j, sent in enumerate(sentences[:2]):
                print(f"      [{j}] {sent[:60]}...")
        
        print()
    
    # 检查 W2NER 数据
    print("\n=== W2NER 数据 ===")
    w2ner_path = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev/flight_orders_with_queries/train.json")
    w2ner_data = load_json(w2ner_path)
    
    print(f"样本数：{len(w2ner_data)}")
    
    # 检查 doc_id 分布
    from collections import defaultdict, Counter
    doc_sent_map = defaultdict(list)
    for sample in w2ner_data:
        doc_id = sample.get("doc_id")
        sent_id = sample.get("sent_id")
        doc_sent_map[doc_id].append((sent_id, sample.get("text", "")[:40]))
    
    # 显示有冲突的 doc
    print(f"\n有 ID 冲突的 doc 示例:")
    conflict_count = 0
    for doc_id, sent_list in doc_sent_map.items():
        sent_id_counts = Counter(s[0] for s in sent_list)
        if any(c > 1 for c in sent_id_counts.values()):
            conflict_count += 1
            if conflict_count <= 3:
                print(f"  {doc_id}:")
                for sent_id, text in sent_list[:5]:
                    print(f"    sent_id={sent_id}: {text}...")
    
    print(f"\n总冲突 doc 数：{conflict_count}")


if __name__ == "__main__":
    main()
