#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 W2NER 和 W2NER_folded 数据中的 ID 冲突：
1. 同一个 doc_id 下是否有多个 sent_id=0
2. sample_id 是否重复
"""

import json
from pathlib import Path
from collections import defaultdict

W2NER_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")
W2NER_FOLDED_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner_folded")

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

SPLITS = ["train", "dev", "test"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_ids(data, data_name):
    """检查数据中的 ID 冲突"""
    issues = []
    
    # 检查 sample_id 重复
    sample_id_counts = defaultdict(int)
    for sample in data:
        sample_id = sample.get("sample_id", "unknown")
        sample_id_counts[sample_id] += 1
    
    duplicate_sample_ids = {k: v for k, v in sample_id_counts.items() if v > 1}
    if duplicate_sample_ids:
        issues.append(f"sample_id 重复：{len(duplicate_sample_ids)} 个重复 ID")
        for sid, count in list(duplicate_sample_ids.items())[:5]:
            issues.append(f"    - {sid} (出现 {count} 次)")
    
    # 检查 (doc_id, sent_id) 组合重复
    doc_sent_counts = defaultdict(int)
    for sample in data:
        doc_id = sample.get("doc_id", "unknown")
        sent_id = sample.get("sent_id", -1)
        key = (doc_id, sent_id)
        doc_sent_counts[key] += 1
    
    duplicate_doc_sents = {k: v for k, v in doc_sent_counts.items() if v > 1}
    if duplicate_doc_sents:
        issues.append(f"(doc_id, sent_id) 组合重复：{len(duplicate_doc_sents)} 个重复组合")
        for (doc_id, sent_id), count in list(duplicate_doc_sents.items())[:5]:
            issues.append(f"    - doc={doc_id}, sent_id={sent_id} (出现 {count} 次)")
    
    # 检查每个 doc_id 下 sent_id=0 的数量
    doc_zero_sent_counts = defaultdict(int)
    for sample in data:
        doc_id = sample.get("doc_id", "unknown")
        sent_id = sample.get("sent_id", -1)
        if sent_id == 0:
            doc_zero_sent_counts[doc_id] += 1
    
    multi_zero_sent = {k: v for k, v in doc_zero_sent_counts.items() if v > 1}
    if multi_zero_sent:
        issues.append(f"同一个 doc_id 下有多个 sent_id=0: {len(multi_zero_sent)} 个 doc")
        for doc_id, count in list(multi_zero_sent.items())[:5]:
            issues.append(f"    - {doc_id} (有 {count} 个 sent_id=0)")
    
    # 检查 sent_id 是否连续
    doc_sent_ids = defaultdict(list)
    for sample in data:
        doc_id = sample.get("doc_id", "unknown")
        sent_id = sample.get("sent_id", -1)
        doc_sent_ids[doc_id].append(sent_id)
    
    non_continuous_docs = 0
    for doc_id, sent_ids in doc_sent_ids.items():
        sent_ids_sorted = sorted(sent_ids)
        expected = list(range(len(sent_ids_sorted)))
        if sent_ids_sorted != expected:
            non_continuous_docs += 1
    
    if non_continuous_docs > 0:
        issues.append(f"sent_id 不连续：{non_continuous_docs} 个 doc")
    
    return issues


def main():
    print("=" * 80)
    print("检查 W2NER 和 W2NER_folded 数据中的 ID 冲突")
    print("=" * 80)
    
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"【{dataset}】")
        print("=" * 80)
        
        for split in SPLITS:
            print(f"\n  --- {split}.json ---")
            
            # 检查 W2NER (data_w2ner_folded_with_dev)
            w2ner_path = W2NER_PATH / dataset / f"{split}.json"
            if w2ner_path.exists():
                w2ner_data = load_json(w2ner_path)
                w2ner_issues = check_ids(w2ner_data, "W2NER")
                
                if w2ner_issues:
                    print(f"  ❌ W2NER/data_w2ner_folded_with_dev:")
                    for issue in w2ner_issues[:10]:
                        print(f"      {issue}")
                else:
                    print(f"  ✅ W2NER/data_w2ner_folded_with_dev: 无 ID 冲突")
            
            # 检查 W2NER_folded (data_w2ner_folded)
            folded_path = W2NER_FOLDED_PATH / dataset / f"{split}.json"
            if folded_path.exists():
                folded_data = load_json(folded_path)
                folded_issues = check_ids(folded_data, "W2NER_folded")
                
                if folded_issues:
                    print(f"  ❌ data_w2ner_folded:")
                    for issue in folded_issues[:10]:
                        print(f"      {issue}")
                else:
                    print(f"  ✅ data_w2ner_folded: 无 ID 冲突")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
