#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 ID 冲突：
- 为每个样本分配唯一的 doc_id
- 所有样本的 sent_id 都设为 0（因为每个样本是独立的句子）
- 更新 sample_id
"""

import json
from pathlib import Path

W2NER_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")
W2NER_FOLDED_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner_folded")
W2NER_ALT_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner")

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


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fix_sample_ids(data, start_doc_id=0):
    """
    修复样本 ID：
    - 每个样本分配唯一的 doc_id
    - sent_id 全部设为 0（单句文档）
    - sample_id 重新生成
    """
    fixed = []
    doc_id_counter = start_doc_id
    
    for sample in data:
        new_doc_id = f"doc_{doc_id_counter:06d}"
        new_sent_id = 0  # 每个样本是独立文档，sent_id 都是 0
        
        new_sample = dict(sample)
        new_sample["doc_id"] = new_doc_id
        new_sample["sent_id"] = new_sent_id
        new_sample["sample_id"] = f"{new_doc_id}__sent_{new_sent_id}"
        
        # 更新 entities 中的 doc_id 和 sent_idx
        if "entities" in new_sample:
            for ent in new_sample["entities"]:
                ent["doc_id"] = new_doc_id
                ent["sent_idx"] = new_sent_id
        
        fixed.append(new_sample)
        doc_id_counter += 1
    
    return fixed, doc_id_counter


def verify_no_conflicts(data, data_name):
    """验证没有 ID 冲突"""
    from collections import defaultdict, Counter
    
    # 检查 sample_id 重复
    sample_ids = [s.get("sample_id") for s in data]
    dup_sample_ids = {k: v for k, v in Counter(sample_ids).items() if v > 1}
    
    # 检查 (doc_id, sent_id) 重复
    doc_sent = [(s.get("doc_id"), s.get("sent_id")) for s in data]
    dup_doc_sent = {k: v for k, v in Counter(doc_sent).items() if v > 1}
    
    if dup_sample_ids or dup_doc_sent:
        print(f"  ❌ {data_name}: 仍有冲突!")
        if dup_sample_ids:
            print(f"      sample_id 重复：{len(dup_sample_ids)} 个")
        if dup_doc_sent:
            print(f"      (doc_id, sent_id) 重复：{len(dup_doc_sent)} 个")
        return False
    else:
        print(f"  ✅ {data_name}: 无 ID 冲突")
        return True


def main():
    print("=" * 80)
    print("修复 ID 冲突")
    print("=" * 80)
    
    for dataset in DATASETS:
        print(f"\n【{dataset}】")
        print("-" * 80)
        
        for split in SPLITS:
            # 修复 W2NER/data/data_w2ner_folded_with_dev
            path = W2NER_PATH / dataset / f"{split}.json"
            if path.exists():
                data = load_json(path)
                fixed_data, _ = fix_sample_ids(data)
                dump_json(fixed_data, path)
                verify_no_conflicts(fixed_data, "W2NER/data_w2ner_folded_with_dev")
            
            # 修复 data_w2ner_folded
            path = W2NER_FOLDED_PATH / dataset / f"{split}.json"
            if path.exists():
                data = load_json(path)
                fixed_data, _ = fix_sample_ids(data)
                dump_json(fixed_data, path)
                verify_no_conflicts(fixed_data, "data_w2ner_folded")
            
            # 修复 data_w2ner
            path = W2NER_ALT_PATH / dataset / f"{split}.json"
            if path.exists():
                data = load_json(path)
                fixed_data, _ = fix_sample_ids(data)
                dump_json(fixed_data, path)
                verify_no_conflicts(fixed_data, "data_w2ner")
    
    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
