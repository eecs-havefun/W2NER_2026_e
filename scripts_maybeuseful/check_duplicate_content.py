
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入检查重复样本：它们的内容是否相同？
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

W2NER_PATH = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

DATASET = "flight_orders_with_queries"
SPLIT = "train"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_to_hash(sample):
    """将样本内容转换为哈希（用于比较内容是否相同）"""
    # 只比较关键内容
    content = {
        "doc_id": sample.get("doc_id"),
        "sent_id": sample.get("sent_id"),
        "text": sample.get("text"),
        "sentence": sample.get("sentence"),
        "ner": sample.get("ner"),
    }
    return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()


def main():
    path = W2NER_PATH / DATASET / f"{SPLIT}.json"
    data = load_json(path)
    
    print(f"检查 {DATASET}/{SPLIT}.json")
    print(f"总样本数：{len(data)}")
    
    # 按 (doc_id, sent_id) 分组
    groups = defaultdict(list)
    for i, sample in enumerate(data):
        doc_id = sample.get("doc_id", "unknown")
        sent_id = sample.get("sent_id", -1)
        key = (doc_id, sent_id)
        groups[key].append((i, sample))
    
    # 找出重复的 (doc_id, sent_id)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    
    print(f"\n重复的 (doc_id, sent_id) 组合：{len(duplicates)} 个")
    
    # 检查前 5 个重复组
    for i, ((doc_id, sent_id), samples) in enumerate(list(duplicates.items())[:5]):
        print(f"\n  重复组 {i+1}: doc={doc_id}, sent_id={sent_id}")
        
        # 检查内容是否相同
        hashes = [sample_to_hash(s) for _, s in samples]
        
        if len(set(hashes)) == 1:
            print(f"    ✅ 内容完全相同（{len(samples)} 份复制）")
        else:
            print(f"    ❌ 内容不同！")
            for idx, (sample_idx, sample) in enumerate(samples):
                print(f"      样本 {idx+1} (索引={sample_idx}):")
                print(f"        text: {sample.get('text', '')[:50]}...")
                print(f"        ner: {len(sample.get('ner', []))} 实体")
                print(f"        hash: {hashes[idx][:16]}...")
    
    # 统计有多少重复是内容相同的
    same_content_count = 0
    diff_content_count = 0
    
    for (doc_id, sent_id), samples in duplicates.items():
        hashes = [sample_to_hash(s) for _, s in samples]
        if len(set(hashes)) == 1:
            same_content_count += 1
        else:
            diff_content_count += 1
    
    print(f"\n{'='*60}")
    print(f"总结:")
    print(f"  内容相同的重复：{same_content_count} 组 ({same_content_count/len(duplicates)*100:.1f}%)")
    print(f"  内容不同的重复：{diff_content_count} 组 ({diff_content_count/len(duplicates)*100:.1f}%)")
    
    if same_content_count > 0:
        print(f"\n  → 这些是简单的数据复制，可以安全去重")
    if diff_content_count > 0:
        print(f"\n  → 这些是真正的 ID 冲突，需要修复")


if __name__ == "__main__":
    main()
