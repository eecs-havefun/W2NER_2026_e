
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查重复文本的实体标注是否一致
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_ROOT = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

DATASETS = [
    "flight_orders_with_queries",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_entity_signature(sample):
    """获取实体的签名（类型 + 位置）"""
    entities = sample.get("ner", [])
    sigs = []
    for ent in entities:
        indices = tuple(ent.get("index", []))
        etype = ent.get("type", "")
        sigs.append((indices, etype))
    return tuple(sorted(sigs))


def main():
    print("=" * 70)
    print("检查重复文本的实体标注")
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
            dev_by_text[text].append({
                "idx": i,
                "doc_id": sample.get("doc_id", "unknown"),
                "sent_id": sample.get("sent_id", -1),
                "entities": sample.get("ner", []),
                "entity_sig": get_entity_signature(sample),
            })
        
        for i, sample in enumerate(test_data):
            text = sample.get("text", "")
            test_by_text[text].append({
                "idx": i,
                "doc_id": sample.get("doc_id", "unknown"),
                "sent_id": sample.get("sent_id", -1),
                "entities": sample.get("ner", []),
                "entity_sig": get_entity_signature(sample),
            })
        
        # 找出重复文本
        repeated_texts = set(dev_by_text.keys()) & set(test_by_text.keys())
        
        if not repeated_texts:
            print("  ✓ 无重复文本")
            return
        
        print(f"  发现 {len(repeated_texts)} 条重复文本\n")
        
        # 分析前 5 个重复文本的实体标注
        for text in list(repeated_texts)[:5]:
            dev_samples = dev_by_text[text]
            test_samples = test_by_text[text]
            
            print(f"  文本：{text[:80]}...")
            print(f"  长度：{len(text)} 字符")
            
            print(f"    dev 样本 ({len(dev_samples)} 个):")
            for s in dev_samples[:3]:
                print(f"      - {s['doc_id']}__sent_{s['sent_id']}: {len(s['entities'])} 实体")
                if s['entities']:
                    print(f"        实体：{[(e['type'], e['index']) for e in s['entities']][:3]}")
            
            print(f"    test 样本 ({len(test_samples)} 个):")
            for s in test_samples[:3]:
                print(f"      - {s['doc_id']}__sent_{s['sent_id']}: {len(s['entities'])} 实体")
                if s['entities']:
                    print(f"        实体：{[(e['type'], e['index']) for e in s['entities']][:3]}")
            
            # 检查实体签名是否相同
            dev_sigs = set(s['entity_sig'] for s in dev_samples)
            test_sigs = set(s['entity_sig'] for s in test_samples)
            
            if dev_sigs == test_sigs and len(dev_sigs) == 1 and dev_sigs == {()}:
                print(f"    → 所有样本都无实体标注（模板文本）")
            elif dev_sigs & test_sigs:
                print(f"    → 有相同的实体标注模式")
            else:
                print(f"    → 实体标注不同：dev 有 {len(dev_sigs)} 种模式，test 有 {len(test_sigs)} 种模式")
            
            print()


if __name__ == "__main__":
    main()
