
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析为什么有大量句子和实体丢失
"""

import json
import re
from pathlib import Path

DATA_V1B = project_root / "data_v1b"
W2NER_PATH = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

DATASET = "flight_orders_with_queries"


def load_rasa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("rasa_nlu_data", {}).get("common_examples", [])


def load_w2ner_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_into_sentences_v1(text):
    """原始转换脚本的句子分割逻辑"""
    pattern = r'([。！？!?；;]+)'
    parts = re.split(pattern, text)
    
    sentences = []
    current_sentence = ""
    
    for part in parts:
        current_sentence += part
        if re.match(pattern, part):
            cleaned = current_sentence.strip()
            if cleaned:
                sentences.append(cleaned)
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences


def main():
    print("=" * 80)
    print("深入分析数据丢失原因")
    print("=" * 80)
    
    # 加载原始数据
    v1b_train = load_rasa_data(DATA_V1B / DATASET / "train.json")
    v1b_test = load_rasa_data(DATA_V1B / DATASET / "test.json")
    
    # 加载 W2NER 数据
    w2ner_train = load_w2ner_data(W2NER_PATH / DATASET / "train.json")
    w2ner_test = load_w2ner_data(W2NER_PATH / DATASET / "test.json")
    
    print(f"\n【{DATASET}】")
    print("-" * 60)
    
    # 1. 检查原始数据的句子分割
    print("\n1. 原始数据句子分割分析")
    
    v1b_all_sentences = []
    for ex in v1b_train:
        text = ex.get("text", "")
        sents = split_into_sentences_v1(text)
        for sent in sents:
            v1b_all_sentences.append({
                "text": sent,
                "entities": []
            })
        
        # 为每个句子分配实体
        for ent in ex.get("entities", []):
            ent_text = ent.get("value", "")
            ent_type = ent.get("entity", "")
            
            # 找到实体所在的句子
            for i, sent in enumerate(sents):
                if ent_text in sent:
                    v1b_all_sentences[-(len(sents)-i)]["entities"].append({
                        "text": ent_text,
                        "type": ent_type
                    })
                    break
    
    print(f"  原始数据 train 分割后：{len(v1b_all_sentences)} 句子")
    
    # 2. 检查 W2NER 数据
    print(f"  W2NER train: {len(w2ner_train)} 句子")
    
    # 3. 找出丢失的句子
    v1b_texts = set(s["text"] for s in v1b_all_sentences)
    w2ner_texts = set(s.get("text", "") for s in w2ner_train)
    
    missing = v1b_texts - w2ner_texts
    
    print(f"\n  丢失的句子数：{len(missing)}")
    
    # 分析丢失句子的特征
    print(f"\n2. 丢失句子分析")
    
    short_missing = [s for s in missing if len(s) < 20]
    long_missing = [s for s in missing if len(s) >= 20]
    
    print(f"  短句 (<20 字符): {len(short_missing)}")
    print(f"  长句 (>=20 字符): {len(long_missing)}")
    
    if short_missing:
        print(f"\n  丢失的短句示例:")
        for s in list(short_missing)[:10]:
            print(f"    - '{s}'")
    
    if long_missing:
        print(f"\n  丢失的长句示例:")
        for s in list(long_missing)[:10]:
            print(f"    - '{s[:80]}...'")
    
    # 4. 检查是否某些原始样本完全丢失
    print(f"\n3. 检查原始样本是否完全丢失")
    
    v1b_full_texts = set(ex.get("text", "") for ex in v1b_train)
    
    # W2NER 样本按 doc_id 重组为完整文本
    w2ner_full_texts = set()
    for sample in w2ner_train:
        w2ner_full_texts.add(sample.get("text", ""))
    
    # 检查原始完整文本是否在 W2NER 中
    missing_full = []
    for v1b_text in v1b_full_texts:
        # 检查是否有任何 W2NER 句子来自这个文本
        found = False
        v1b_sents = split_into_sentences_v1(v1b_text)
        for sent in v1b_sents:
            if sent in w2ner_texts:
                found = True
                break
        if not found:
            missing_full.append(v1b_text)
    
    print(f"  完全丢失的原始样本数：{len(missing_full)}")
    
    if missing_full:
        print(f"\n  完全丢失的样本示例:")
        for text in missing_full[:5]:
            print(f"    - '{text[:80]}...'")
    
    # 5. 检查实体丢失的原因
    print(f"\n4. 实体丢失分析")
    
    # 统计原始实体
    v1b_entity_types = {}
    for ex in v1b_train:
        for ent in ex.get("entities", []):
            etype = ent.get("entity", "")
            v1b_entity_types[etype] = v1b_entity_types.get(etype, 0) + 1
    
    # 统计 W2NER 实体
    w2ner_entity_types = {}
    for sample in w2ner_train:
        for ner in sample.get("ner", []):
            etype = ner.get("type", "")
            w2ner_entity_types[etype] = w2ner_entity_types.get(etype, 0) + 1
    
    print(f"\n  实体类型对比:")
    all_types = set(v1b_entity_types.keys()) | set(w2ner_entity_types.keys())
    
    for etype in sorted(all_types):
        v1b_count = v1b_entity_types.get(etype, 0)
        w2ner_count = w2ner_entity_types.get(etype, 0)
        
        if v1b_count > 0 and w2ner_count == 0:
            print(f"    {etype}: {v1b_count} -> 0 (完全丢失)")
        elif v1b_count > w2ner_count:
            print(f"    {etype}: {v1b_count} -> {w2ner_count} (减少 {v1b_count-w2ner_count})")
        elif w2ner_count > v1b_count:
            print(f"    {etype}: {v1b_count} -> {w2ner_count} (增加 {w2ner_count-v1b_count})")
        else:
            print(f"    {etype}: {v1b_count} -> {w2ner_count} (持平)")
    
    # 6. 结论
    print(f"\n{'='*60}")
    print("结论:")
    
    if len(short_missing) > len(long_missing):
        print("  主要丢失的是短句（可能是模板短信如'退订回 T'等）")
        print("  这些短句可能被视为无价值而被过滤")
    
    if 'startDate' in v1b_entity_types and 'date' in w2ner_entity_types:
        print("  实体类型合并：startDate/endDate -> date (预期行为)")
    
    if len(missing_full) > 0:
        print(f"  ⚠️  有 {len(missing_full)} 个完整样本丢失，需要检查数据转换管道")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
