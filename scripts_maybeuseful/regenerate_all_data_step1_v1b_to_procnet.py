#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 data_v1b (Rasa NLU 格式) 生成 ProcNet 格式数据

数据流：
data_v1b (Rasa) → procnet/procnet_format

覆盖目标目录：
- procnet/procnet_format/
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# 配置
DATA_V1B_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/data_v1b")
PROCNET_OUTPUT_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format")

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]


def split_into_sentences(text: str) -> List[str]:
    """
    将文本分割成句子
    
    规则：
    - 按句号、问号、感叹号、分号分割
    - 保留标点符号在句子末尾
    - 过滤空句子
    """
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


def compute_sentence_offsets(sentences: List[str], original_text: str) -> List[int]:
    """计算每个句子在原文本中的起始位置"""
    offsets = []
    current_pos = 0
    
    for sent in sentences:
        pos = original_text.find(sent, current_pos)
        if pos != -1:
            offsets.append(pos)
            current_pos = pos + len(sent)
        else:
            offsets.append(current_pos)
            current_pos += len(sent)
    
    return offsets


def map_entity_to_sentences(
    entity_text: str,
    entity_start: int,
    entity_end: int,
    sentences: List[str],
    sentence_offsets: List[int]
) -> List[Tuple[int, int, int]]:
    """
    将实体位置映射到句子中
    
    Returns: [(sent_idx, start_in_sentence, end_in_sentence), ...]
    """
    positions = []
    
    for sent_idx, (sentence, sent_offset) in enumerate(zip(sentences, sentence_offsets)):
        rel_start = entity_start - sent_offset
        rel_end = entity_end - sent_offset
        
        if 0 <= rel_start < len(sentence) and 0 <= rel_end <= len(sentence):
            if sentence[rel_start:rel_end] == entity_text:
                positions.append((sent_idx, rel_start, rel_end))
    
    return positions


def convert_rasa_example_to_procnet(
    example: Dict[str, Any],
    doc_id: str
) -> List[Dict[str, Any]]:
    """
    将单个 Rasa example 转换为 ProcNet 文档格式
    
    每个 Rasa example 可能包含多个句子，转换为一个 ProcNet 文档
    """
    text = example.get('text', '')
    intent = example.get('intent', 'unknown')
    entities = example.get('entities', [])
    
    if not text.strip():
        return []
    
    # 1. 分割句子
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    # 2. 计算句子偏移量
    sentence_offsets = compute_sentence_offsets(sentences, text)
    
    # 3. 处理实体
    ann_valid_mspans = []
    ann_valid_dranges = []
    ann_mspan2dranges = defaultdict(list)
    ann_mspan2guess_field = {}
    
    processed_spans = set()
    
    for entity in entities:
        entity_text = entity.get('value', '')
        entity_type = entity.get('entity', 'unknown')
        entity_start = entity.get('start', -1)
        entity_end = entity.get('end', -1)
        
        if not entity_text or entity_start < 0 or entity_end < 0:
            continue
        
        # 找到实体在句子中的位置
        positions = map_entity_to_sentences(
            entity_text, entity_start, entity_end,
            sentences, sentence_offsets
        )
        
        for sent_idx, start, end in positions:
            unique_key = f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"
            
            span_key = (sent_idx, start, end, entity_type)
            if span_key in processed_spans:
                continue
            processed_spans.add(span_key)
            
            ann_valid_mspans.append(entity_text)
            
            drange = [sent_idx, start, end]
            ann_mspan2dranges[unique_key].append(drange)
            ann_valid_dranges.append(drange)
            
            ann_mspan2guess_field[unique_key] = entity_type
    
    # 4. 构建事件
    event_type = intent
    event_dict = {}
    for entity in entities:
        entity_text = entity.get('value', '')
        entity_type = entity.get('entity', 'unknown')
        if entity_text:
            event_dict[entity_type] = entity_text
    
    recguid_eventname_eventdict_list = []
    if event_dict:
        recguid_eventname_eventdict_list.append([0, event_type, event_dict])
    
    # 5. 构建 ProcNet 文档
    procnet_doc = {
        'sentences': sentences,
        'ann_valid_mspans': ann_valid_mspans,
        'ann_valid_dranges': ann_valid_dranges,
        'ann_mspan2dranges': dict(ann_mspan2dranges),
        'ann_mspan2guess_field': ann_mspan2guess_field,
        'recguid_eventname_eventdict_list': recguid_eventname_eventdict_list
    }
    
    return [[doc_id, procnet_doc]]


def load_rasa_data(path: Path) -> List[Dict[str, Any]]:
    """加载 Rasa NLU 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("rasa_nlu_data", {}).get("common_examples", [])


def save_procnet_data(docs: List, path: Path):
    """保存 ProcNet 格式数据"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def convert_dataset(dataset: str):
    """转换单个数据集"""
    print(f"\n处理：{dataset}")
    print("-" * 60)
    
    stats = {"train": 0, "dev": 0, "test": 0}
    
    for split in ["train", "dev", "test"]:
        input_path = DATA_V1B_ROOT / dataset / f"{split}.json"
        output_path = PROCNET_OUTPUT_ROOT / dataset / f"{split}.json"
        
        if not input_path.exists():
            print(f"  ⚠️  {split}: 输入文件不存在")
            continue
        
        # 加载 Rasa 数据
        rasa_examples = load_rasa_data(input_path)
        
        # 转换每个 example
        procnet_docs = []
        doc_id_counter = 0
        
        for example in rasa_examples:
            doc_id = f"doc_{doc_id_counter:06d}"
            docs = convert_rasa_example_to_procnet(example, doc_id)
            procnet_docs.extend(docs)
            doc_id_counter += 1
        
        # 保存
        save_procnet_data(procnet_docs, output_path)
        
        # 统计
        total_sentences = sum(len(doc[1].get('sentences', [])) for doc in procnet_docs)
        total_entities = sum(len(doc[1].get('ann_valid_dranges', [])) for doc in procnet_docs)
        
        stats[split] = len(procnet_docs)
        
        print(f"  {split}: {len(procnet_docs)} 文档，{total_sentences} 句子，{total_entities} 实体")
    
    print(f"  总计：{sum(stats.values())} 文档")
    return stats


def main():
    print("=" * 80)
    print("从 data_v1b 生成 ProcNet 格式数据")
    print("=" * 80)
    
    all_stats = {}
    
    for dataset in DATASETS:
        all_stats[dataset] = convert_dataset(dataset)
    
    print("\n" + "=" * 80)
    print("生成完成!")
    print("=" * 80)
    
    # 打印汇总
    print("\n汇总:")
    print(f"{'数据集':<35} {'train':>10} {'dev':>10} {'test':>10}")
    print("-" * 80)
    for dataset, stats in all_stats.items():
        print(f"{dataset:<35} {stats['train']:>10} {stats['dev']:>10} {stats['test']:>10}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
