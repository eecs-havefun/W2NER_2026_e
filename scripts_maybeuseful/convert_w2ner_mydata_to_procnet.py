#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量转换 W2NER 原始数据到 ProcNET sidecar 格式（带句子长度过滤）
源目录：W2NER/data/mydata/{train,dev,test}.json
目标目录：procnet/sidecar/{train,dev,test}_doc_typed_entities.jsonl
过滤规则：丢弃任何包含句子长度>450 的文档
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 配置
INPUT_DIR = Path("W2NER/data/mydata")
OUTPUT_DIR = Path("procnet/sidecar")
SPLITS = ["train", "dev", "test"]
MAX_SENTENCE_LENGTH = 450

# W2NER 实体类型 -> ProcNET 类型映射
TYPE_MAP = {
    "StockCode": ("StockCode", 1),
    "StockAbbr": ("StockAbbr", 2),
    "StockFullName": ("StockFullName", 3),
    "CompanyName": ("CompanyName", 3),
    "CompanyAbbr": ("CompanyAbbr", 3),
    "PersonName": ("PersonName", 4),
    "PersonTitle": ("PersonTitle", 5),
    "Date": ("Date", 6),
    "Time": ("Time", 6),
    "Organization": ("Organization", 7),
    "Institution": ("Institution", 7),
    "Location": ("Location", 8),
    "Address": ("Address", 8),
    "Money": ("Money", 9),
    "Percentage": ("Percentage", 10),
    "Number": ("Number", 11),
    "EquityHolder": ("EquityHolder", 12),
    "FrozeShares": ("FrozeShares", 13),
    "LegalInstitution": ("LegalInstitution", 14),
    "Pledger": ("Pledger", 15),
    "PledgedShares": ("PledgedShares", 16),
    "Pledgee": ("Pledgee", 17),
    "TotalHoldingShares": ("TotalHoldingShares", 18),
    "TotalHoldingRatio": ("TotalHoldingRatio", 19),
    "TotalPledgedShares": ("TotalPledgedShares", 20),
    "StartDate": ("StartDate", 21),
    "EndDate": ("EndDate", 22),
    "ReleasedDate": ("ReleasedDate", 23),
    "UnfrozeDate": ("UnfrozeDate", 24),
    "EquityRepurchase": ("EquityRepurchase", 25),
    "RepurchasedShares": ("RepurchasedShares", 26),
    "ClosingDate": ("ClosingDate", 27),
    "RepurchaseAmount": ("RepurchaseAmount", 28),
    "TradedShares": ("TradedShares", 29),
    "LaterHoldingShares": ("LaterHoldingShares", 30),
    "AveragePrice": ("AveragePrice", 31),
    "HighestTradingPrice": ("HighestTradingPrice", 32),
    "LowestTradingPrice": ("LowestTradingPrice", 33),
}


def group_by_doc(data: List[Dict]) -> Dict[str, List[Dict]]:
    """将句子列表按 doc_id 分组"""
    docs: Dict[str, List[Dict]] = defaultdict(list)
    for sent in data:
        docs[sent["doc_id"]].append(sent)
    # 每篇文档内按 sent_id 排序
    for doc_id in docs:
        docs[doc_id].sort(key=lambda x: x["sent_id"])
    return docs


def filter_docs_by_sentence_length(
    docs: Dict[str, List[Dict]], 
    max_len: int = MAX_SENTENCE_LENGTH
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """
    过滤掉包含句子长度>max_len 的文档
    返回：(过滤后的文档，被移除的文档信息)
    """
    filtered = {}
    removed = []
    
    for doc_id, sentences in docs.items():
        max_sent_len = max(len(s.get("sentence", [])) for s in sentences)
        if max_sent_len > max_len:
            removed.append({
                "doc_id": doc_id,
                "num_sentences": len(sentences),
                "max_sentence_length": max_sent_len,
                "reason": f"max_sentence_length>{max_len}"
            })
        else:
            filtered[doc_id] = sentences
    
    return filtered, removed


def convert_doc_to_procnet(doc_id: str, sentences: List[Dict]) -> Dict:
    """将一篇文档转换为 ProcNET 格式"""
    # 构建句子列表
    procnet_sents = []
    for sent in sentences:
        chars = sent.get("sentence", [])
        procnet_sents.append({
            "sent_id": sent["sent_id"],
            "sentence": chars,
            "text": "".join(chars)
        })
    
    # 构建实体列表
    entities = []
    for sent in sentences:
        chars = sent.get("sentence", [])
        for ner in sent.get("ner", []):
            indices = sorted(ner.get("index", []))
            if not indices:
                continue
            b, e = indices[0], indices[-1] + 1
            w2ner_type = ner.get("type", "Unknown")
            type_name, type_id = TYPE_MAP.get(w2ner_type, (w2ner_type, 99))
            key = f"{doc_id}:{sent['sent_id']}:{b}:{e}:{type_id}"
            entities.append({
                "key": key,
                "cluster_key": key,
                "doc_id": doc_id,
                "sent_id": sent["sent_id"],
                "token_indices": indices,
                "b": b,
                "e": e,
                "type_id": type_id,
                "type": type_name,
                "score": 0.95,
                "head": b,
                "text": "".join(chars[b:e]),
                "source": "w2ner"
            })
    
    return {
        "doc_id": doc_id,
        "sentences": procnet_sents,
        "typed_entities": entities,
        "num_sentences": len(procnet_sents),
        "num_typed_entities": len(entities)
    }


def convert_split(split: str) -> Tuple[int, int, List[Dict]]:
    """
    转换一个数据集
    返回：(文档数，实体数，被移除的文档列表)
    """
    input_file = INPUT_DIR / f"{split}.json"
    output_file = OUTPUT_DIR / f"{split}_doc_typed_entities.jsonl"
    removed_file = OUTPUT_DIR / f"{split}_removed_docs.json"
    
    print(f"\n读取：{input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  总句子数：{len(data)}")
    
    # 按文档分组
    docs = group_by_doc(data)
    print(f"  总文档数：{len(docs)}")
    
    # 过滤
    filtered, removed = filter_docs_by_sentence_length(docs)
    print(f"  过滤后文档数：{len(filtered)} (移除 {len(removed)} 篇)")
    
    # 转换并输出
    total_entities = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for doc_id, sentences in filtered.items():
            procnet_doc = convert_doc_to_procnet(doc_id, sentences)
            f.write(json.dumps(procnet_doc, ensure_ascii=False) + "\n")
            total_entities += procnet_doc["num_typed_entities"]
    
    # 保存被移除的文档信息
    with open(removed_file, "w", encoding="utf-8") as f:
        json.dump(removed, f, ensure_ascii=False, indent=2)
    
    print(f"  输出：{output_file} ({len(filtered)} 文档，{total_entities} 实体)")
    print(f"  移除列表：{removed_file}")
    
    return len(filtered), total_entities, removed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("W2NER (mydata) -> ProcNET sidecar 全量转换")
    print(f"过滤规则：句子长度 > {MAX_SENTENCE_LENGTH} 的文档将被丢弃")
    print("=" * 70)
    
    stats = []
    total_removed = 0
    
    for split in SPLITS:
        docs, ents, removed = convert_split(split)
        stats.append((split, docs, ents, len(removed)))
        total_removed += len(removed)
    
    print("\n" + "=" * 70)
    print("转换完成!")
    print("-" * 70)
    print(f"{'数据集':<10} {'原文档':>10} {'过滤后':>10} {'移除':>10} {'实体数':>12}")
    print("-" * 70)
    for split, docs, ents, removed in stats:
        original = docs + removed
        print(f"{split:<10} {original:>10} {docs:>10} {removed:>10} {ents:>12}")
    print("-" * 70)
    print(f"{'总计':<10} {'':>10} {sum(s[1] for s in stats):>10} {total_removed:>10} {sum(s[2] for s in stats):>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
