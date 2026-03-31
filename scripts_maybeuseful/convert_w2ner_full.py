
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量转换 W2NER 数据到 ProcNET sidecar 格式
源目录：W2NER/data/mydata_doc_drop450/{train,dev,test}.json
目标目录：procnet/tmp_sidecar/{train,dev,test}_doc_typed_entities.jsonl
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# 配置
INPUT_DIR = Path("W2NER/data/mydata_doc_drop450")
OUTPUT_DIR = Path("procnet/tmp_sidecar")
SPLITS = ["train", "dev", "test"]

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


def convert_file(input_path: Path, output_path: Path) -> tuple:
    """转换单个文件，返回 (文档数，实体数)"""
    print(f"读取：{input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 按 doc_id 分组
    docs: Dict[str, List] = defaultdict(list)
    for sent in data:
        docs[sent["doc_id"]].append(sent)
    
    # 排序句子
    for doc_id in docs:
        docs[doc_id].sort(key=lambda x: x["sent_id"])
    
    total_docs = 0
    total_entities = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, sentences in docs.items():
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
            
            doc = {
                "doc_id": doc_id,
                "sentences": procnet_sents,
                "typed_entities": entities,
                "num_sentences": len(procnet_sents),
                "num_typed_entities": len(entities)
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_docs += 1
            total_entities += len(entities)
    
    print(f"输出：{output_path} ({total_docs} 文档，{total_entities} 实体)")
    return total_docs, total_entities


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("W2NER -> ProcNET 全量转换")
    print("=" * 60)
    
    stats = []
    for split in SPLITS:
        input_file = INPUT_DIR / f"{split}.json"
        output_file = OUTPUT_DIR / f"{split}_doc_typed_entities.jsonl"
        docs, ents = convert_file(input_file, output_file)
        stats.append((split, docs, ents))
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("-" * 60)
    print(f"{'数据集':<10} {'文档数':>10} {'实体数':>10}")
    print("-" * 60)
    for split, docs, ents in stats:
        print(f"{split:<10} {docs:>10} {ents:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
