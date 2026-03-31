
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ProcNet 格式生成 W2NER 格式数据

数据流：
procnet/procnet_format → W2NER/data/data_w2ner_folded_with_dev

覆盖目标目录：
- W2NER/data/data_w2ner_folded_with_dev/
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# 配置
PROCNET_INPUT_ROOT = project_root / "procnet" / "procnet_format"
W2NER_OUTPUT_ROOT = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

# 角色折叠映射（用于折叠版本）
ROLE_FOLD_MAP = {
    "startDate": "date",
    "endDate": "date",
    "startTime": "time",
    "endTime": "time",
}


def load_procnet_data(path: Path) -> List:
    """加载 ProcNet 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_w2ner_data(data: List, path: Path):
    """保存 W2NER 格式数据"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recover_entity_text_from_mspan_key(mspan_key: str) -> str:
    """从 ProcNet mspan key 恢复实体文本"""
    if not isinstance(mspan_key, str):
        return str(mspan_key)
    if "#" in mspan_key:
        return mspan_key.split("#", 1)[0]
    return mspan_key


def convert_procnet_to_w2ner(
    procnet_docs: List,
    fold_role_types: bool = False
) -> List[Dict[str, Any]]:
    """
    将 ProcNet 文档转换为 W2NER 句子级别格式
    
    Args:
        procnet_docs: ProcNet 格式文档列表
        fold_role_types: 是否折叠角色类型（startDate→date 等）
    
    Returns:
        W2NER 格式样本列表
    """
    w2ner_samples = []
    
    for doc in procnet_docs:
        if not (isinstance(doc, list) and len(doc) >= 2):
            continue
        
        doc_id = doc[0]
        doc_data = doc[1]
        
        sentences = doc_data.get("sentences", [])
        ann_mspan2dranges = doc_data.get("ann_mspan2dranges", {})
        ann_mspan2guess_field = doc_data.get("ann_mspan2guess_field", {})
        
        # 为每个句子生成一个 W2NER 样本
        for sent_idx, sentence in enumerate(sentences):
            # 获取句子文本（可能是字符串或字符列表）
            if isinstance(sentence, list):
                sent_text = "".join(sentence)
                sent_chars = sentence
            else:
                sent_text = sentence
                sent_chars = list(sentence)
            
            # 收集这个句子的所有实体
            ner = []
            entities = []
            
            for mspan_key, dranges in ann_mspan2dranges.items():
                orig_type = ann_mspan2guess_field.get(mspan_key)
                if orig_type is None:
                    continue
                
                # 折叠角色类型
                if fold_role_types:
                    entity_type = ROLE_FOLD_MAP.get(orig_type, orig_type)
                else:
                    entity_type = orig_type
                
                entity_text = recover_entity_text_from_mspan_key(mspan_key)
                
                for dr in dranges:
                    if not (isinstance(dr, list) and len(dr) == 3):
                        continue
                    
                    dr_sent_idx, start, end = dr
                    
                    if dr_sent_idx != sent_idx:
                        continue
                    
                    # 验证实体位置
                    if not (0 <= start < end <= len(sent_text)):
                        continue
                    
                    if sent_text[start:end] != entity_text:
                        continue
                    
                    # 添加实体
                    indices = list(range(start, end))
                    ner.append({
                        "index": indices,
                        "type": entity_type,
                    })
                    
                    entities.append({
                        "start": start,
                        "end": end,
                        "text": entity_text,
                        "type_name": entity_type,
                        "orig_type_name": orig_type,
                        "mspan_key": mspan_key,
                        "doc_id": doc_id,
                        "sent_idx": sent_idx,
                    })
            
            # 构建样本
            sample = {
                "sample_id": f"{doc_id}__sent_{sent_idx}",
                "doc_id": doc_id,
                "sent_id": sent_idx,
                "text": sent_text,
                "sentence": sent_chars,
                "ner": ner,
                "entities": entities,
            }
            
            w2ner_samples.append(sample)
    
    # 按 sample_id 排序
    w2ner_samples.sort(key=lambda x: x["sample_id"])
    
    return w2ner_samples


def split_train_dev_test(
    all_samples: List[Dict],
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    将样本划分为 train/dev/test
    
    按文档分组后划分，确保同一文档的句子在同一集合中
    """
    import random
    random.seed(seed)
    
    # 按文档分组
    docs = defaultdict(list)
    for sample in all_samples:
        doc_id = sample.get("doc_id")
        docs[doc_id].append(sample)
    
    # 获取文档列表并打乱
    doc_ids = list(docs.keys())
    random.shuffle(doc_ids)
    
    # 计算划分点
    n_docs = len(doc_ids)
    dev_n = max(1, int(n_docs * dev_ratio))
    test_n = max(1, int(n_docs * test_ratio))
    
    # 划分文档
    test_doc_ids = doc_ids[:test_n]
    dev_doc_ids = doc_ids[test_n:test_n + dev_n]
    train_doc_ids = doc_ids[test_n + dev_n:]
    
    # 收集样本
    train_samples = []
    for doc_id in train_doc_ids:
        train_samples.extend(docs[doc_id])
    
    dev_samples = []
    for doc_id in dev_doc_ids:
        dev_samples.extend(docs[doc_id])
    
    test_samples = []
    for doc_id in test_doc_ids:
        test_samples.extend(docs[doc_id])
    
    # 排序
    train_samples.sort(key=lambda x: x["sample_id"])
    dev_samples.sort(key=lambda x: x["sample_id"])
    test_samples.sort(key=lambda x: x["sample_id"])
    
    return train_samples, dev_samples, test_samples


def convert_dataset(dataset: str):
    """转换单个数据集"""
    print(f"\n处理：{dataset}")
    print("-" * 60)
    
    # 加载 ProcNet 数据
    train_path = PROCNET_INPUT_ROOT / dataset / "train.json"
    test_path = PROCNET_INPUT_ROOT / dataset / "test.json"
    
    if not train_path.exists():
        print(f"  ❌ train.json 不存在")
        return
    
    procnet_train = load_procnet_data(train_path)
    procnet_test = load_procnet_data(test_path) if test_path.exists() else []
    
    # 合并所有文档
    all_procnet_docs = procnet_train + procnet_test
    
    # 转换为 W2NER 格式（不折叠）
    all_w2ner_samples = convert_procnet_to_w2ner(all_procnet_docs, fold_role_types=False)
    
    # 划分 train/dev/test
    train_samples, dev_samples, test_samples = split_train_dev_test(
        all_w2ner_samples,
        dev_ratio=0.15,
        test_ratio=0.15
    )
    
    # 保存
    output_train = W2NER_OUTPUT_ROOT / dataset / "train.json"
    output_dev = W2NER_OUTPUT_ROOT / dataset / "dev.json"
    output_test = W2NER_OUTPUT_ROOT / dataset / "test.json"
    
    save_w2ner_data(train_samples, output_train)
    save_w2ner_data(dev_samples, output_dev)
    save_w2ner_data(test_samples, output_test)
    
    # 统计
    train_entities = sum(len(s.get("ner", [])) for s in train_samples)
    dev_entities = sum(len(s.get("ner", [])) for s in dev_samples)
    test_entities = sum(len(s.get("ner", [])) for s in test_samples)
    
    print(f"  train: {len(train_samples)} 句子，{train_entities} 实体")
    print(f"  dev:   {len(dev_samples)} 句子，{dev_entities} 实体")
    print(f"  test:  {len(test_samples)} 句子，{test_entities} 实体")
    
    return {
        "train": len(train_samples),
        "dev": len(dev_samples),
        "test": len(test_samples),
    }


def main():
    print("=" * 80)
    print("从 ProcNet 生成 W2NER 格式数据 (未折叠)")
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
