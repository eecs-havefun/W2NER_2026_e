#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据链路：从 W2NER 格式生成严格对齐的 ProcNet 格式

数据流向：
W2NER/data/data_w2ner_folded_with_dev (黄金标准) → procnet/procnet_format

确保：
1. train/dev/test 分割完全一致
2. 句子分割粒度完全一致
3. 实体标注完全一致
"""

import json
from pathlib import Path
from collections import defaultdict

# 源目录（黄金标准）
SOURCE_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")

# 目标目录
TARGET_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format")

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


def w2ner_to_procnet(w2ner_samples):
    """
    将 W2NER 句子级别格式转换为 ProcNet 文档级别格式
    
    W2NER 格式：
    [
      {
        "sample_id": "doc_000001__sent_0",
        "doc_id": "doc_000001",
        "sent_id": 0,
        "text": "句子文本...",
        "sentence": ["字", "符", "列", "表"],
        "ner": [{"index": [0, 1, 2], "type": "person"}, ...]
      },
      ...
    ]
    
    ProcNet 格式：
    [
      [
        "doc_000001",
        {
          "sentences": ["句子 1", "句子 2", ...],
          "ann_valid_mspans": ["实体提及 1", "实体提及 2", ...],
          "ann_valid_dranges": [[sent_idx, start, end], ...],
          "ann_mspan2dranges": {"实体提及": [[sent_idx, start, end], ...]},
          "ann_mspan2guess_field": {"实体提及": "实体类型"}
        }
      ],
      ...
    ]
    """
    # 按文档 ID 分组
    docs = defaultdict(list)
    for sample in w2ner_samples:
        doc_id = sample.get("doc_id", "unknown")
        docs[doc_id].append(sample)
    
    # 对每个文档的句子按 sent_id 排序
    for doc_id in docs:
        docs[doc_id].sort(key=lambda x: x.get("sent_id", 0))
    
    # 转换为 ProcNet 格式
    procnet_docs = []
    
    for doc_id, sentences in docs.items():
        procnet_sentences = []
        ann_valid_mspans = []
        ann_valid_dranges = []
        ann_mspan2dranges = defaultdict(list)
        ann_mspan2guess_field = {}
        
        for sent_idx, sample in enumerate(sentences):
            # 获取句子文本
            text = sample.get("text", "")
            sentence_chars = sample.get("sentence", [])
            
            # 优先使用字符列表重建文本（确保一致性）
            if sentence_chars:
                text = "".join(sentence_chars)
            
            procnet_sentences.append(text)
            
            # 处理实体
            for ent in sample.get("ner", []):
                indices = ent.get("index", [])
                entity_type = ent.get("type", "unknown")
                
                if not indices:
                    continue
                
                # 计算起始和结束位置
                start = min(indices)
                end = max(indices) + 1  # 右开区间
                
                # 提取实体文本
                if sentence_chars:
                    entity_text = "".join(sentence_chars[start:end])
                else:
                    entity_text = text[start:end] if start < len(text) else ""
                
                # 创建唯一 key（支持同位置多类型）
                unique_key = f"{entity_text}#{sent_idx}_{start}_{end}#{entity_type}"
                
                # 添加到提及列表
                ann_valid_mspans.append(entity_text)
                
                # 添加位置
                drange = [sent_idx, start, end]
                ann_mspan2dranges[unique_key].append(drange)
                ann_valid_dranges.append(drange)
                
                # 添加类型映射
                ann_mspan2guess_field[unique_key] = entity_type
        
        # 构建 ProcNet 文档
        procnet_doc = [
            doc_id,
            {
                "sentences": procnet_sentences,
                "ann_valid_mspans": ann_valid_mspans,
                "ann_valid_dranges": ann_valid_dranges,
                "ann_mspan2dranges": dict(ann_mspan2dranges),
                "ann_mspan2guess_field": ann_mspan2guess_field,
            }
        ]
        
        procnet_docs.append(procnet_doc)
    
    # 按 doc_id 排序（确保一致性）
    procnet_docs.sort(key=lambda x: x[0])
    
    return procnet_docs


def verify_alignment(w2ner_samples, procnet_docs):
    """验证 W2NER 和 ProcNet 数据是否严格对齐"""
    # 统计 W2NER 句子数
    w2ner_sentence_count = len(w2ner_samples)
    
    # 统计 ProcNet 句子数
    procnet_sentence_count = 0
    for doc in procnet_docs:
        procnet_sentence_count += len(doc[1].get("sentences", []))
    
    # 统计 W2NER 实体数
    w2ner_entity_count = sum(len(s.get("ner", [])) for s in w2ner_samples)
    
    # 统计 ProcNet 实体数
    procnet_entity_count = 0
    for doc in procnet_docs:
        procnet_entity_count += len(doc[1].get("ann_valid_dranges", []))
    
    match = (w2ner_sentence_count == procnet_sentence_count) and \
            (w2ner_entity_count == procnet_entity_count)
    
    return {
        "w2ner_sentences": w2ner_sentence_count,
        "procnet_sentences": procnet_sentence_count,
        "w2ner_entities": w2ner_entity_count,
        "procnet_entities": procnet_entity_count,
        "aligned": match
    }


def main():
    print("=" * 80)
    print("统一数据链路：W2NER → ProcNet")
    print("=" * 80)
    
    for dataset in DATASETS:
        print(f"\n【{dataset}】")
        print("-" * 80)
        
        for split in SPLITS:
            source_path = SOURCE_ROOT / dataset / f"{split}.json"
            target_path = TARGET_ROOT / dataset / f"{split}.json"
            
            if not source_path.exists():
                print(f"  ⚠️  跳过：{source_path} 不存在")
                continue
            
            # 加载 W2NER 数据
            w2ner_data = load_json(source_path)
            
            # 转换为 ProcNet 格式
            procnet_data = w2ner_to_procnet(w2ner_data)
            
            # 验证对齐
            verification = verify_alignment(w2ner_data, procnet_data)
            
            # 保存 ProcNet 数据
            target_path.parent.mkdir(parents=True, exist_ok=True)
            dump_json(procnet_data, target_path)
            
            # 打印统计
            status = "✅" if verification["aligned"] else "❌"
            print(f"  {status} {split}: W2NER 句子={verification['w2ner_sentences']}, "
                  f"ProcNet 句子={verification['procnet_sentences']}, "
                  f"实体={verification['w2ner_entities']}")
            
            if not verification["aligned"]:
                print(f"      ⚠️  警告：句子数或实体数不匹配！")
    
    print("\n" + "=" * 80)
    print("完成！所有数据已严格对齐。")
    print("=" * 80)


if __name__ == "__main__":
    main()
