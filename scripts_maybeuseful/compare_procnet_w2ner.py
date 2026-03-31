
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 ProcNet 格式和 W2NER 格式之间的句子差异
"""

import json
from pathlib import Path

PROCNET_PATH = project_root / "procnet" / "procnet_format" / "flight_orders_with_queries"
W2NER_PATH = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev" / "flight_orders_with_queries"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_procnet_sentences(data):
    """从 ProcNet 数据提取所有句子"""
    sentences = []
    for doc in data:
        doc_id, doc_data = doc[0], doc[1]
        doc_sentences = doc_data.get("sentences", [])
        for sent in doc_sentences:
            if isinstance(sent, list):
                sent = "".join(sent)
            sentences.append((doc_id, sent))
    return sentences

def get_w2ner_sentences(data):
    """从 W2NER 数据提取所有句子"""
    sentences = []
    for sample in data:
        doc_id = sample.get("doc_id", "unknown")
        text = sample.get("text", "")
        sentences.append((doc_id, text))
    return sentences

def main():
    print("比较 ProcNet 和 W2NER 的句子内容\n")
    
    # 加载数据
    print("加载 ProcNet train.json...")
    procnet_train = load_json(PROCNET_PATH / "train.json")
    procnet_sentences = get_procnet_sentences(procnet_train)
    print(f"ProcNet train: {len(procnet_sentences)} 句子")
    
    print("加载 W2NER train.json...")
    w2ner_train = load_json(W2NER_PATH / "train.json")
    w2ner_sentences = get_w2ner_sentences(w2ner_train)
    print(f"W2NER train: {len(w2ner_sentences)} 句子\n")
    
    # 比较句子文本
    procnet_texts = set(sent[1] for sent in procnet_sentences)
    w2ner_texts = set(sent[1] for sent in w2ner_sentences)
    
    only_in_procnet = procnet_texts - w2ner_texts
    only_in_w2ner = w2ner_texts - procnet_texts
    in_both = procnet_texts & w2ner_texts
    
    print(f"只在 ProcNet 的句子：{len(only_in_procnet)}")
    print(f"只在 W2NER 的句子：{len(only_in_w2ner)}")
    print(f"共同的句子：{len(in_both)}\n")
    
    # 显示一些差异
    if only_in_w2ner:
        print("只在 W2NER 的样本句子 (前 10 个):")
        for text in list(only_in_w2ner)[:10]:
            print(f"  - {text[:60]}...")
    
    if only_in_procnet:
        print("\n只在 ProcNet 的样本句子 (前 10 个):")
        for text in list(only_in_procnet)[:10]:
            print(f"  - {text[:60]}...")
    
    # 检查文档 ID
    procnet_doc_ids = set(sent[0] for sent in procnet_sentences)
    w2ner_doc_ids = set(sent[0] for sent in w2ner_sentences)
    
    print(f"\nProcNet 文档数：{len(procnet_doc_ids)}")
    print(f"W2NER 文档数：{len(w2ner_doc_ids)}")
    
    common_docs = procnet_doc_ids & w2ner_doc_ids
    print(f"共同文档数：{len(common_docs)}")
    
    # 检查共同文档中的句子
    if common_docs:
        sample_doc = list(common_docs)[0]
        procnet_doc_sents = [s[1] for s in procnet_sentences if s[0] == sample_doc]
        w2ner_doc_sents = [s[1] for s in w2ner_sentences if s[0] == sample_doc]
        
        print(f"\n示例文档 {sample_doc}:")
        print(f"  ProcNet 句子数：{len(procnet_doc_sents)}")
        print(f"  W2NER 句子数：{len(w2ner_doc_sents)}")

if __name__ == "__main__":
    main()
