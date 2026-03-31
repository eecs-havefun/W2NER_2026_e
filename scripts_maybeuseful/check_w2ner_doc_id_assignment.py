
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 W2NER 数据中 doc_id 的分配是否合理
"""

import json
from pathlib import Path

W2NER_PATH = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev" / "flight_orders_with_queries"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("检查 W2NER 数据中 doc_id 的分配\n")
    
    for split in ["train", "dev", "test"]:
        path = W2NER_PATH / f"{split}.json"
        if not path.exists():
            continue
        
        data = load_json(path)
        
        print(f"=== {split}.json ===")
        print(f"样本数：{len(data)}")
        
        # 检查 sample_id 的格式
        print(f"\nSample_id 格式检查 (前 5 个):")
        for i, sample in enumerate(data[:5]):
            print(f"  {i+1}. sample_id={sample.get('sample_id')}, "
                  f"doc_id={sample.get('doc_id')}, "
                  f"sent_id={sample.get('sent_id')}")
            print(f"      text: {sample.get('text', '')[:60]}...")
        
        # 检查是否有 doc_id 相同但内容完全不同的情况
        from collections import defaultdict
        doc_texts = defaultdict(list)
        for sample in data:
            doc_id = sample.get("doc_id")
            text = sample.get("text", "")
            doc_texts[doc_id].append(text)
        
        # 找出同一个 doc_id 下有多少不同的文本
        multi_text_docs = 0
        for doc_id, texts in doc_texts.items():
            unique_texts = set(texts)
            if len(unique_texts) > 1:
                multi_text_docs += 1
        
        print(f"\n同一个 doc_id 下有不同文本的文档数：{multi_text_docs}")
        
        # 显示一个具体例子
        print(f"\n具体例子 (doc_000001):")
        if "doc_000001" in doc_texts:
            for i, text in enumerate(doc_texts["doc_000001"][:5]):
                print(f"  [{i}] {text[:70]}...")
    
    print("\n\n结论:")
    print("如果同一个 doc_id 下有多个完全不同的文本，说明 doc_id 分配有问题。")
    print("这些'文档'实际上不是真正的文档，而是多个独立订单的混合。")


if __name__ == "__main__":
    main()
