
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 DocEE_processor 初始化错误
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "procnet"))

from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_example.DocEEexample import DocEEEntity

def debug_parse_json_one():
    """调试 parse_json_one 方法"""
    print("=" * 70)
    print("调试 parse_json_one")
    print("=" * 70)
    
    dataset_dir = Path(__file__).parent / "procnet" / "procnet_format" / "mixed_data_with_queries"
    
    # 手动加载数据并检查
    for split in ["train", "dev", "test"]:
        split_path = dataset_dir / f"{split}.json"
        if not split_path.exists():
            continue
        
        print(f"\n处理 {split}...")
        
        with open(split_path, "r") as f:
            raw_data = json.load(f)
        
        for doc_idx, doc in enumerate(raw_data[:100]):  # 检查前 100 个文档
            doc_id = doc[0]
            doc_data = doc[1]
            sentences = doc_data.get("sentences", [])
            ann_mspan2dranges = doc_data.get("ann_mspan2dranges", {})
            ann_mspan2guess_field = doc_data.get("ann_mspan2guess_field", {})
            
            # 模拟创建 entities
            entities = []
            for raw_key, positions in ann_mspan2dranges.items():
                # 解析 mspan_key
                match = re.match(r"^(.*)#(\d+)_(\d+)_(\d+)#([^#]+)$", raw_key)
                if match:
                    span_text, sent_idx, b, e, field_from_key = match.groups()
                    sent_idx, b, e = int(sent_idx), int(b), int(e)
                else:
                    span_text = raw_key
                    sent_idx, b, e = positions[0] if positions else (0, 0, 0)
                
                field = ann_mspan2guess_field[raw_key]
                
                # 检查 span 是否匹配
                if sent_idx < len(sentences):
                    expected_span = sentences[sent_idx][b:e]
                    if span_text != expected_span:
                        print(f"\n❌ {doc_id}: span 不匹配")
                        print(f"   mspan_key: {raw_key}")
                        print(f"   span_text: '{span_text}'")
                        print(f"   position: sent={sent_idx}, [{b}:{e}]")
                        print(f"   expected:  '{expected_span}'")
                        print(f"   sentence:  '{sentences[sent_idx][:80]}...'")
                        return doc_id, raw_key, span_text, expected_span, sentences[sent_idx], sent_idx, b, e
            
            if doc_idx % 20 == 0:
                print(f"  已检查 {doc_idx}/{len(raw_data)} 文档")
        
        print(f"  ✅ {split} 集检查通过")
    
    return None


def main():
    result = debug_parse_json_one()
    
    if result:
        doc_id, raw_key, span_text, expected_span, sentence, sent_idx, b, e = result
        print("\n" + "=" * 70)
        print("详细分析")
        print("=" * 70)
        print(f"\n句子 ({len(sentence)} chars):")
        print(f"  {sentence}")
        print(f"\n位置 [{b}:{e}]:")
        print(f"  {' ' * b}{'^' * (e-b)}")
        print(f"\n提取的 span: '{expected_span}'")
        print(f"mspan_key 中的 span: '{span_text}'")
        
        # 检查是否有编码问题
        print(f"\n字符编码检查:")
        for i in range(max(0, b-2), min(len(sentence), e+2)):
            char = sentence[i]
            marker = " <--" if b <= i < e else ""
            print(f"  [{i}] '{char}' (U+{ord(char):04X}){marker}")


if __name__ == "__main__":
    main()
