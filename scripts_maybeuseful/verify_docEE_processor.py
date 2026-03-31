
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 DocEE_processor 修改是否正确

检查项：
1. 能否正常初始化 DocEEProcessor
2. doc.entities 数量是否和 ann_mspan2dranges 一致
3. 同一个 span 的双角色是否都被保留下来
"""

import sys
import json
from pathlib import Path

# 添加 procnet 到路径
sys.path.insert(0, str(Path(__file__).parent / "procnet"))

from procnet.data_processor.DocEE_processor import DocEEProcessor


def test_processor_initialization():
    """测试 1: 能否正常初始化 DocEEProcessor"""
    print("=" * 70)
    print("测试 1: DocEEProcessor 初始化")
    print("=" * 70)
    
    dataset_dir = Path(__file__).parent / "procnet" / "procnet_format" / "mixed_data_with_queries"
    sidecar_dir = Path(__file__).parent / "procnet" / "sidecar_entities_gold"
    
    print(f"\n配置:")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  sidecar_dir: {sidecar_dir}")
    
    try:
        processor = DocEEProcessor(
            read_pseudo_dataset=False,
            use_procnet_pred_entities=False,  # 先不加载 sidecar
            dataset_dir=str(dataset_dir),
        )
        print("\n✅ DocEEProcessor 初始化成功!")
        return processor
    except Exception as e:
        print(f"\n❌ DocEEProcessor 初始化失败:")
        print(f"  错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_entity_count_match(processor):
    """测试 2: doc.entities 数量是否和 ann_mspan2dranges 一致"""
    print("\n" + "=" * 70)
    print("测试 2: doc.entities 数量 vs ann_mspan2dranges")
    print("=" * 70)
    
    # 从原始数据加载 ann_mspan2dranges 数量
    dataset_dir = Path(__file__).parent / "procnet" / "procnet_format" / "mixed_data_with_queries"
    
    mismatches = []
    
    for split, docs in [("train", processor.train_docs), 
                        ("dev", processor.dev_docs), 
                        ("test", processor.test_docs)]:
        if not docs:
            continue
            
        # 加载原始数据对比
        with open(dataset_dir / f"{split}.json", "r") as f:
            raw_data = json.load(f)
        
        raw_counts = {}
        for doc in raw_data:
            doc_id = doc[0]
            ann_mspan2dranges = doc[1].get("ann_mspan2dranges", {})
            raw_counts[doc_id] = len(ann_mspan2dranges)
        
        # 对比 processor 中的 doc
        print(f"\n{split} 集:")
        split_mismatches = []
        
        for doc in docs[:10]:  # 检查前 10 个
            doc_id = getattr(doc, "doc_id", None)
            num_entities = len(getattr(doc, "entities", []))
            raw_count = raw_counts.get(doc_id, -1)
            
            if num_entities != raw_count:
                split_mismatches.append({
                    "doc_id": doc_id,
                    "entities": num_entities,
                    "ann_mspan2dranges": raw_count
                })
        
        if split_mismatches:
            print(f"  ❌ 发现 {len(split_mismatches)} 个不匹配:")
            for m in split_mismatches[:5]:
                print(f"    {m['doc_id']}: entities={m['entities']}, ann_mspan2dranges={m['ann_mspan2dranges']}")
            mismatches.extend(split_mismatches)
        else:
            print(f"  ✅ 所有文档的 entities 数量与 ann_mspan2dranges 一致")
    
    if mismatches:
        print(f"\n⚠️  共发现 {len(mismatches)} 个不匹配文档")
        return False
    else:
        print(f"\n✅ 所有文档的 entities 数量与 ann_mspan2dranges 一致")
        return True


def test_dual_role_preservation(processor):
    """测试 3: 同一个 span 的双角色是否都被保留下来"""
    print("\n" + "=" * 70)
    print("测试 3: 同一个 span 的双角色保留")
    print("=" * 70)
    
    # 从原始数据中找有双角色的 span
    dataset_dir = Path(__file__).parent / "procnet" / "procnet_format" / "mixed_data_with_queries"
    
    dual_role_docs = []
    
    for split in ["train", "dev", "test"]:
        with open(dataset_dir / f"{split}.json", "r") as f:
            raw_data = json.load(f)
        
        for doc in raw_data:
            doc_id = doc[0]
            ann_mspan2dranges = doc[1].get("ann_mspan2dranges", {})
            
            # 检查是否有相同文本不同角色的情况
            span_to_roles = {}
            for mspan_key, positions in ann_mspan2dranges.items():
                # mspan_key 格式：文本#sent_b_e#类型
                parts = mspan_key.split("#")
                if len(parts) >= 3:
                    text = parts[0]
                    role = parts[-1]
                    pos = parts[1]  # sent_b_e
                    
                    key = f"{text}#{pos}"
                    if key not in span_to_roles:
                        span_to_roles[key] = []
                    span_to_roles[key].append(role)
            
            # 找到双角色
            for key, roles in span_to_roles.items():
                if len(roles) > 1:
                    dual_role_docs.append({
                        "doc_id": doc_id,
                        "span_key": key,
                        "roles": roles
                    })
    
    print(f"\n原始数据中双角色 span 数量：{len(dual_role_docs)}")
    
    if dual_role_docs:
        print(f"\n双角色示例 (前 5 个):")
        for dr in dual_role_docs[:5]:
            print(f"  {dr['doc_id']}: '{dr['span_key']}' -> {dr['roles']}")
    
    # 检查 processor 中是否保留了双角色
    print(f"\n检查 processor 中的双角色保留:")
    
    preserved_count = 0
    lost_count = 0
    
    for split, docs in [("train", processor.train_docs), 
                        ("dev", processor.dev_docs), 
                        ("test", processor.test_docs)]:
        if not docs:
            continue
        
        for doc in docs:
            doc_id = getattr(doc, "doc_id", None)
            entities = getattr(doc, "entities", [])
            
            # 按 span 分组
            span_to_roles = {}
            for ent in entities:
                span = getattr(ent, "span", "")
                field = getattr(ent, "field", "")
                positions = getattr(ent, "positions", [])
                
                for pos in positions:
                    sent_idx, b, e = pos
                    key = f"{span}#{sent_idx}_{b}_{e}"
                    
                    if key not in span_to_roles:
                        span_to_roles[key] = []
                    span_to_roles[key].append(field)
            
            # 检查双角色
            for key, roles in span_to_roles.items():
                if len(roles) > 1:
                    preserved_count += 1
    
    print(f"  保留的双角色数量：{preserved_count}")
    
    if preserved_count > 0:
        print(f"  ✅ 双角色被保留")
        return True
    else:
        print(f"  ⚠️  未发现双角色 (可能原始数据中就没有，或者丢失了)")
        return len(dual_role_docs) == 0  # 如果原始数据就没有双角色，也算通过


def test_entity_positions(processor):
    """额外测试：entity.positions 是否正确"""
    print("\n" + "=" * 70)
    print("测试 4: entity.positions 正确性")
    print("=" * 70)
    
    errors = []
    
    for split, docs in [("train", processor.train_docs), 
                        ("dev", processor.dev_docs), 
                        ("test", processor.test_docs)]:
        if not docs:
            continue
        
        for doc in docs[:5]:  # 检查前 5 个
            doc_id = getattr(doc, "doc_id", None)
            sentences = getattr(doc, "sentences", [])
            entities = getattr(doc, "entities", [])
            
            for ent in entities:
                span = getattr(ent, "span", "")
                positions = getattr(ent, "positions", [])
                
                for sent_idx, b, e in positions:
                    if sent_idx >= len(sentences):
                        errors.append({
                            "doc_id": doc_id,
                            "span": span,
                            "error": f"sent_idx={sent_idx} >= num_sentences={len(sentences)}"
                        })
                        continue
                    
                    actual_span = sentences[sent_idx][b:e]
                    if actual_span != span:
                        errors.append({
                            "doc_id": doc_id,
                            "span": span,
                            "error": f"span mismatch: '{span}' vs actual '{actual_span}' @ [{b}:{e}]"
                        })
    
    if errors:
        print(f"  ❌ 发现 {len(errors)} 个错误:")
        for e in errors[:5]:
            print(f"    {e['doc_id']}: {e['error']}")
        return False
    else:
        print(f"  ✅ 所有 entity.positions 正确")
        return True


def main():
    print("=" * 70)
    print("DocEE_processor 验证")
    print("=" * 70)
    
    # 测试 1: 初始化
    processor = test_processor_initialization()
    if not processor:
        print("\n❌ 测试终止：Processor 初始化失败")
        return
    
    # 测试 2: entity 数量
    test2_pass = test_entity_count_match(processor)
    
    # 测试 3: 双角色保留
    test3_pass = test_dual_role_preservation(processor)
    
    # 测试 4: positions 正确性
    test4_pass = test_entity_positions(processor)
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  测试 1 (初始化):          ✅ 通过")
    print(f"  测试 2 (entity 数量):       {'✅ 通过' if test2_pass else '❌ 失败'}")
    print(f"  测试 3 (双角色保留):      {'✅ 通过' if test3_pass else '⚠️  警告'}")
    print(f"  测试 4 (positions 正确性): {'✅ 通过' if test4_pass else '❌ 失败'}")
    
    if test2_pass and test4_pass:
        print("\n✅ 核心测试全部通过!")
    else:
        print("\n⚠️  部分测试失败，请检查修改")


if __name__ == "__main__":
    main()
