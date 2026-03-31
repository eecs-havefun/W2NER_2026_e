#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 ProcNet Sidecar 格式的正确性

检查项：
1. 必填字段是否存在
2. doc_id 格式是否正确
3. sent_id 是否合理
4. b/e 位置是否合法
5. 是否有重复 entity（按 key）
6. 与 ProcNet raw 数据的对齐
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple

SIDECAR_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/sidecar_entities/test_typed_entities.jsonl")
PROCNET_RAW_PATH = Path("/home/mengfanrong/finaldesign/W2NERproject/procnet/procnet_format/mixed_data_with_queries/test.json")

# 必填字段
REQUIRED_FIELDS = ["doc_id", "sent_id", "b", "e", "text", "type_name"]

# 推荐字段
RECOMMENDED_FIELDS = ["token_indices", "head", "key", "cluster_key", "source"]


def load_sidecar(path: Path) -> List[Dict]:
    """加载 JSONL sidecar 文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_procnet_raw(path: Path) -> List:
    """加载 ProcNet raw 数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_fields(records: List[Dict]) -> Tuple[int, List[str]]:
    """检查必填字段"""
    errors = []
    for i, record in enumerate(records):
        for field in REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"记录 {i+1}: 缺少必填字段 '{field}'")
    return len(records) - len(set(errors)), errors


def check_doc_id_format(records: List[Dict]) -> Tuple[int, List[str]]:
    """检查 doc_id 格式"""
    errors = []
    for i, record in enumerate(records):
        doc_id = record.get("doc_id", "")
        if not doc_id.startswith("doc_"):
            errors.append(f"记录 {i+1}: doc_id 格式错误 '{doc_id}'")
    return len(records) - len(set(errors)), errors


def check_span_positions(records: List[Dict]) -> Tuple[int, List[str]]:
    """检查 b/e 位置合法性"""
    errors = []
    for i, record in enumerate(records):
        b = record.get("b", -1)
        e = record.get("e", -1)
        text = record.get("text", "")
        
        if b < 0:
            errors.append(f"记录 {i+1}: b 位置为负数 {b}")
        if e < 0:
            errors.append(f"记录 {i+1}: e 位置为负数 {e}")
        if e <= b:
            errors.append(f"记录 {i+1}: e <= b ({e} <= {b})")
        if e - b != len(text):
            errors.append(f"记录 {i+1}: e-b ({e-b}) != text 长度 ({len(text)})")
    
    return len(records) - len(set(errors)), errors


def check_duplicate_entities(records: List[Dict]) -> Tuple[int, List[str]]:
    """检查重复 entity（按 key）"""
    errors = []
    seen_keys = set()
    duplicates = []
    
    for i, record in enumerate(records):
        key = record.get("key", "")
        if not key:
            # 如果没有 key，用 doc_id+sent_id+b+e+type 作为替代
            key = f"{record.get('doc_id')}#{record.get('sent_id')}_{record.get('b')}_{record.get('e')}#{record.get('type_name')}"
        
        if key in seen_keys:
            # 只有当 doc_id+sent_id+b+e+type 完全一样才算重复
            minimal_key = f"{record.get('doc_id')}#{record.get('sent_id')}_{record.get('b')}_{record.get('e')}#{record.get('type_name')}"
            if minimal_key in seen_keys:
                duplicates.append(f"记录 {i+1}: 重复 key '{minimal_key}'")
        else:
            minimal_key = f"{record.get('doc_id')}#{record.get('sent_id')}_{record.get('b')}_{record.get('e')}#{record.get('type_name')}"
            seen_keys.add(minimal_key)
    
    return len(duplicates), duplicates


def check_sent_id_range(records: List[Dict]) -> Tuple[int, List[str]]:
    """检查 sent_id 范围"""
    errors = []
    doc_sent_max = defaultdict(int)
    
    # 先统计每个 doc 的最大 sent_id
    for record in records:
        doc_id = record.get("doc_id", "")
        sent_id = record.get("sent_id", -1)
        if sent_id >= 0:
            doc_sent_max[doc_id] = max(doc_sent_max[doc_id], sent_id)
    
    # 检查是否有 sent_id 跳跃
    doc_sents = defaultdict(set)
    for record in records:
        doc_id = record.get("doc_id", "")
        sent_id = record.get("sent_id", -1)
        doc_sents[doc_id].add(sent_id)
    
    for doc_id, sents in doc_sents.items():
        max_sent = max(sents)
        expected = set(range(max_sent + 1))
        missing = expected - sents
        if missing:
            errors.append(f"文档 {doc_id}: 缺少 sent_id {sorted(missing)}")
    
    return len(errors), errors


def compare_with_procnet_raw(
    sidecar_records: List[Dict],
    procnet_docs: List
) -> List[str]:
    """与 ProcNet raw 数据对比"""
    errors = []
    
    # 构建 ProcNet raw 的 entity 索引
    procnet_entities = defaultdict(list)
    for doc in procnet_docs:
        if not (isinstance(doc, list) and len(doc) >= 2):
            continue
        doc_id = doc[0]
        doc_data = doc[1]
        
        sentences = doc_data.get("sentences", [])
        ann_mspan2dranges = doc_data.get("ann_mspan2dranges", {})
        ann_mspan2guess_field = doc_data.get("ann_mspan2guess_field", {})
        
        for mspan_key, dranges in ann_mspan2dranges.items():
            entity_type = ann_mspan2guess_field.get(mspan_key, "unknown")
            entity_text = mspan_key.split("#", 1)[0]
            
            for dr in dranges:
                if not (isinstance(dr, list) and len(dr) == 3):
                    continue
                sent_idx, start, end = dr
                procnet_entities[doc_id].append({
                    "sent_id": sent_idx,
                    "b": start,
                    "e": end,
                    "text": entity_text,
                    "type": entity_type,
                })
    
    # 对比 sidecar 和 ProcNet
    sidecar_by_doc = defaultdict(list)
    for record in sidecar_records:
        doc_id = record.get("doc_id", "")
        sidecar_by_doc[doc_id].append(record)
    
    # 检查每个 doc
    all_docs = set(procnet_entities.keys()) | set(sidecar_by_doc.keys())
    
    for doc_id in all_docs:
        procnet_ents = procnet_entities.get(doc_id, [])
        sidecar_ents = sidecar_by_doc.get(doc_id, [])
        
        # 简化对比：只检查数量差异
        if len(procnet_ents) != len(sidecar_ents):
            errors.append(
                f"文档 {doc_id}: ProcNet 有{len(procnet_ents)}个 entity, "
                f"Sidecar 有{len(sidecar_ents)}个 entity"
            )
    
    return errors[:20]  # 只返回前 20 个错误


def main():
    print("=" * 70)
    print("ProcNet Sidecar 格式验证")
    print("=" * 70)
    
    # 检查文件是否存在
    if not SIDECAR_PATH.exists():
        print(f"\n❌ Sidecar 文件不存在：{SIDECAR_PATH}")
        return
    
    # 加载数据
    print(f"\n加载 Sidecar 文件：{SIDECAR_PATH}")
    sidecar_records = load_sidecar(SIDECAR_PATH)
    print(f"  记录数：{len(sidecar_records)}")
    
    all_errors = []
    
    # 1. 检查必填字段
    print("\n1. 检查必填字段...")
    count, errors = check_required_fields(sidecar_records)
    if errors:
        print(f"  ❌ {len(errors)} 个错误")
        all_errors.extend(errors)
    else:
        print(f"  ✅ 所有记录都有必填字段")
    
    # 2. 检查 doc_id 格式
    print("\n2. 检查 doc_id 格式...")
    count, errors = check_doc_id_format(sidecar_records)
    if errors:
        print(f"  ❌ {len(errors)} 个错误")
        all_errors.extend(errors)
    else:
        print(f"  ✅ 所有 doc_id 格式正确")
    
    # 3. 检查 b/e 位置
    print("\n3. 检查 span 位置 (b/e)...")
    count, errors = check_span_positions(sidecar_records)
    if errors:
        print(f"  ❌ {len(errors)} 个错误")
        for e in errors[:5]:
            print(f"      {e}")
        all_errors.extend(errors)
    else:
        print(f"  ✅ 所有 span 位置合法")
    
    # 4. 检查重复 entity
    print("\n4. 检查重复 entity...")
    count, errors = check_duplicate_entities(sidecar_records)
    if errors:
        print(f"  ❌ {len(errors)} 个重复")
        for e in errors[:5]:
            print(f"      {e}")
        all_errors.extend(errors)
    else:
        print(f"  ✅ 没有重复 entity")
    
    # 5. 检查 sent_id 范围（只警告，不是错误，因为 W2NER 只输出有 entity 的句子）
    print("\n5. 检查 sent_id 范围...")
    count, errors = check_sent_id_range(sidecar_records)
    if errors:
        print(f"  ⚠️ {len(errors)} 个文档有没有 entity 的句子（预期内）")
        # 这不是错误，只是信息
    else:
        print(f"  ✅ 所有句子都有 entity")
    
    # 6. 与 ProcNet raw 对比（如果有 raw 文件）
    if PROCNET_RAW_PATH.exists():
        print("\n6. 与 ProcNet raw 数据对比...")
        procnet_docs = load_procnet_raw(PROCNET_RAW_PATH)
        errors = compare_with_procnet_raw(sidecar_records, procnet_docs)
        if errors:
            print(f"  ⚠️ 发现差异（预期内，W2NER 是预测结果）")
            for e in errors[:5]:
                print(f"      {e}")
        else:
            print(f"  ✅ Entity 数量一致")
    
    # 总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    
    # 只报告真正的错误（必填字段、doc_id 格式、span 位置、重复）
    real_errors = [e for e in all_errors if not e.startswith("文档 ") and "缺少 sent_id" not in e]
    
    if real_errors:
        print(f"\n❌ 共发现 {len(real_errors)} 个错误")
        print("\n前 10 个错误:")
        for e in real_errors[:10]:
            print(f"  - {e}")
    else:
        print("\n✅ 所有检查通过!")
    
    # 显示 sidecar 统计
    print("\nSidecar 统计:")
    docs = set(r.get("doc_id") for r in sidecar_records)
    sents = set((r.get("doc_id"), r.get("sent_id")) for r in sidecar_records)
    types = set(r.get("type_name") for r in sidecar_records)
    
    print(f"  文档数：{len(docs)}")
    print(f"  句子数：{len(sents)}")
    print(f"  Entity 类型数：{len(types)}")
    print(f"  Entity 总数：{len(sidecar_records)}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
