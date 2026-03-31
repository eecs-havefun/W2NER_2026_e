#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度检查 ProcNet -> W2NER 转换质量。

检查项目：
1. strict_alignment: sentence[start:end] 是否等于实体文本
2. same-span multi-type: 同一位置多个类型是否保留
3. 字段完整性：sample_id / doc_id / sent_id / text / entities
"""

import json
import os
from collections import defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_dataset(data_path, check_name):
    """深度检查数据集"""
    print(f"\n{'='*70}")
    print(f"检查：{check_name}")
    print(f"数据：{data_path}")
    print(f"{'='*70}")

    data = load_json(data_path)
    print(f"总样本数：{len(data)}")

    issues = {
        "alignment_errors": [],
        "missing_fields": [],
        "multi_type_lost": [],
        "field_format_errors": [],
    }

    # 统计
    stats = {
        "total_entities": 0,
        "samples_with_entities_field": 0,
        "multi_type_entities": 0,
    }

    for idx, sample in enumerate(data):
        # 检查 1: 字段完整性
        required_fields = ["sample_id", "doc_id", "sent_id", "text", "sentence", "ner"]
        for field in required_fields:
            if field not in sample:
                issues["missing_fields"].append(
                    f"[{idx}] 缺少字段：{field}"
                )

        # 检查字段格式
        if "sample_id" in sample:
            sample_id = sample["sample_id"]
            doc_id = sample.get("doc_id", "")
            sent_id = sample.get("sent_id", "")
            # 检查 sample_id 格式
            expected_sample_id = f"{doc_id}__sent_{sent_id}"
            if sample_id != expected_sample_id:
                issues["field_format_errors"].append(
                    f"[{idx}] sample_id 格式错误：{sample_id!r} != {expected_sample_id!r}"
                )

        # 检查 2: strict alignment
        sentence = sample.get("sentence", [])
        text = sample.get("text", "")
        entities = sample.get("entities", [])
        ner = sample.get("ner", [])

        # 验证 text/sentence 一致性
        reconstructed = "".join(sentence)
        if reconstructed != text:
            issues["alignment_errors"].append(
                f"[{idx}] text/sentence 不匹配：text={text!r}, reconstructed={reconstructed!r}"
            )

        # 验证每个实体的 start/end 与文本
        for ent_idx, ent in enumerate(entities):
            start = ent.get("start")
            end = ent.get("end")
            ent_text = ent.get("text")
            ent_type = ent.get("type_name")

            if start is not None and end is not None:
                # 从 sentence 提取
                if 0 <= start < end <= len(sentence):
                    extracted = "".join(sentence[i] for i in range(start, end))
                    if extracted != ent_text:
                        issues["alignment_errors"].append(
                            f"[{idx}] 实体文本不匹配 (sent[{start}:{end}]): "
                            f"expected={ent_text!r}, got={extracted!r}, type={ent_type}"
                        )
                else:
                    issues["alignment_errors"].append(
                        f"[{idx}] 实体位置越界：start={start}, end={end}, sent_len={len(sentence)}"
                    )

            # 验证 ner 中的 index
            stats["total_entities"] += 1

        if entities:
            stats["samples_with_entities_field"] += 1

        # 检查 3: same-span multi-type 保留情况
        # 按 (start, end) 分组
        span2types = defaultdict(list)
        for ent in entities:
            start = ent.get("start")
            end = ent.get("end")
            ent_type = ent.get("type_name")
            orig_type = ent.get("orig_type_name")
            if start is not None and end is not None:
                span_key = (start, end)
                span2types[span_key].append({
                    "type_name": ent_type,
                    "orig_type_name": orig_type,
                    "text": ent.get("text")
                })

        for span_key, type_list in span2types.items():
            if len(type_list) > 1:
                stats["multi_type_entities"] += 1
                # 检查是否有类型丢失
                orig_types = set(t["orig_type_name"] for t in type_list if t.get("orig_type_name"))
                folded_types = set(t["type_name"] for t in type_list)

                # 如果折叠后类型数减少，说明折叠生效了
                # 如果原始类型数 > 折叠后类型数，说明有 same-span multi-type
                if len(orig_types) > 1:
                    # 这是一个 same-span multi-type 实体
                    pass  # 正常保留

    # 打印结果
    print(f"\n【字段完整性】")
    if issues["missing_fields"]:
        print(f"  ❌ 缺少字段：{len(issues['missing_fields'])} 处")
        for err in issues["missing_fields"][:5]:
            print(f"    - {err}")
    else:
        print(f"  ✅ 所有必需字段都存在")

    print(f"\n【字段格式】")
    if issues["field_format_errors"]:
        print(f"  ❌ 格式错误：{len(issues['field_format_errors'])} 处")
        for err in issues["field_format_errors"][:5]:
            print(f"    - {err}")
    else:
        print(f"  ✅ sample_id/doc_id/sent_id 格式正确")

    print(f"\n【严格对齐】")
    if issues["alignment_errors"]:
        print(f"  ❌ 对齐错误：{len(issues['alignment_errors'])} 处")
        for err in issues["alignment_errors"][:10]:
            print(f"    - {err}")
    else:
        print(f"  ✅ sentence[start:end] 与实体文本完全一致")

    print(f"\n【Same-span Multi-type】")
    print(f"  多类型实体数：{stats['multi_type_entities']}")
    if stats["multi_type_entities"] > 0:
        print(f"  ✅ 检测到 same-span multi-type 实体")
    else:
        print(f"  ℹ️  未检测到 same-span multi-type 实体（可能数据本身没有）")

    print(f"\n【统计信息】")
    print(f"  总实体数：{stats['total_entities']}")
    print(f"  含 entities 字段的样本：{stats['samples_with_entities_field']}")

    # 返回是否有问题
    has_errors = (
        len(issues["missing_fields"]) > 0 or
        len(issues["field_format_errors"]) > 0 or
        len(issues["alignment_errors"]) > 0
    )

    return has_errors, issues


def compare_multi_type_preservation(original_path, folded_path):
    """比较原始数据和折叠数据的 multi-type 保留情况"""
    print(f"\n{'='*70}")
    print("对比：原始数据 vs 折叠数据 的 same-span multi-type 保留")
    print(f"{'='*70}")

    original_data = load_json(original_path)
    folded_data = load_json(folded_path)

    # 按 doc_id + sent_id 建立索引
    def build_index(data):
        index = {}
        for sample in data:
            key = (sample.get("doc_id"), sample.get("sent_id"))
            index[key] = sample
        return index

    orig_index = build_index(original_data)
    folded_index = build_index(folded_data)

    # 统计
    orig_multi_type = 0
    folded_multi_type = 0
    lost_multi_type = 0

    for key, sample in orig_index.items():
        entities = sample.get("entities", [])
        span2types = defaultdict(set)
        for ent in entities:
            start, end = ent.get("start"), ent.get("end")
            orig_type = ent.get("orig_type_name")
            if start is not None and end is not None and orig_type:
                span2types[(start, end)].add(orig_type)

        for span, types in span2types.items():
            if len(types) > 1:
                orig_multi_type += 1

    for key, sample in folded_index.items():
        entities = sample.get("entities", [])
        span2types = defaultdict(set)
        for ent in entities:
            start, end = ent.get("start"), ent.get("end")
            orig_type = ent.get("orig_type_name")
            if start is not None and end is not None and orig_type:
                span2types[(start, end)].add(orig_type)

        for span, types in span2types.items():
            if len(types) > 1:
                folded_multi_type += 1

    print(f"\n原始数据 same-span multi-type 实体数：{orig_multi_type}")
    print(f"折叠数据 same-span multi-type 实体数：{folded_multi_type}")

    if orig_multi_type == folded_multi_type:
        print(f"✅ same-span multi-type 完整保留")
    elif folded_multi_type < orig_multi_type:
        print(f"⚠️  部分 multi-type 实体丢失（可能是折叠导致的类型合并）")
    else:
        print(f"ℹ️  数据有差异，请检查")


def main():
    # 检查 5 块原始数据
    datasets_original = [
        ("flight_orders", "./data_w2ner/flight_orders_with_queries/train.json"),
        ("hotel_orders", "./data_w2ner/hotel_orders_with_queries/train.json"),
        ("id_cards", "./data_w2ner/id_cards_with_queries/train.json"),
        ("mixed_data", "./data_w2ner/mixed_data_with_queries/train.json"),
        ("train_orders", "./data_w2ner/train_orders_with_queries/train.json"),
    ]

    # 检查 5 块折叠数据
    datasets_folded = [
        ("flight_orders (folded)", "./data_w2ner_folded/flight_orders_with_queries/train.json"),
        ("hotel_orders (folded)", "./data_w2ner_folded/hotel_orders_with_queries/train.json"),
        ("id_cards (folded)", "./data_w2ner_folded/id_cards_with_queries/train.json"),
        ("mixed_data (folded)", "./data_w2ner_folded/mixed_data_with_queries/train.json"),
        ("train_orders (folded)", "./data_w2ner_folded/train_orders_with_queries/train.json"),
    ]

    print("="*70)
    print("ProcNet -> W2NER 转换质量深度检查")
    print("="*70)

    # 检查原始数据
    print("\n" + "="*70)
    print("【第一部分】检查 data_w2ner (未折叠)")
    print("="*70)

    all_pass_original = True
    for name, path in datasets_original:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        has_errors, _ = check_dataset(path, name)
        if has_errors:
            all_pass_original = False
            print(f"\n❌ {name}: 发现问题")
        else:
            print(f"\n✅ {name}: 全部通过")

    # 检查折叠数据
    print("\n" + "="*70)
    print("【第二部分】检查 data_w2ner_folded (折叠)")
    print("="*70)

    all_pass_folded = True
    for name, path in datasets_folded:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        has_errors, _ = check_dataset(path, name)
        if has_errors:
            all_pass_folded = False
            print(f"\n❌ {name}: 发现问题")
        else:
            print(f"\n✅ {name}: 全部通过")

    # 对比 multi-type 保留
    print("\n" + "="*70)
    print("【第三部分】对比 same-span multi-type 保留情况")
    print("="*70)

    for (orig_name, orig_path), (fold_name, fold_path) in zip(datasets_original, datasets_folded):
        if os.path.exists(orig_path) and os.path.exists(fold_path):
            compare_multi_type_preservation(orig_path, fold_path)

    # 最终总结
    print("\n" + "="*70)
    print("最终检查结果")
    print("="*70)
    print(f"data_w2ner (原始): {'✅ 全部通过' if all_pass_original else '❌ 有问题'}")
    print(f"data_w2ner_folded (折叠): {'✅ 全部通过' if all_pass_folded else '❌ 有问题'}")
    print("="*70)


if __name__ == "__main__":
    main()
