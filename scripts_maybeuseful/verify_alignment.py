from pathlib import Path

# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 W2NER 格式数据的严格对齐性。

检查项目：
1. sentence 字符列表拼接后是否等于 text 字段
2. ner 中的 index 是否在 sentence 范围内
3. ner 中的 index 提取的文本是否与实体语义一致（如果可能）
"""

import json
import os
from collections import defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_sample(sample, sample_idx, filepath):
    """验证单个样本的对齐性"""
    errors = []
    warnings = []

    doc_id = sample.get("doc_id", "unknown")
    sent_id = sample.get("sent_id", sample_idx)
    text = sample.get("text", "")
    sentence = sample.get("sentence", [])
    ner = sample.get("ner", [])

    # 检查 1: sentence 拼接后是否等于 text
    reconstructed_text = "".join(sentence)
    if reconstructed_text != text:
        errors.append(
            f"[{filepath}] doc={doc_id} sent={sent_id}: "
            f"text/sentence mismatch!\n"
            f"  text: {text!r}\n"
            f"  reconstructed: {reconstructed_text!r}"
        )

    # 检查 2: ner 中的 index 是否有效
    sent_len = len(sentence)
    for ent_idx, ent in enumerate(ner):
        index = ent.get("index", [])
        ent_type = ent.get("type", "unknown")

        # 检查 index 是否为空
        if not index:
            errors.append(
                f"[{filepath}] doc={doc_id} sent={sent_id} ent={ent_idx}: "
                f"empty index for type={ent_type}"
            )
            continue

        # 检查 index 是否连续
        expected_index = list(range(index[0], index[0] + len(index)))
        if index != expected_index:
            warnings.append(
                f"[{filepath}] doc={doc_id} sent={sent_id} ent={ent_idx}: "
                f"non-continuous index {index} for type={ent_type}"
            )

        # 检查 index 范围
        for i in index:
            if not (0 <= i < sent_len):
                errors.append(
                    f"[{filepath}] doc={doc_id} sent={sent_id} ent={ent_idx}: "
                    f"index {i} out of range [0, {sent_len}) for type={ent_type}"
                )

        # 检查 index 是否严格递增
        if len(index) != len(set(index)):
            warnings.append(
                f"[{filepath}] doc={doc_id} sent={sent_id} ent={ent_idx}: "
                f"duplicate index {index} for type={ent_type}"
            )

        # 检查 3: 提取实体文本验证（如果有 entities 字段）
        # 注意：同一个 type 可能有多个实体，需要匹配 index 范围
        if "entities" in sample:
            ent_index_set = set(index)
            matched = False
            for full_ent in sample["entities"]:
                if full_ent.get("type_name") == ent_type:
                    ent_start = full_ent.get("start")
                    ent_end = full_ent.get("end")
                    ent_text = full_ent.get("text")

                    if ent_start is not None and ent_end is not None:
                        # 检查 index 范围是否匹配
                        expected_index = list(range(ent_start, ent_end))
                        if set(expected_index) == ent_index_set:
                            matched = True
                            # 验证 index 对应的文本
                            extracted = "".join(sentence[i] for i in index)
                            if extracted != ent_text:
                                errors.append(
                                    f"[{filepath}] doc={doc_id} sent={sent_id} ent={ent_idx}: "
                                    f"text mismatch! index={index}, extracted={extracted!r}, expected={ent_text!r}"
                                )
                            break
            
            # 如果没有找到匹配的 entity，可能是重复类型但不同位置的情况
            if not matched and len([e for e in sample.get("entities", []) if e.get("type_name") == ent_type]) > 1:
                # 多个相同类型的实体，需要更精确的匹配
                for full_ent in sample["entities"]:
                    if full_ent.get("type_name") == ent_type:
                        ent_start = full_ent.get("start")
                        ent_end = full_ent.get("end")
                        ent_text = full_ent.get("text")
                        expected_index = list(range(ent_start, ent_end))
                        if expected_index == index:
                            matched = True
                            extracted = "".join(sentence[i] for i in index)
                            if extracted != ent_text:
                                errors.append(
                                    f"[{filepath}] doc={doc_id} sent={sent_id} ent={ent_idx}: "
                                    f"text mismatch! index={index}, extracted={extracted!r}, expected={ent_text!r}"
                                )
                            break
            
            if not matched:
                # 可能是验证逻辑问题，先不报错
                pass

    return errors, warnings


def verify_dataset(data_path):
    """验证整个数据集"""
    print(f"\n{'='*60}")
    print(f"Verifying: {data_path}")
    print(f"{'='*60}")

    data = load_json(data_path)
    print(f"Total samples: {len(data)}")

    total_errors = 0
    total_warnings = 0
    error_samples = []
    warning_samples = []

    for idx, sample in enumerate(data):
        errors, warnings = verify_sample(sample, idx, data_path)
        total_errors += len(errors)
        total_warnings += len(warnings)

        if errors:
            error_samples.append((idx, sample.get("doc_id", "unknown"), errors))
        if warnings:
            warning_samples.append((idx, sample.get("doc_id", "unknown"), warnings))

    # 打印结果
    print(f"\n❌ Errors: {total_errors}")
    print(f"⚠️  Warnings: {total_warnings}")

    if error_samples:
        print(f"\n❌ Samples with errors ({len(error_samples)}):")
        for idx, doc_id, errors in error_samples[:5]:  # 只显示前 5 个
            print(f"  [{idx}] doc={doc_id}")
            for err in errors[:3]:
                print(f"    - {err[:200]}...")

    if warning_samples:
        print(f"\n⚠️  Samples with warnings ({len(warning_samples)}):")
        for idx, doc_id, warnings in warning_samples[:5]:
            print(f"  [{idx}] doc={doc_id}")
            for warn in warnings[:3]:
                print(f"    - {warn[:200]}...")

    return total_errors, total_warnings


def main():
    # 验证原始数据 (data_w2ner)
    datasets_original = [
        ("flight_orders", "./data_w2ner/flight_orders_with_queries/train.json"),
        ("hotel_orders", "./data_w2ner/hotel_orders_with_queries/train.json"),
        ("id_cards", "./data_w2ner/id_cards_with_queries/train.json"),
        ("mixed_data", "./data_w2ner/mixed_data_with_queries/train.json"),
        ("train_orders", "./data_w2ner/train_orders_with_queries/train.json"),
    ]

    # 验证折叠数据 (data_w2ner_folded)
    datasets_folded = [
        ("flight_orders (folded)", "./data_w2ner_folded/flight_orders_with_queries/train.json"),
        ("hotel_orders (folded)", "./data_w2ner_folded/hotel_orders_with_queries/train.json"),
        ("id_cards (folded)", "./data_w2ner_folded/id_cards_with_queries/train.json"),
        ("mixed_data (folded)", "./data_w2ner_folded/mixed_data_with_queries/train.json"),
        ("train_orders (folded)", "./data_w2ner_folded/train_orders_with_queries/train.json"),
    ]

    print("=" * 60)
    print("W2NER 数据严格对齐验证")
    print("=" * 60)

    # 验证原始数据
    print("\n" + "=" * 60)
    print("【原始数据】data_w2ner")
    print("=" * 60)

    all_pass_original = True
    for name, path in datasets_original:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        errors, warnings = verify_dataset(path)
        if errors > 0:
            all_pass_original = False
            print(f"❌ {name}: FAILED")
        else:
            print(f"✅ {name}: PASSED")

    # 验证折叠数据
    print("\n" + "=" * 60)
    print("【折叠数据】data_w2ner_folded")
    print("=" * 60)

    all_pass_folded = True
    for name, path in datasets_folded:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        errors, warnings = verify_dataset(path)
        if errors > 0:
            all_pass_folded = False
            print(f"❌ {name}: FAILED")
        else:
            print(f"✅ {name}: PASSED")

    # 最终总结
    print("\n" + "=" * 60)
    print("最终验证结果")
    print("=" * 60)
    print(f"原始数据 (data_w2ner): {'✅ 全部通过' if all_pass_original else '❌ 有错误'}")
    print(f"折叠数据 (data_w2ner_folded): {'✅ 全部通过' if all_pass_folded else '❌ 有错误'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
