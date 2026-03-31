from pathlib import Path

# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 data_w2ner_folded 复制数据到新目录，并重新按 doc_id 划分 train/dev/test。

关键约束：
- 同一 doc_id 的样本必须在同一个集合中（train/dev/test）
- 先合并原始 train+test，然后按 doc_id 重新划分
- 划分比例：train 60%, dev 20%, test 20%（近似）

问题分析：
原始数据的 train/test 划分不是按 doc_id 划分的，导致同一篇文档的句子
既在训练集也在测试集，这会造成数据泄露。本脚本会修复这个问题。
"""

import json
import os
import shutil
from collections import defaultdict
import random

# 配置
SOURCE_ROOT = "./data_w2ner_folded"
TARGET_ROOT = "./W2NER/data/data_w2ner_folded_with_dev"

RANDOM_SEED = 42

# 划分比例（按 doc 数量）
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2
TEST_RATIO = 0.2


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def group_by_doc_id(samples):
    """按 doc_id 分组样本"""
    doc_groups = defaultdict(list)
    for sample in samples:
        doc_id = sample.get("doc_id")
        if doc_id is None:
            # 如果没有 doc_id，用 sample_id 生成一个
            sample_id = sample.get("sample_id", "")
            if "__sent_" in sample_id:
                doc_id = sample_id.split("__sent_")[0]
            else:
                doc_id = f"unknown_{id(sample)}"
        doc_groups[doc_id].append(sample)
    return doc_groups


def split_by_doc_ratio(doc_groups, train_ratio, dev_ratio, test_ratio):
    """
    按 doc_id 划分数据集，确保同一 doc 在同一集合。
    
    Args:
        doc_groups: {doc_id: [samples]}
        train_ratio, dev_ratio, test_ratio: 划分比例
    
    Returns:
        train_data, dev_data, test_data
    """
    doc_ids = list(doc_groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(doc_ids)
    
    n_docs = len(doc_ids)
    n_train = int(n_docs * train_ratio)
    n_dev = int(n_docs * dev_ratio)
    
    train_doc_ids = doc_ids[:n_train]
    dev_doc_ids = doc_ids[n_train:n_train + n_dev]
    test_doc_ids = doc_ids[n_train + n_dev:]
    
    train_data = []
    dev_data = []
    test_data = []
    
    for doc_id in doc_groups:
        samples = doc_groups[doc_id]
        # 按 sent_id 排序，保证顺序一致
        samples_sorted = sorted(samples, key=lambda x: x.get("sent_id", 0))
        
        if doc_id in train_doc_ids:
            train_data.extend(samples_sorted)
        elif doc_id in dev_doc_ids:
            dev_data.extend(samples_sorted)
        else:
            test_data.extend(samples_sorted)
    
    return train_data, dev_data, test_data


def process_dataset(dataset_name):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"处理：{dataset_name}")
    print(f"{'='*60}")
    
    source_dir = os.path.join(SOURCE_ROOT, dataset_name)
    target_dir = os.path.join(TARGET_ROOT, dataset_name)
    
    # 检查源文件
    train_path = os.path.join(source_dir, "train.json")
    test_path = os.path.join(source_dir, "test.json")
    
    if not os.path.exists(train_path):
        print(f"  [SKIP] 训练集不存在：{train_path}")
        return False
    if not os.path.exists(test_path):
        print(f"  [SKIP] 测试集不存在：{test_path}")
        return False
    
    # 加载数据
    print(f"  加载数据...")
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    
    print(f"  原始训练集：{len(train_data)} 样本")
    print(f"  原始测试集：{len(test_data)} 样本")
    
    # 检查原始数据的 doc_id 重叠
    train_docs_orig = set(s.get("doc_id") for s in train_data)
    test_docs_orig = set(s.get("doc_id") for s in test_data)
    overlap_orig = train_docs_orig & test_docs_orig
    print(f"  ⚠️  原始数据重叠文档数：{len(overlap_orig)} (需要修复)")
    
    # 合并数据
    all_data = train_data + test_data
    print(f"  合并后总计：{len(all_data)} 样本")
    
    # 按 doc_id 分组
    doc_groups = group_by_doc_id(all_data)
    n_docs = len(doc_groups)
    print(f"  唯一文档数：{n_docs}")
    
    # 按 doc_id 重新划分
    print(f"  按 doc_id 重新划分 (train:{TRAIN_RATIO}, dev:{DEV_RATIO}, test:{TEST_RATIO})...")
    new_train_data, new_dev_data, new_test_data = split_by_doc_ratio(
        doc_groups, TRAIN_RATIO, DEV_RATIO, TEST_RATIO
    )
    
    # 统计
    train_docs_new = set(s.get("doc_id") for s in new_train_data)
    dev_docs_new = set(s.get("doc_id") for s in new_dev_data)
    test_docs_new = set(s.get("doc_id") for s in new_test_data)
    
    print(f"\n  新训练集：{len(new_train_data)} 样本 ({len(train_docs_new)} 篇文档)")
    print(f"  新开发集：{len(new_dev_data)} 样本 ({len(dev_docs_new)} 篇文档)")
    print(f"  新测试集：{len(new_test_data)} 样本 ({len(test_docs_new)} 篇文档)")
    
    # 检查 doc_id 是否有重叠
    overlap_train_dev = train_docs_new & dev_docs_new
    overlap_train_test = train_docs_new & test_docs_new
    overlap_dev_test = dev_docs_new & test_docs_new
    
    has_error = False
    if overlap_train_dev:
        print(f"  ❌ 错误：训练集和开发集有 {len(overlap_train_dev)} 篇重复文档！")
        has_error = True
    if overlap_train_test:
        print(f"  ❌ 错误：训练集和测试集有 {len(overlap_train_test)} 篇重复文档！")
        has_error = True
    if overlap_dev_test:
        print(f"  ❌ 错误：开发集和测试集有 {len(overlap_dev_test)} 篇重复文档！")
        has_error = True
    
    if has_error:
        return False
    
    print(f"  ✅ 文档无重叠，开始保存...")
    
    # 保存数据
    dump_json(new_train_data, os.path.join(target_dir, "train.json"))
    dump_json(new_dev_data, os.path.join(target_dir, "dev.json"))
    dump_json(new_test_data, os.path.join(target_dir, "test.json"))
    
    print(f"  ✅ 保存完成：{target_dir}")
    return True


def main():
    print("="*60)
    print("从 data_w2ner_folded 复制数据并切分开发集")
    print("="*60)
    print(f"源目录：{SOURCE_ROOT}")
    print(f"目标目录：{TARGET_ROOT}")
    print(f"随机种子：{RANDOM_SEED}")
    
    # 5 块数据
    datasets = [
        "flight_orders_with_queries",
        "hotel_orders_with_queries",
        "id_cards_with_queries",
        "mixed_data_with_queries",
        "train_orders_with_queries",
    ]
    
    # 处理每个数据集
    success_count = 0
    for dataset in datasets:
        if process_dataset(dataset):
            success_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print("处理完成")
    print(f"{'='*60}")
    print(f"成功：{success_count}/{len(datasets)}")
    print(f"目标目录：{TARGET_ROOT}")
    
    # 统计信息
    print(f"\n{'='*60}")
    print("数据集统计")
    print(f"{'='*60}")
    
    total_train = 0
    total_dev = 0
    total_test = 0
    
    for dataset in datasets:
        target_dir = os.path.join(TARGET_ROOT, dataset)
        train_path = os.path.join(target_dir, "train.json")
        dev_path = os.path.join(target_dir, "dev.json")
        test_path = os.path.join(target_dir, "test.json")
        
        if os.path.exists(train_path):
            train_data = load_json(train_path)
            dev_data = load_json(dev_path) if os.path.exists(dev_path) else []
            test_data = load_json(test_path)
            
            total_train += len(train_data)
            total_dev += len(dev_data)
            total_test += len(test_data)
            
            print(f"{dataset}:")
            print(f"  train: {len(train_data)}")
            print(f"  dev:   {len(dev_data)}")
            print(f"  test:  {len(test_data)}")
    
    print(f"\n总计:")
    print(f"  train: {total_train}")
    print(f"  dev:   {total_dev}")
    print(f"  test:  {total_test}")


if __name__ == "__main__":
    main()
