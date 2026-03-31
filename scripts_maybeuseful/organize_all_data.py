#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理所有备份数据，按 doc_id 去重并划分 train/dev/test。

处理的数据链路：
1. data_v1b_backup/ - 原始 RASA 数据（跳过，格式不同）
2. procnet/Data_v1b_backup/ - ProcNet 格式
3. data_w2ner_backup/ - W2NER 格式 (未折叠)
4. data_w2ner_folded_backup/ - W2NER 格式 (折叠)
5. W2NER/data/data_w2ner_backup/ - W2NER 项目数据
6. W2NER/data/data_w2ner_folded_backup/ - W2NER 项目折叠数据

划分策略：
- 按 doc_id 分组，确保同一文档在同一集合
- 划分比例：train 60%, dev 20%, test 20%
"""

import json
import os
import shutil
from collections import defaultdict
import random

# 需要处理的数据目录（按 doc 格式）
W2NER_FORMAT_DIRS = [
    "./data_w2ner_backup",
    "./data_w2ner_folded_backup",
    "./W2NER/data/data_w2ner_backup",
    "./W2NER/data/data_w2ner_folded_backup",
]

# ProcNet 格式目录（需要特殊处理）
PROCNET_FORMAT_DIRS = [
    "./procnet/Data_v1b_backup",
]

# 输出根目录
OUTPUT_ROOT = "./data_organized"

# 划分比例
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2
TEST_RATIO = 0.2

RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def group_w2ner_by_doc_id(samples):
    """按 doc_id 分组 W2NER 格式样本"""
    doc_groups = defaultdict(list)
    for sample in samples:
        doc_id = sample.get("doc_id")
        if doc_id is None:
            sample_id = sample.get("sample_id", "")
            if "__sent_" in sample_id:
                doc_id = sample_id.split("__sent_")[0]
            else:
                doc_id = f"unknown_{id(sample)}"
        doc_groups[doc_id].append(sample)
    return doc_groups


def group_procnet_by_doc_id(procnet_docs):
    """按 doc_id 分组 ProcNet 格式文档"""
    doc_groups = {}
    for doc in procnet_docs:
        doc_id = doc[0] if isinstance(doc, list) else doc.get("doc_id")
        if doc_id:
            doc_groups[doc_id] = doc
    return doc_groups


def split_by_doc_ratio(doc_groups, train_ratio, dev_ratio, test_ratio):
    """按 doc_id 划分数据集"""
    doc_ids = list(doc_groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(doc_ids)
    
    n_docs = len(doc_ids)
    n_train = int(n_docs * train_ratio)
    n_dev = int(n_docs * dev_ratio)
    
    train_doc_ids = set(doc_ids[:n_train])
    dev_doc_ids = set(doc_ids[n_train:n_train + n_dev])
    test_doc_ids = set(doc_ids[n_train + n_dev:])
    
    return train_doc_ids, dev_doc_ids, test_doc_ids


def process_w2ner_dataset(dataset_name, source_dir, output_dir):
    """处理 W2NER 格式数据集"""
    print(f"\n{'='*60}")
    print(f"处理 W2NER 数据：{dataset_name}")
    print(f"{'='*60}")
    
    # 检查源文件
    train_path = os.path.join(source_dir, dataset_name, "train.json")
    test_path = os.path.join(source_dir, dataset_name, "test.json")
    
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
    
    # 检查原始重叠
    train_docs_orig = set(s.get("doc_id") for s in train_data)
    test_docs_orig = set(s.get("doc_id") for s in test_data)
    overlap_orig = train_docs_orig & test_docs_orig
    print(f"  ⚠️  原始重叠文档数：{len(overlap_orig)}")
    
    # 合并数据
    all_data = train_data + test_data
    print(f"  合并后总计：{len(all_data)} 样本")
    
    # 按 doc_id 分组
    doc_groups = group_w2ner_by_doc_id(all_data)
    n_docs = len(doc_groups)
    print(f"  唯一文档数：{n_docs}")
    
    # 按 doc_id 重新划分
    print(f"  按 doc_id 重新划分 (train:{TRAIN_RATIO}, dev:{DEV_RATIO}, test:{TEST_RATIO})...")
    train_doc_ids, dev_doc_ids, test_doc_ids = split_by_doc_ratio(
        doc_groups, TRAIN_RATIO, DEV_RATIO, TEST_RATIO
    )
    
    # 分离数据
    train_data_new = []
    dev_data_new = []
    test_data_new = []
    
    for doc_id, samples in doc_groups.items():
        samples_sorted = sorted(samples, key=lambda x: x.get("sent_id", 0))
        if doc_id in train_doc_ids:
            train_data_new.extend(samples_sorted)
        elif doc_id in dev_doc_ids:
            dev_data_new.extend(samples_sorted)
        else:
            test_data_new.extend(samples_sorted)
    
    # 统计
    train_docs_new = set(s.get("doc_id") for s in train_data_new)
    dev_docs_new = set(s.get("doc_id") for s in dev_data_new)
    test_docs_new = set(s.get("doc_id") for s in test_data_new)
    
    print(f"\n  新训练集：{len(train_data_new)} 样本 ({len(train_docs_new)} 篇文档)")
    print(f"  新开发集：{len(dev_data_new)} 样本 ({len(dev_docs_new)} 篇文档)")
    print(f"  新测试集：{len(test_data_new)} 样本 ({len(test_docs_new)} 篇文档)")
    
    # 检查重叠
    overlap_train_dev = train_docs_new & dev_docs_new
    overlap_train_test = train_docs_new & test_docs_new
    overlap_dev_test = dev_docs_new & test_docs_new
    
    has_error = False
    if overlap_train_dev or overlap_train_test or overlap_dev_test:
        print(f"  ❌ 错误：存在文档重叠！")
        has_error = True
    
    if has_error:
        return False
    
    print(f"  ✅ 文档无重叠，开始保存...")
    
    # 保存数据
    target_dir = os.path.join(output_dir, "w2ner_format", dataset_name)
    dump_json(train_data_new, os.path.join(target_dir, "train.json"))
    dump_json(dev_data_new, os.path.join(target_dir, "dev.json"))
    dump_json(test_data_new, os.path.join(target_dir, "test.json"))
    
    print(f"  ✅ 保存完成：{target_dir}")
    return True


def process_procnet_dataset(dataset_name, source_dir, output_dir):
    """处理 ProcNet 格式数据集"""
    print(f"\n{'='*60}")
    print(f"处理 ProcNet 数据：{dataset_name}")
    print(f"{'='*60}")
    
    train_path = os.path.join(source_dir, dataset_name, "train.json")
    test_path = os.path.join(source_dir, dataset_name, "test.json")
    
    if not os.path.exists(train_path):
        print(f"  [SKIP] 训练集不存在：{train_path}")
        return False
    if not os.path.exists(test_path):
        print(f"  [SKIP] 测试集不存在：{test_path}")
        return False
    
    # 加载 ProcNet 数据
    print(f"  加载数据...")
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    
    print(f"  原始训练集：{len(train_data)} 文档")
    print(f"  原始测试集：{len(test_data)} 文档")
    
    # 合并数据
    all_data = train_data + test_data
    print(f"  合并后总计：{len(all_data)} 文档")
    
    # 按 doc_id 分组
    doc_groups = group_procnet_by_doc_id(all_data)
    n_docs = len(doc_groups)
    print(f"  唯一文档数：{n_docs}")
    
    # 按 doc_id 重新划分
    print(f"  按 doc_id 重新划分...")
    train_doc_ids, dev_doc_ids, test_doc_ids = split_by_doc_ratio(
        doc_groups, TRAIN_RATIO, DEV_RATIO, TEST_RATIO
    )
    
    # 分离数据
    train_data_new = []
    dev_data_new = []
    test_data_new = []
    
    for doc_id, doc in doc_groups.items():
        if doc_id in train_doc_ids:
            train_data_new.append(doc)
        elif doc_id in dev_doc_ids:
            dev_data_new.append(doc)
        else:
            test_data_new.append(doc)
    
    print(f"\n  新训练集：{len(train_data_new)} 文档")
    print(f"  新开发集：{len(dev_data_new)} 文档")
    print(f"  新测试集：{len(test_data_new)} 文档")
    
    # 保存数据
    target_dir = os.path.join(output_dir, "procnet_format", dataset_name)
    dump_json(train_data_new, os.path.join(target_dir, "train.json"))
    dump_json(dev_data_new, os.path.join(target_dir, "dev.json"))
    dump_json(test_data_new, os.path.join(target_dir, "test.json"))
    
    print(f"  ✅ 保存完成：{target_dir}")
    return True


def main():
    print("="*70)
    print("整理所有备份数据，按 doc_id 划分 train/dev/test")
    print("="*70)
    print(f"输出目录：{OUTPUT_ROOT}")
    
    # 获取所有子目录
    all_datasets = set()
    
    for base_dir in W2NER_FORMAT_DIRS:
        if os.path.exists(base_dir):
            for subdir in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, subdir)):
                    all_datasets.add(subdir)
    
    for base_dir in PROCNET_FORMAT_DIRS:
        if os.path.exists(base_dir):
            for subdir in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, subdir)):
                    all_datasets.add(subdir)
    
    print(f"\n发现数据集：{sorted(all_datasets)}")
    
    # 处理 W2NER 格式数据
    print("\n" + "="*70)
    print("【第一部分】处理 W2NER 格式数据")
    print("="*70)
    
    w2ner_success = 0
    for base_dir in W2NER_FORMAT_DIRS:
        if not os.path.exists(base_dir):
            continue
        for dataset in all_datasets:
            if os.path.exists(os.path.join(base_dir, dataset)):
                if process_w2ner_dataset(dataset, base_dir, OUTPUT_ROOT):
                    w2ner_success += 1
    
    # 处理 ProcNet 格式数据
    print("\n" + "="*70)
    print("【第二部分】处理 ProcNet 格式数据")
    print("="*70)
    
    procnet_success = 0
    for base_dir in PROCNET_FORMAT_DIRS:
        if not os.path.exists(base_dir):
            continue
        for dataset in all_datasets:
            if os.path.exists(os.path.join(base_dir, dataset)):
                if process_procnet_dataset(dataset, base_dir, OUTPUT_ROOT):
                    procnet_success += 1
    
    # 总结
    print("\n" + "="*70)
    print("处理完成")
    print("="*70)
    print(f"W2NER 格式成功：{w2ner_success} 个数据集")
    print(f"ProcNet 格式成功：{procnet_success} 个数据集")
    print(f"输出目录：{OUTPUT_ROOT}")
    
    # 统计输出
    print("\n" + "="*70)
    print("输出数据统计")
    print("="*70)
    
    for fmt in ["w2ner_format", "procnet_format"]:
        fmt_dir = os.path.join(OUTPUT_ROOT, fmt)
        if os.path.exists(fmt_dir):
            print(f"\n{fmt}:")
            for dataset in os.listdir(fmt_dir):
                dataset_dir = os.path.join(fmt_dir, dataset)
                if os.path.isdir(dataset_dir):
                    train_path = os.path.join(dataset_dir, "train.json")
                    dev_path = os.path.join(dataset_dir, "dev.json")
                    test_path = os.path.join(dataset_dir, "test.json")
                    
                    if os.path.exists(train_path):
                        train_data = load_json(train_path)
                        dev_data = load_json(dev_path) if os.path.exists(dev_path) else []
                        test_data = load_json(test_path)
                        
                        print(f"  {dataset}:")
                        print(f"    train: {len(train_data)}")
                        print(f"    dev:   {len(dev_data)}")
                        print(f"    test:  {len(test_data)}")


if __name__ == "__main__":
    main()
