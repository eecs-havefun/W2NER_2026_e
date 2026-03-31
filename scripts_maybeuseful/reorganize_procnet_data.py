from pathlib import Path

# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新整理 ProcNet 格式数据，按 doc_id 划分 train/dev/test。
其中 dev 和 test 完全相同（用于过拟合测试）。

原地修改，覆盖原有数据。
"""

import json
import os
from collections import defaultdict
import random

# 配置
BASE_DIR = "./procnet/Data_v1b"

# 划分比例
TRAIN_RATIO = 0.6
DEV_TEST_RATIO = 0.2

RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_train_dev_test(procnet_docs, train_ratio, dev_test_ratio):
    """
    按 doc_id 划分 train/dev/test。
    dev 和 test 完全相同（用于过拟合测试）。
    
    ProcNet 格式：每个文档是 [doc_id, {...}]
    """
    # 按 doc_id 分组
    doc_dict = {}
    for doc in procnet_docs:
        doc_id = doc[0] if isinstance(doc, list) and len(doc) > 0 else doc.get("doc_id")
        if doc_id:
            doc_dict[doc_id] = doc
    
    doc_ids = list(doc_dict.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(doc_ids)
    
    n_docs = len(doc_ids)
    n_train = int(n_docs * train_ratio)
    
    # 划分：train 和 dev_test
    train_doc_ids = set(doc_ids[:n_train])
    dev_test_doc_ids = set(doc_ids[n_train:])
    
    # 分离数据
    train_data = []
    dev_test_data = []
    
    for doc_id in doc_dict:
        doc = doc_dict[doc_id]
        if doc_id in train_doc_ids:
            train_data.append(doc)
        else:
            dev_test_data.append(doc)
    
    # dev 和 test 完全相同
    dev_data = dev_test_data.copy()
    test_data = dev_test_data.copy()
    
    return train_data, dev_data, test_data


def process_dataset(dataset_name):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"处理：{BASE_DIR}/{dataset_name}")
    print(f"{'='*60}")
    
    train_path = os.path.join(BASE_DIR, dataset_name, "train.json")
    test_path = os.path.join(BASE_DIR, dataset_name, "test.json")
    
    if not os.path.exists(train_path):
        print(f"  [SKIP] 训练集不存在")
        return False
    if not os.path.exists(test_path):
        print(f"  [SKIP] 测试集不存在")
        return False
    
    # 加载数据
    print(f"  加载原始数据...")
    train_data = load_json(train_path)
    test_data = load_json(test_path)
    
    # ProcNet 格式：每个文档是 [doc_id, {...}]
    train_doc_ids = set(doc[0] if isinstance(doc, list) else doc.get("doc_id") for doc in train_data)
    test_doc_ids = set(doc[0] if isinstance(doc, list) else doc.get("doc_id") for doc in test_data)
    
    print(f"  原始训练集：{len(train_data)} 文档 ({len(train_doc_ids)} 篇唯一文档)")
    print(f"  原始测试集：{len(test_data)} 文档 ({len(test_doc_ids)} 篇唯一文档)")
    
    # 检查原始重叠
    overlap = train_doc_ids & test_doc_ids
    if overlap:
        print(f"  ⚠️  原始数据有 {len(overlap)} 篇文档重叠")
    
    # 合并数据
    all_data = train_data + test_data
    
    # 去重（按 doc_id）
    doc_dict = {}
    for doc in all_data:
        doc_id = doc[0] if isinstance(doc, list) else doc.get("doc_id")
        if doc_id:
            doc_dict[doc_id] = doc
    
    n_docs = len(doc_dict)
    print(f"  合并去重后：{n_docs} 篇唯一文档")
    
    # 重新划分
    print(f"  重新划分 (train:{TRAIN_RATIO}, dev=test:{DEV_TEST_RATIO})...")
    train_data_new, dev_data_new, test_data_new = split_train_dev_test(
        list(doc_dict.values()), TRAIN_RATIO, DEV_TEST_RATIO
    )
    
    # 验证
    train_doc_ids_new = set(doc[0] if isinstance(doc, list) else doc.get("doc_id") for doc in train_data_new)
    dev_doc_ids_new = set(doc[0] if isinstance(doc, list) else doc.get("doc_id") for doc in dev_data_new)
    test_doc_ids_new = set(doc[0] if isinstance(doc, list) else doc.get("doc_id") for doc in test_data_new)
    
    print(f"\n  新训练集：{len(train_data_new)} 文档 ({len(train_doc_ids_new)} 篇)")
    print(f"  新开发集：{len(dev_data_new)} 文档 ({len(dev_doc_ids_new)} 篇)")
    print(f"  新测试集：{len(test_data_new)} 文档 ({len(test_doc_ids_new)} 篇)")
    
    # 检查 dev 和 test 是否完全相同
    dev_test_same = (dev_data_new == test_data_new)
    
    # 检查 train 与 dev/test 是否有重叠
    overlap_train_dev = train_doc_ids_new & dev_doc_ids_new
    overlap_train_test = train_doc_ids_new & test_doc_ids_new
    overlap_dev_test = dev_doc_ids_new & test_doc_ids_new
    
    has_error = False
    if not dev_test_same:
        print(f"  ❌ dev 和 test 不相同！")
        has_error = True
    if overlap_train_dev or overlap_train_test:
        print(f"  ❌ train 与 dev/test 有重叠！")
        has_error = True
    if overlap_dev_test != dev_doc_ids_new:
        print(f"  ❌ dev 和 test 文档不完全相同！")
        has_error = True
    
    if has_error:
        return False
    
    print(f"  ✅ dev == test，文档无重叠，开始保存...")
    
    # 原地覆盖保存
    dump_json(train_data_new, train_path)
    dump_json(dev_data_new, os.path.join(BASE_DIR, dataset_name, "dev.json"))
    dump_json(test_data_new, test_path)
    
    print(f"  ✅ 保存完成（已覆盖原文件）")
    return True


def main():
    print("="*70)
    print("重新整理 ProcNet 数据：dev 和 test 完全相同（用于过拟合测试）")
    print("="*70)
    print(f"随机种子：{RANDOM_SEED}")
    print(f"划分比例：train={TRAIN_RATIO}, dev=test={DEV_TEST_RATIO}")
    
    # 5 个数据集
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
        if os.path.exists(os.path.join(BASE_DIR, dataset)):
            if process_dataset(dataset):
                success_count += 1
    
    # 总结
    print("\n" + "="*70)
    print("处理完成")
    print("="*70)
    print(f"成功：{success_count}/{len(datasets)} 个数据集")
    
    # 统计结果
    print("\n" + "="*70)
    print("数据统计")
    print("="*70)
    
    for dataset in datasets:
        train_path = os.path.join(BASE_DIR, dataset, "train.json")
        dev_path = os.path.join(BASE_DIR, dataset, "dev.json")
        test_path = os.path.join(BASE_DIR, dataset, "test.json")
        
        if os.path.exists(train_path):
            train_data = load_json(train_path)
            dev_data = load_json(dev_path)
            test_data = load_json(test_path)
            
            print(f"\n{dataset}:")
            print(f"  train: {len(train_data)}")
            print(f"  dev:   {len(dev_data)}")
            print(f"  test:  {len(test_data)}")
            
            # 验证 dev 和 test 是否相同
            if dev_data == test_data:
                print(f"  ✅ dev == test")
            else:
                print(f"  ❌ dev != test")


if __name__ == "__main__":
    main()
