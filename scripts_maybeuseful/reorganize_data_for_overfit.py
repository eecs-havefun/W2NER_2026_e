from pathlib import Path

# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新整理数据，按 doc_id 划分 train/dev/test。
其中 dev 和 test 完全相同（用于过拟合测试）。

原地修改，覆盖原有数据。
"""

import json
import os
from collections import defaultdict
import random

# 配置
DATA_DIRS = [
    "./data_w2ner",
    "./data_w2ner_folded",
]

# 划分比例
TRAIN_RATIO = 0.6
DEV_TEST_RATIO = 0.2  # dev 和 test 各占 20%

RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def group_by_doc_id(samples):
    """按 doc_id 分组样本"""
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


def split_train_dev_test(doc_groups, train_ratio, dev_test_ratio):
    """
    按 doc_id 划分 train/dev/test。
    dev 和 test 完全相同（用于过拟合测试）。
    """
    doc_ids = list(doc_groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(doc_ids)
    
    n_docs = len(doc_ids)
    n_train = int(n_docs * train_ratio)
    n_dev_test = n_docs - n_train  # 剩余的给 dev 和 test
    
    # 划分：train 和 dev_test
    train_doc_ids = set(doc_ids[:n_train])
    dev_test_doc_ids = set(doc_ids[n_train:])
    
    # 分离数据
    train_data = []
    dev_test_data = []
    
    for doc_id, samples in doc_groups.items():
        # 按 sent_id 排序
        samples_sorted = sorted(samples, key=lambda x: x.get("sent_id", 0))
        if doc_id in train_doc_ids:
            train_data.extend(samples_sorted)
        else:
            dev_test_data.extend(samples_sorted)
    
    # dev 和 test 完全相同
    dev_data = dev_test_data.copy()
    test_data = dev_test_data.copy()
    
    return train_data, dev_data, test_data


def process_dataset(base_dir, dataset_name):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"处理：{base_dir}/{dataset_name}")
    print(f"{'='*60}")
    
    train_path = os.path.join(base_dir, dataset_name, "train.json")
    test_path = os.path.join(base_dir, dataset_name, "test.json")
    
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
    
    orig_train_docs = len(set(s.get("doc_id") for s in train_data))
    orig_test_docs = len(set(s.get("doc_id") for s in test_data))
    print(f"  原始训练集：{len(train_data)} 样本 ({orig_train_docs} 篇文档)")
    print(f"  原始测试集：{len(test_data)} 样本 ({orig_test_docs} 篇文档)")
    
    # 检查原始重叠
    train_doc_set = set(s.get("doc_id") for s in train_data)
    test_doc_set = set(s.get("doc_id") for s in test_data)
    overlap = train_doc_set & test_doc_set
    if overlap:
        print(f"  ⚠️  原始数据有 {len(overlap)} 篇文档重叠")
    
    # 合并数据
    all_data = train_data + test_data
    print(f"  合并后：{len(all_data)} 样本")
    
    # 按 doc_id 分组
    doc_groups = group_by_doc_id(all_data)
    n_docs = len(doc_groups)
    print(f"  唯一文档数：{n_docs}")
    
    # 重新划分
    print(f"  重新划分 (train:{TRAIN_RATIO}, dev=test:{DEV_TEST_RATIO})...")
    train_data_new, dev_data_new, test_data_new = split_train_dev_test(
        doc_groups, TRAIN_RATIO, DEV_TEST_RATIO
    )
    
    # 验证
    train_docs_new = set(s.get("doc_id") for s in train_data_new)
    dev_docs_new = set(s.get("doc_id") for s in dev_data_new)
    test_docs_new = set(s.get("doc_id") for s in test_data_new)
    
    print(f"\n  新训练集：{len(train_data_new)} 样本 ({len(train_docs_new)} 篇文档)")
    print(f"  新开发集：{len(dev_data_new)} 样本 ({len(dev_docs_new)} 篇文档)")
    print(f"  新测试集：{len(test_data_new)} 样本 ({len(test_docs_new)} 篇文档)")
    
    # 检查 dev 和 test 是否完全相同
    dev_texts = set(s.get("text") for s in dev_data_new)
    test_texts = set(s.get("text") for s in test_data_new)
    
    if dev_texts == test_texts:
        print(f"  ✅ dev 和 test 完全相同")
    else:
        print(f"  ❌ dev 和 test 不同！")
        return False
    
    # 检查 train 与 dev/test 是否有重叠
    overlap_train_dev = train_docs_new & dev_docs_new
    overlap_train_test = train_docs_new & test_docs_new
    
    if overlap_train_dev or overlap_train_test:
        print(f"  ❌ train 与 dev/test 有重叠！")
        return False
    
    print(f"  ✅ 文档无重叠，开始保存...")
    
    # 原地覆盖保存
    dump_json(train_data_new, train_path)
    dump_json(dev_data_new, os.path.join(base_dir, dataset_name, "dev.json"))
    dump_json(test_data_new, test_path)
    
    print(f"  ✅ 保存完成（已覆盖原文件）")
    return True


def main():
    print("="*70)
    print("重新整理数据：dev 和 test 完全相同（用于过拟合测试）")
    print("="*70)
    print(f"随机种子：{RANDOM_SEED}")
    print(f"划分比例：train={TRAIN_RATIO}, dev=test={DEV_TEST_RATIO}")
    
    # 处理每个数据集
    success_count = 0
    total_count = 0
    
    datasets = [
        "flight_orders_with_queries",
        "hotel_orders_with_queries",
        "id_cards_with_queries",
        "mixed_data_with_queries",
        "train_orders_with_queries",
    ]
    
    for base_dir in DATA_DIRS:
        if not os.path.exists(base_dir):
            print(f"\n[SKIP] {base_dir} 不存在")
            continue
        
        print(f"\n{'='*70}")
        print(f"处理目录：{base_dir}")
        print(f"{'='*70}")
        
        for dataset in datasets:
            if os.path.exists(os.path.join(base_dir, dataset)):
                total_count += 1
                if process_dataset(base_dir, dataset):
                    success_count += 1
    
    # 总结
    print("\n" + "="*70)
    print("处理完成")
    print("="*70)
    print(f"成功：{success_count}/{total_count} 个数据集")
    
    # 统计结果
    print("\n" + "="*70)
    print("数据统计")
    print("="*70)
    
    for base_dir in DATA_DIRS:
        if os.path.exists(base_dir):
            print(f"\n【{base_dir}】")
            for dataset in datasets:
                train_path = os.path.join(base_dir, dataset, "train.json")
                dev_path = os.path.join(base_dir, dataset, "dev.json")
                test_path = os.path.join(base_dir, dataset, "test.json")
                
                if os.path.exists(train_path):
                    train_data = load_json(train_path)
                    dev_data = load_json(dev_path)
                    test_data = load_json(test_path)
                    
                    print(f"  {dataset}:")
                    print(f"    train: {len(train_data)}")
                    print(f"    dev:   {len(dev_data)}")
                    print(f"    test:  {len(test_data)}")
                    
                    # 验证 dev 和 test 是否相同
                    if dev_data == test_data:
                        print(f"    ✅ dev == test")
                    else:
                        print(f"    ❌ dev != test")


if __name__ == "__main__":
    main()
