#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for W2NER data validation.

从 5 块数据中各抽取少量样本，合并成一个小型测试集，
然后用 W2NER 训练 1 个 epoch，验证数据格式是否正确。
"""

import json
import os
import random
import shutil

# 配置
DATA_W2NER_ROOT = "./data_w2ner"
SMOKE_DATA_ROOT = "./W2NER/data/smoke_test"
SMOKE_CONFIG_PATH = "./W2NER/config/smoke_test.json"

# 从每块数据中抽取的样本数
SAMPLES_PER_DATASET = 50
RANDOM_SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sample_data(input_path, n_samples):
    """随机抽取 n 个样本"""
    data = load_json(input_path)
    if len(data) <= n_samples:
        return data
    random.seed(RANDOM_SEED)
    return random.sample(data, n_samples)


def main():
    print("=" * 60)
    print("W2NER Smoke Test Data Preparation")
    print("=" * 60)

    # 5 块数据
    datasets = [
        "flight_orders_with_queries",
        "hotel_orders_with_queries",
        "id_cards_with_queries",
        "mixed_data_with_queries",
        "train_orders_with_queries",
    ]

    # 收集所有样本
    train_samples = []
    dev_samples = []
    test_samples = []

    for dataset in datasets:
        train_path = os.path.join(DATA_W2NER_ROOT, dataset, "train.json")
        test_path = os.path.join(DATA_W2NER_ROOT, dataset, "test.json")

        if not os.path.exists(train_path):
            print(f"[SKIP] {dataset}: train.json not found")
            continue
        if not os.path.exists(test_path):
            print(f"[SKIP] {dataset}: test.json not found")
            continue

        # 抽取样本
        train_subset = sample_data(train_path, SAMPLES_PER_DATASET)
        test_subset = sample_data(test_path, SAMPLES_PER_DATASET // 2)

        train_samples.extend(train_subset)
        test_samples.extend(test_subset)

        print(f"[OK] {dataset}: {len(train_subset)} train + {len(test_subset)} test")

    # 划分 dev set (从 train 中再分一部分)
    dev_size = max(1, len(train_samples) // 5)
    dev_samples = train_samples[:dev_size]
    train_samples = train_samples[dev_size:]

    print(f"\n总计:")
    print(f"  train: {len(train_samples)}")
    print(f"  dev:   {len(dev_samples)}")
    print(f"  test:  {len(test_samples)}")

    # 写入 smoke test 数据目录
    os.makedirs(SMOKE_DATA_ROOT, exist_ok=True)
    dump_json(train_samples, os.path.join(SMOKE_DATA_ROOT, "train.json"))
    dump_json(dev_samples, os.path.join(SMOKE_DATA_ROOT, "dev.json"))
    dump_json(test_samples, os.path.join(SMOKE_DATA_ROOT, "test.json"))

    print(f"\n[OK] Smoke test data saved to {SMOKE_DATA_ROOT}")

    # 创建 smoke test 配置文件
    smoke_config = {
        "dataset": "smoke_test",
        "save_path": "smoke_test_model.pt",
        "predict_path": "smoke_test_output.json",
        "dist_emb_size": 20,
        "type_emb_size": 20,
        "lstm_hid_size": 256,
        "conv_hid_size": 64,
        "bert_hid_size": 768,
        "biaffine_size": 256,
        "ffnn_hid_size": 128,
        "dilation": [1, 2],
        "emb_dropout": 0.5,
        "conv_dropout": 0.5,
        "out_dropout": 0.33,
        "epochs": 1,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "clip_grad_norm": 5.0,
        "bert_name": "../models/bert_base_chinese",
        "bert_learning_rate": 5e-6,
        "warm_factor": 0.1,
        "use_bert_last_4_layers": True,
        "seed": 123
    }

    dump_json(smoke_config, SMOKE_CONFIG_PATH)
    print(f"[OK] Smoke test config saved to {SMOKE_CONFIG_PATH}")

    # 打印统计信息
    all_entity_types = set()
    for sample in train_samples + dev_samples + test_samples:
        for ent in sample.get("ner", []):
            all_entity_types.add(ent["type"])

    print(f"\n实体类型 ({len(all_entity_types)} 种):")
    for t in sorted(all_entity_types):
        print(f"  - {t}")

    print("\n" + "=" * 60)
    print("下一步：运行 W2NER 训练")
    print("=" * 60)
    print(f"""
cd W2NER
python main.py --config config/smoke_test.json --device 0
""")


if __name__ == "__main__":
    main()
