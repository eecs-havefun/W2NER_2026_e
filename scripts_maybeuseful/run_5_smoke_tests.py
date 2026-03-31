#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 5 个数据集分别创建 smoke test 配置文件并运行
使用符号链接避免路径斜杠问题
"""

import json
import subprocess
import os

BASE_CONFIG = {
    "dataset": "",
    "save_path": "",
    "predict_path": "",
    "dist_emb_size": 20,
    "type_emb_size": 20,
    "lstm_hid_size": 256,
    "conv_hid_size": 64,
    "bert_hid_size": 768,
    "biaffine_size": 256,
    "ffnn_hid_size": 128,
    "dilation": [1, 2, 3],
    "emb_dropout": 0.5,
    "conv_dropout": 0.5,
    "out_dropout": 0.33,
    "epochs": 1,
    "batch_size": 12,
    "learning_rate": 0.001,
    "weight_decay": 0,
    "clip_grad_norm": 5.0,
    "bert_name": "../models/bert_base_chinese",
    "bert_learning_rate": 5e-06,
    "warm_factor": 0.1,
    "use_bert_last_4_layers": True,
    "seed": 123
}

DATASETS = [
    ("flight_orders", "data_w2ner_folded_with_dev/flight_orders_with_queries"),
    ("hotel_orders", "data_w2ner_folded_with_dev/hotel_orders_with_queries"),
    ("id_cards", "data_w2ner_folded_with_dev/id_cards_with_queries"),
    ("mixed_data", "data_w2ner_folded_with_dev/mixed_data_with_queries"),
    ("train_orders", "data_w2ner_folded_with_dev/train_orders_with_queries"),
]

CONFIG_DIR = "/home/mengfanrong/finaldesign/W2NERproject/W2NER/config"
W2NER_DIR = "/home/mengfanrong/finaldesign/W2NERproject/W2NER"
W2NER_DATA_DIR = "/home/mengfanrong/finaldesign/W2NERproject/W2NER/data"

def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # 创建符号链接
    for short_name, actual_path in DATASETS:
        link_path = os.path.join(W2NER_DATA_DIR, short_name)
        target_path = os.path.join(W2NER_DATA_DIR, actual_path)
        
        if os.path.islink(link_path):
            os.unlink(link_path)
            print(f"Removed existing symlink: {link_path}")
        elif os.path.exists(link_path):
            print(f"Warning: {link_path} exists but is not a symlink")
        
        os.symlink(target_path, link_path)
        print(f"Created symlink: {link_path} -> {target_path}")
    
    print("=" * 60)
    
    for short_name, actual_path in DATASETS:
        config = BASE_CONFIG.copy()
        config["dataset"] = short_name
        config["save_path"] = f"{short_name}_model.pt"
        config["predict_path"] = f"{short_name}_output.json"
        
        config_path = os.path.join(CONFIG_DIR, f"{short_name}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\nCreated config: {config_path}")
        
        # 运行训练
        cmd = [
            "python", "main.py",
            "--config", f"config/{short_name}.json",
            "--device", "0"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, cwd=W2NER_DIR, capture_output=False)
        
        if result.returncode == 0:
            print(f"\n[OK] {short_name} completed successfully")
        else:
            print(f"\n[FAIL] {short_name} failed with code {result.returncode}")
        
        print("=" * 60)

if __name__ == "__main__":
    main()
