#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 W2NER/data 下各子目录的 dev/test 是否相同
"""

import hashlib
from pathlib import Path

def md5_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

W2NER_DATA_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data")

print("=== 检查 W2NER/data 下各子目录的 dev/test ===")
for subdir in sorted(W2NER_DATA_ROOT.iterdir()):
    if not subdir.is_dir():
        continue
    
    dev_path = subdir / "dev.json"
    test_path = subdir / "test.json"
    
    if dev_path.exists() and test_path.exists():
        dev_md5 = md5_file(dev_path)
        test_md5 = md5_file(test_path)
        
        status = "SAME ❌" if dev_md5 == test_md5 else "DIFF ✓"
        print(f"{status}: {subdir.name}")
    else:
        # 可能是包含 5 个子数据集的目录
        datasets = ["flight_orders_with_queries", "hotel_orders_with_queries", 
                    "id_cards_with_queries", "mixed_data_with_queries", "train_orders_with_queries"]
        has_datasets = all((subdir / ds / "dev.json").exists() for ds in datasets)
        if has_datasets:
            print(f"\n--- {subdir.name} (contains 5 datasets) ---")
            all_diff = True
            for dataset in datasets:
                ds_dev = subdir / dataset / "dev.json"
                ds_test = subdir / dataset / "test.json"
                if ds_dev.exists() and ds_test.exists():
                    d_md5 = md5_file(ds_dev)
                    t_md5 = md5_file(ds_test)
                    status = "SAME ❌" if d_md5 == t_md5 else "DIFF ✓"
                    if d_md5 == t_md5:
                        all_diff = False
                    print(f"  {status}: {dataset}")
