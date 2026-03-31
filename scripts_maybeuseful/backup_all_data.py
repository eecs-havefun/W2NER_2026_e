#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
备份数据链路上的所有 JSON 文件。

数据链路：
1. data_v1b/ - 原始 RASA 数据
2. procnet/Data_v1b/ - ProcNet 格式数据
3. procnet/Data/ - ProcNet 其他数据
4. data_w2ner/ - W2NER 格式数据 (未折叠)
5. data_w2ner_folded/ - W2NER 格式数据 (折叠)
6. W2NER/data/data_w2ner/ - W2NER 项目中的数据
7. W2NER/data/data_w2ner_folded/ - W2NER 项目中的折叠数据
8. W2NER/data/data_w2ner_folded_with_dev/ - W2NER 项目中带 dev 的折叠数据
"""

import json
import os
import shutil
from pathlib import Path

# 需要备份的目录列表
DATA_DIRS = [
    "./data_v1b",
    "./data_w2ner",
    "./data_w2ner_folded",
    "./procnet/Data",
    "./procnet/Data_v1b",
    "./W2NER/data/data_w2ner",
    "./W2NER/data/data_w2ner_folded",
    "./W2NER/data/data_w2ner_folded_with_dev",
]

# 备份后缀
BACKUP_SUFFIX = "_backup"


def count_json_files(dir_path):
    """统计目录中的 JSON 文件数量"""
    count = 0
    for root, dirs, files in os.walk(dir_path):
        # 跳过 __pycache__ 和 .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
        for f in files:
            if f.endswith('.json'):
                count += 1
    return count


def backup_directory(src_dir):
    """备份目录"""
    src_path = Path(src_dir)
    
    if not src_path.exists():
        print(f"  [SKIP] {src_dir} 不存在")
        return False
    
    # 创建备份目录名
    backup_name = src_path.name + BACKUP_SUFFIX
    backup_path = src_path.parent / backup_name
    
    # 如果备份已存在，先删除
    if backup_path.exists():
        print(f"  删除旧备份：{backup_path}")
        shutil.rmtree(backup_path)
    
    # 复制目录
    print(f"  复制 {src_dir} -> {backup_path}")
    shutil.copytree(src_path, backup_path, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '.git'
    ))
    
    # 统计
    src_count = count_json_files(src_path)
    backup_count = count_json_files(backup_path)
    
    print(f"  ✅ 备份完成：{src_count} 个 JSON 文件 -> {backup_count} 个 JSON 文件")
    return True


def main():
    print("="*70)
    print("备份数据链路上的所有 JSON 文件")
    print("="*70)
    
    # 统计
    total_src_files = 0
    total_backup_files = 0
    success_count = 0
    
    for dir_path in DATA_DIRS:
        print(f"\n处理：{dir_path}")
        print("-" * 50)
        
        if backup_directory(dir_path):
            success_count += 1
            total_src_files += count_json_files(dir_path)
    
    # 总结
    print("\n" + "="*70)
    print("备份完成")
    print("="*70)
    print(f"成功备份：{success_count}/{len(DATA_DIRS)} 个目录")
    print(f"源数据 JSON 文件总数：{total_src_files}")
    
    # 列出所有备份
    print("\n备份目录列表:")
    for dir_path in DATA_DIRS:
        src_path = Path(dir_path)
        backup_path = src_path.parent / (src_path.name + BACKUP_SUFFIX)
        if backup_path.exists():
            count = count_json_files(backup_path)
            print(f"  {backup_path}: {count} 个 JSON 文件")


if __name__ == "__main__":
    main()
