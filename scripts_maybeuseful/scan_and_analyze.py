
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描项目中的所有数据，分析冗余和可删除的文件。
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# 需要扫描的根目录
ROOT_DIR = "."

# 忽略的目录
IGNORE_DIRS = {'.git', '__pycache__', '.qwen', 'models', 'cache', 'wheelhouse', 
               'log', 'logs', 'figures', 'exports', 'sidecar', 'tmp_sidecar',
               'procnet/procnet', 'W2NER/W2NER'}


def get_file_size(path):
    """获取文件大小（MB）"""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except:
        return 0


def count_json_lines(path):
    """统计 JSON 文件中的样本数"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                if 'rasa_nlu_data' in data:
                    return len(data['rasa_nlu_data'].get('common_examples', []))
                elif 'common_examples' in data:
                    return len(data['common_examples'])
                else:
                    return 1
    except:
        return -1


def scan_directories():
    """扫描所有目录"""
    print("="*80)
    print("扫描项目中的所有数据目录")
    print("="*80)
    
    data_dirs = []
    total_size = 0
    
    for root, dirs, files in os.walk(ROOT_DIR):
        # 跳过忽略的目录
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        # 查找数据目录
        if any(keyword in root for keyword in ['data_', 'Data_', '_backup', '_organized']):
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                dir_size = sum(get_file_size(os.path.join(root, f)) for f in json_files)
                total_size += dir_size
                data_dirs.append({
                    'path': root,
                    'json_count': len(json_files),
                    'size_mb': dir_size,
                    'files': json_files
                })
    
    # 按路径排序
    data_dirs.sort(key=lambda x: x['path'])
    
    return data_dirs, total_size


def analyze_data_chains(data_dirs):
    """分析数据链路"""
    print("\n" + "="*80)
    print("数据链路分析")
    print("="*80)
    
    # 按数据集名称分组
    datasets = defaultdict(list)
    
    for dir_info in data_dirs:
        path = dir_info['path']
        parts = path.strip('./').split('/')
        
        # 提取数据集名称
        if len(parts) >= 2:
            dataset_name = parts[-1]
            data_type = parts[-2] if 'backup' not in parts[-1] else 'backup'
            
            datasets[dataset_name].append({
                'path': path,
                'type': data_type,
                'json_count': dir_info['json_count'],
                'size_mb': dir_info['size_mb']
            })
    
    return datasets


def identify_redundant(data_dirs):
    """识别冗余数据"""
    print("\n" + "="*80)
    print("冗余数据分析")
    print("="*80)
    
    redundant = []
    can_delete = []
    
    for dir_info in data_dirs:
        path = dir_info['path']
        
        # 1. 备份目录（已整理后可删除）
        if '_backup' in path:
            can_delete.append({
                'path': path,
                'reason': '备份目录（原始数据已整理到 data_organized）',
                'size_mb': dir_info['size_mb'],
                'json_count': dir_info['json_count']
            })
        
        # 2. 原始未整理数据（建议删除，保留整理后的）
        elif path.startswith('./data_w2ner') or path.startswith('./data_w2ner_folded'):
            if path not in ['./data_w2ner', './data_w2ner_folded']:
                can_delete.append({
                    'path': path,
                    'reason': '原始数据（已整理到 data_organized）',
                    'size_mb': dir_info['size_mb'],
                    'json_count': dir_info['json_count']
                })
        
        # 3. W2NER 项目中的重复数据
        elif 'W2NER/data/' in path and '_backup' not in path:
            redundant.append({
                'path': path,
                'reason': 'W2NER 项目中的数据副本（与根目录 data_w2ner* 重复）',
                'size_mb': dir_info['size_mb'],
                'json_count': dir_info['json_count']
            })
        
        # 4. procnet 中的备份数据
        elif 'procnet/Data' in path and '_backup' in path:
            can_delete.append({
                'path': path,
                'reason': 'ProcNet 备份数据（已整理到 data_organized）',
                'size_mb': dir_info['size_mb'],
                'json_count': dir_info['json_count']
            })
    
    return redundant, can_delete


def print_summary(data_dirs, total_size, redundant, can_delete):
    """打印总结"""
    print("\n" + "="*80)
    print("数据总结")
    print("="*80)
    
    print(f"\n📊 总览:")
    print(f"   数据目录数：{len(data_dirs)}")
    print(f"   总大小：{total_size:.2f} MB")
    
    print(f"\n📁 数据目录列表:")
    for dir_info in data_dirs:
        print(f"   {dir_info['path']}: {dir_info['json_count']} 个 JSON, {dir_info['size_mb']:.2f} MB")
    
    print(f"\n⚠️  冗余数据（{len(redundant)} 个目录）:")
    for item in redundant:
        print(f"   {item['path']}")
        print(f"      原因：{item['reason']}")
        print(f"      大小：{item['size_mb']:.2f} MB, {item['json_count']} 个 JSON")
    
    print(f"\n🗑️  可删除数据（{len(can_delete)} 个目录）:")
    can_delete_size = sum(item['size_mb'] for item in can_delete)
    can_delete_json = sum(item['json_count'] for item in can_delete)
    for item in can_delete:
        print(f"   {item['path']}")
        print(f"      原因：{item['reason']}")
        print(f"      大小：{item['size_mb']:.2f} MB, {item['json_count']} 个 JSON")
    
    print(f"\n   可删除总计：{can_delete_size:.2f} MB, {can_delete_json} 个 JSON 文件")
    
    print("\n" + "="*80)
    print("建议保留的数据")
    print("="*80)
    print("""
   1. data_v1b/ - 原始 RASA 标注数据（源头）
   2. data_organized/ - 整理后的数据（推荐使用）
      - w2ner_format/ - W2NER 格式，已划分 train/dev/test
      - procnet_format/ - ProcNet 格式，已划分 train/dev/test
   3. W2NER/data/data_w2ner_folded_with_dev/ - W2NER 项目专用数据
   
   其他大部分是备份或中间产物，可以删除。
    """)


def main():
    # 扫描
    data_dirs, total_size = scan_directories()
    
    # 分析
    datasets = analyze_data_chains(data_dirs)
    
    # 识别冗余
    redundant, can_delete = identify_redundant(data_dirs)
    
    # 总结
    print_summary(data_dirs, total_size, redundant, can_delete)
    
    # 生成删除脚本
    print("\n" + "="*80)
    print("删除脚本（请谨慎执行）")
    print("="*80)
    print("\n# 可删除的目录:")
    for item in can_delete:
        print(f"rm -rf {item['path']}")
    print("\n# 冗余的目录（W2NER 项目中的副本）:")
    for item in redundant:
        print(f"rm -rf {item['path']}")


if __name__ == "__main__":
    main()
