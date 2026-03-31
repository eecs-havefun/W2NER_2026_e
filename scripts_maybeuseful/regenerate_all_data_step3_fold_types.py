
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 W2NER 折叠格式数据

从已生成的未折叠数据复制，然后折叠角色类型：
- startDate, endDate → date
- startTime, endTime → time

覆盖目标目录：
- data_w2ner/
- data_w2ner_folded/
"""

import json
from pathlib import Path
from typing import Dict, List, Any

W2NER_SOURCE_ROOT = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev"
W2NER_FOLDED_OUTPUT = project_root / "data_w2ner_folded"
W2NER_ALT_OUTPUT = project_root / "data_w2ner"

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

SPLITS = ["train", "dev", "test"]

# 角色折叠映射
ROLE_FOLD_MAP = {
    "startDate": "date",
    "endDate": "date",
    "startTime": "time",
    "endTime": "time",
}


def load_w2ner_data(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_w2ner_data(data: List, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fold_sample(sample: Dict) -> Dict:
    """折叠单个样本的实体类型"""
    new_sample = dict(sample)
    
    # 折叠 ner 中的类型
    new_ner = []
    for ent in sample.get("ner", []):
        new_ent = dict(ent)
        orig_type = ent.get("type", "")
        new_ent["type"] = ROLE_FOLD_MAP.get(orig_type, orig_type)
        new_ner.append(new_ent)
    
    new_sample["ner"] = new_ner
    
    # 折叠 entities 中的类型
    new_entities = []
    for ent in sample.get("entities", []):
        new_ent = dict(ent)
        orig_type = ent.get("type_name", "")
        new_ent["type_name"] = ROLE_FOLD_MAP.get(orig_type, orig_type)
        new_entities.append(new_ent)
    
    new_sample["entities"] = new_entities
    
    return new_sample


def fold_dataset(dataset: str, output_root: Path):
    """折叠单个数据集"""
    print(f"\n处理：{dataset} → {output_root.name}")
    print("-" * 60)
    
    stats = {}
    
    for split in SPLITS:
        source_path = W2NER_SOURCE_ROOT / dataset / f"{split}.json"
        output_path = output_root / dataset / f"{split}.json"
        
        if not source_path.exists():
            print(f"  ⚠️  {split}: 源文件不存在")
            continue
        
        # 加载数据
        data = load_w2ner_data(source_path)
        
        # 折叠
        folded_data = [fold_sample(s) for s in data]
        
        # 保存
        save_w2ner_data(folded_data, output_path)
        
        # 统计
        entity_count = sum(len(s.get("ner", [])) for s in folded_data)
        stats[split] = {"samples": len(folded_data), "entities": entity_count}
        
        print(f"  {split}: {len(folded_data)} 句子，{entity_count} 实体")
    
    return stats


def main():
    print("=" * 80)
    print("生成 W2NER 折叠格式数据")
    print("=" * 80)
    
    for output_root in [W2NER_FOLDED_OUTPUT, W2NER_ALT_OUTPUT]:
        print(f"\n{'='*80}")
        print(f"输出目录：{output_root}")
        print("=" * 80)
        
        all_stats = {}
        
        for dataset in DATASETS:
            all_stats[dataset] = fold_dataset(dataset, output_root)
        
        print("\n" + "=" * 80)
        print(f"{output_root.name} 汇总:")
        print(f"{'数据集':<35} {'train':>15} {'dev':>15} {'test':>15}")
        print("-" * 80)
        for dataset, stats in all_stats.items():
            train_str = f"{stats['train']['samples']} 句/{stats['train']['entities']} 实"
            dev_str = f"{stats['dev']['samples']} 句/{stats['dev']['entities']} 实"
            test_str = f"{stats['test']['samples']} 句/{stats['test']['entities']} 实"
            print(f"{dataset:<35} {train_str:>15} {dev_str:>15} {test_str:>15}")
    
    print("\n" + "=" * 80)
    print("生成完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
