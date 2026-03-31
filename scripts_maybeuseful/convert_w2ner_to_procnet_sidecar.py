
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 W2NER 预测输出转换为 ProcNet Sidecar 实体格式

输入：W2NER output.json (预测结果)
输出：ProcNet sidecar JSONL 格式

设计原则：
1. 保留 doc_id, sent_id 确保文档/句子对齐
2. 保留 b, e (字符级别位置)
3. 保留 type_name (实体类型)
4. 保留 token_indices (用于调试)
5. 不折叠角色特定标签 (如 startDate/endDate)
6. 不按文本去重 (相同文本可以是不同角色)
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# 配置
W2NER_OUTPUT_PATH = project_root / "W2NER" / "output.json"
PROCNET_SIDECAR_OUTPUT = project_root / "procnet" / "sidecar_entities"

# 标签映射（W2NER 输出类型 → ProcNet 标准类型）
# W2NER 输出是小写，ProcNet 使用驼峰命名
LABEL_MAP = {
    "orderapp": "orderApp",
    "ordernumber": "orderNumber",
    "seatclass": "seatClass",
    "seatnumber": "seatNumber",
    "seattype": "seatType",
    "departurecity": "departureCity",
    "departurestation": "departureStation",
    "arrivalcity": "arrivalCity",
    "arrivalstation": "arrivalStation",
    "startdate": "startDate",
    "enddate": "endDate",
    "starttime": "startTime",
    "endtime": "endTime",
    "vehiclenumber": "vehicleNumber",
    "idnumber": "idNumber",
    "cardnumber": "cardNumber",
    "dateofbirth": "dateOfBirth",
    "ethnicgroup": "ethnicGroup",
    "validfrom": "validFrom",
    "validto": "validTo",
    "roomtype": "roomType",
    "ticketgate": "ticketGate",
    "cardaddress": "cardAddress",
    # 保持不变的
    "person": "person",
    "name": "name",
    "gender": "gender",
    "date": "date",
    "time": "time",
    "status": "status",
    "price": "price",
    "address": "address",
    "city": "city",
}


def load_w2ner_output(path: Path) -> List[Dict[str, Any]]:
    """加载 W2NER 输出"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_char_positions(sentence: List[str]) -> List[int]:
    """
    计算每个 token 在句子中的起始字符位置
    
    Args:
        sentence: token 列表（字符级别）
    
    Returns:
        每个 token 的起始字符位置
    """
    positions = []
    current_pos = 0
    for token in sentence:
        positions.append(current_pos)
        current_pos += len(token)
    return positions


def convert_entity_to_sidecar(
    sample: Dict[str, Any],
    entity_index: int,
    entity: Dict[str, Any]
) -> Dict[str, Any]:
    """
    将单个 W2NER entity 转换为 ProcNet sidecar 格式
    
    Args:
        sample: W2NER 样本
        entity_index: entity 在样本中的索引
        entity: entity 数据
    
    Returns:
        ProcNet sidecar 格式的 entity 记录
    """
    doc_id = sample.get("doc_id", "")
    sent_id = sample.get("sent_id", 0)
    sentence = sample.get("sentence", [])
    
    # 计算字符位置
    char_positions = compute_char_positions(sentence)
    
    # Entity 文本（token 列表 → 字符串）
    entity_tokens = entity.get("text", [])
    entity_text = "".join(entity_tokens)
    
    # 计算 entity 的字符级 b/e 位置
    # 找到 entity 第一个 token 在 sentence 中的索引
    start_token_idx = -1
    for i, sent_token in enumerate(sentence):
        if i + len(entity_tokens) <= len(sentence):
            match = True
            for j, ent_token in enumerate(entity_tokens):
                if sentence[i + j] != ent_token:
                    match = False
                    break
            if match:
                start_token_idx = i
                break
    
    if start_token_idx == -1:
        # 找不到匹配，使用 0
        start_token_idx = 0
    
    end_token_idx = start_token_idx + len(entity_tokens) - 1
    
    # 字符级别位置
    b = char_positions[start_token_idx] if start_token_idx < len(char_positions) else 0
    e = char_positions[end_token_idx] + len(sentence[end_token_idx]) if end_token_idx < len(sentence) else b + len(entity_text)
    
    # 类型映射
    orig_type = entity.get("type", "unknown")
    normalized_type = LABEL_MAP.get(orig_type, orig_type)
    
    # 构建 sidecar 记录
    sidecar_entity = {
        "doc_id": doc_id,
        "sent_id": sent_id,
        "b": b,
        "e": e,
        "text": entity_text,
        "type_name": normalized_type,
        "type": normalized_type,  # 兼容性字段
        "token_indices": list(range(start_token_idx, end_token_idx + 1)),
        "head": start_token_idx,  # head token 索引
        "source": "w2ner",
        "w2ner_type": orig_type,  # 保留原始 W2NER 类型
    }
    
    # 如果有置信度，添加
    if "score" in entity:
        sidecar_entity["score"] = entity["score"]
    
    # 构建稳定的 key（用于 entity 链接/聚类）
    # 格式：doc_id#sent_id_b_e#text#type（确保全局唯一）
    stable_key = f"{doc_id}#{sent_id}_{b}_{e}#{entity_text}#{normalized_type}"
    sidecar_entity["key"] = stable_key
    sidecar_entity["cluster_key"] = stable_key
    
    return sidecar_entity


def convert_w2ner_to_procnet_sidecar(
    w2ner_data: List[Dict[str, Any]],
    output_path: Path
) -> int:
    """
    将 W2NER 输出转换为 ProcNet sidecar JSONL 格式
    
    Args:
        w2ner_data: W2NER 输出数据
        output_path: 输出文件路径
    
    Returns:
        转换的 entity 数量
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    entity_count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in w2ner_data:
            doc_id = sample.get("doc_id", "")
            sent_id = sample.get("sent_id", 0)
            
            entities = sample.get("entity", [])
            
            for entity_index, entity in enumerate(entities):
                sidecar_entity = convert_entity_to_sidecar(sample, entity_index, entity)
                
                # 写入 JSONL（一行一个 JSON 对象）
                f.write(json.dumps(sidecar_entity, ensure_ascii=False) + "\n")
                entity_count += 1
    
    return entity_count


def main():
    print("=" * 70)
    print("W2NER → ProcNet Sidecar 转换")
    print("=" * 70)
    
    # 检查输入文件
    if not W2NER_OUTPUT_PATH.exists():
        print(f"❌ 输入文件不存在：{W2NER_OUTPUT_PATH}")
        return
    
    # 加载数据
    print(f"\n加载 W2NER 输出：{W2NER_OUTPUT_PATH}")
    w2ner_data = load_w2ner_output(W2NER_OUTPUT_PATH)
    print(f"  样本数：{len(w2ner_data)}")
    
    # 统计 entity 数量
    total_entities = sum(len(s.get("entity", [])) for s in w2ner_data)
    print(f"  Entity 总数：{total_entities}")
    
    # 转换
    output_path = PROCNET_SIDECAR_OUTPUT / "test_typed_entities.jsonl"
    print(f"\n转换并保存至：{output_path}")
    
    entity_count = convert_w2ner_to_procnet_sidecar(w2ner_data, output_path)
    
    # 输出统计
    print(f"\n转换完成!")
    print(f"  输出 entity 数：{entity_count}")
    
    # 验证输出
    print(f"\n验证输出文件:")
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"  行数：{len(lines)}")
    
    # 显示前 5 条记录
    print(f"\n前 5 条 sidecar 记录:")
    for i, line in enumerate(lines[:5]):
        record = json.loads(line)
        print(f"\n  [{i+1}]")
        for key, value in record.items():
            print(f"      {key}: {value}")
    
    # 类型统计
    print(f"\nSidecar Entity 类型统计:")
    type_counts = {}
    for line in lines:
        record = json.loads(line)
        t = record.get("type_name", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {t}: {c}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
