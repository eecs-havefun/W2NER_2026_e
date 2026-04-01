# W2NER → ProcNet 数据转换摘要

## 背景

W2NER 产出句子级实体预测，ProcNet 需要文档级实体+事件结构。本摘要记录将 W2NER 预测输出转换为 ProcNet 训练输入的关键设计。

## 数据格式差异

| 维度 | W2NER 输出 | ProcNet 输入 |
|------|-----------|-------------|
| 粒度 | 句子级（每个样本一个句子） | 文档级（每个文档含多个句子） |
| 实体结构 | `[{text: [...], type: "xxx"}]` | `ann_mspan2dranges: {实体文本: [[sent_idx, start, end], ...]}` |
| 类型映射 | `type` 字段（字符串） | `ann_mspan2guess_field: {实体文本: 类型名}` |
| 事件 | ❌ 不产出 | `recguid_eventname_eventdict_list`（可留空 `[]`） |
| 标识符 | `doc_id`, `sent_id` | `doc_id`（如 `"doc_000000"`） |

## W2NER 输出结构

```json
[
  {
    "doc_id": "doc_000000",
    "sent_id": 0,
    "sentence": ["我", "叫", "张", "三"],
    "entity": [{"text": ["张", "三"], "type": "person"}],
    "procnet_entities": [
      {
        "key": [0, 2, 1],
        "cluster_key": [0, 2, 1],
        "token_indices": [0, 1],
        "b": 0,
        "e": 2,
        "type_id": 1,
        "score": 0.95,
        "head": 0
      }
    ]
  }
]
```

## 目标 ProcNet 格式

```json
[
  [
    "doc_000000",
    {
      "sentences": ["我叫张三", "第二句话..."],
      "ann_mspan2dranges": {
        "张三": [[0, 2, 4]],
        "北京": [[1, 0, 2]]
      },
      "ann_mspan2guess_field": {
        "张三": "person",
        "北京": "location"
      },
      "recguid_eventname_eventdict_list": []
    }
  ]
]
```

## 转换步骤

1. **按 doc_id 分组**：将 W2NER 输出按 `doc_id` 聚合，同一文档的句子按 `sent_id` 排序
2. **拼接句子**：`"".join(sentence_tokens)` 得到完整句子字符串
3. **构建实体映射**：
   - 从 `procnet_entities` 提取 `b`（起始字符索引）、`e`（结束字符索引）、`type_id`
   - 用 `sentence[b:e]` 得到实体文本
   - 构建 `ann_mspan2dranges[实体文本] = [[sent_id, b, e], ...]`
   - 构建 `ann_mspan2guess_field[实体文本] = type_name`
4. **事件留空**：`recguid_eventname_eventdict_list = []`（ProcNet 可正常训练，事件 loss 为 0）
5. **划分 train/dev/test**：按文档划分（70/15/15），防止数据泄露

## 关键注意事项

- **实体文本去重**：同一文档中相同文本的实体可能出现在多个位置，`ann_mspan2dranges` 的值是列表
- **类型名映射**：W2NER 的 `type_id` 需要通过 vocab 映射回类型名（如 `1 → "person"`）
- **连续实体过滤**：W2NER 输出中的 `continuous_only` 标志控制是否过滤不连续实体
- **文档划分**：必须按 `doc_id` 划分，不能按句子划分，否则同一文档的句子会同时出现在 train 和 dev 中

## 相关文件

- W2NER 预测输出：`W2NER/mixed_data_full_output.json`
- 转换脚本位置：待创建（建议放在 `W2NER/scripts_maybeuseful/`）
- ProcNet 数据格式参考：`official_procnet/procnet/procnet/data_processor/DocEE_processor.py`
