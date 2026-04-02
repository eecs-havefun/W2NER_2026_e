# W2NER ↔ ProcNet 耦合补充信息清单

> 用途：用于排查 **W2NER 输出 → ProcNet 输入** 的数据耦合问题。  
> 以下 6 项已根据实际数据链路填充。

---

## 仓库信息

### 你的当前仓库
- ProcNet（你的修改版）：`https://github.com/eecs-havefun/procnet_2026_e`
- W2NER（你的修改版）：`https://github.com/eecs-havefun/W2NER_2026_e`

### 上游官方仓库
- ProcNet（官方）：`https://github.com/xnyuwg/procnet`
- W2NER（官方）：`https://github.com/ljynlp/W2NER`

### 说明
- `eecs-havefun/procnet_2026_e` 基于官方 `xnyuwg/procnet` 修改  
- `eecs-havefun/W2NER_2026_e` 基于官方 `ljynlp/W2NER` 修改

---

## 已补充的 6 项

---

### 1. 一条原始 W2NER 输出

来源：`W2NER/mixed_data_full_output.json`（test split 预测结果）

```json
{
  "doc_id": "doc_000007",
  "sent_id": 0,
  "sentence": [
    "【", "春", "秋", "航", "空", "】", "全", "美", "花", "女", "士", "，",
    "您", "8", "月", "2", "5", "日", "首", "都", "航", "空", "机", "场",
    "至", "机", "场", "经", "济", "舱", "4", "2", "B", "已", "值", "机",
    "成", "功", "（", "票", "价", "1", "9", "2", "1", ".", "1", "3", "元", "）", "。"
  ],
  "entity": [
    {"text": ["8", "月", "2", "5", "日"], "type": "date"},
    {"text": ["春", "秋", "航", "空"], "type": "orderapp"},
    {"text": ["1", "9", "2", "1", ".", "1", "3"], "type": "price"},
    {"text": ["经", "济", "舱"], "type": "seatclass"},
    {"text": ["首", "都", "航", "空", "机", "场"], "type": "departurestation"},
    {"text": ["全", "美", "花"], "type": "person"},
    {"text": ["4", "2", "B"], "type": "seatnumber"}
  ],
  "procnet_entities": [
    {
      "key": [1, 5, 3],
      "cluster_key": [1, 5, 3],
      "token_indices": [1, 2, 3, 4],
      "b": 1,
      "e": 5,
      "type_id": 3,
      "type": "orderapp",
      "score": 0.941175,
      "head": 1,
      "text": "春秋航空"
    },
    {
      "key": [6, 9, 2],
      "cluster_key": [6, 9, 2],
      "token_indices": [6, 7, 8],
      "b": 6,
      "e": 9,
      "type_id": 2,
      "type": "person",
      "score": 0.990086,
      "head": 6,
      "text": "全美花"
    },
    {
      "key": [13, 18, 9],
      "cluster_key": [13, 18, 9],
      "token_indices": [13, 14, 15, 16, 17],
      "b": 13,
      "e": 18,
      "type_id": 9,
      "type": "date",
      "score": 0.995283,
      "head": 13,
      "text": "8月25日"
    },
    {
      "key": [18, 24, 6],
      "cluster_key": [18, 24, 6],
      "token_indices": [18, 19, 20, 21, 22, 23],
      "b": 18,
      "e": 24,
      "type_id": 6,
      "type": "departurestation",
      "score": 0.922555,
      "head": 18,
      "text": "首都航空机场"
    },
    {
      "key": [27, 30, 15],
      "cluster_key": [27, 30, 15],
      "token_indices": [27, 28, 29],
      "b": 27,
      "e": 30,
      "type_id": 15,
      "type": "seatclass",
      "score": 0.991958,
      "head": 27,
      "text": "经济舱"
    },
    {
      "key": [30, 33, 16],
      "cluster_key": [30, 33, 16],
      "token_indices": [30, 31, 32],
      "b": 30,
      "e": 33,
      "type_id": 16,
      "type": "seatnumber",
      "score": 0.991385,
      "head": 30,
      "text": "42B"
    },
    {
      "key": [41, 48, 13],
      "cluster_key": [41, 48, 13],
      "token_indices": [41, 42, 43, 44, 45, 46, 47],
      "b": 41,
      "e": 48,
      "type_id": 13,
      "type": "price",
      "score": 0.997499,
      "head": 41,
      "text": "1921.13"
    }
  ]
}
```

---

### 2. 对应的 ProcNet 格式样本（同一数据链路）

来源：`procnet/procnet_format/mixed_data_with_queries/test.json`（与 W2NER 输出同一文档 `doc_000007`）

```json
[
  "doc_000007",
  {
    "sentences": [
      "【春秋航空】全美花女士，您8月25日首都航空机场至机场经济舱42B已值机成功（票价1921.13元）。",
      "下载APP查看电子登机牌 x.axcpx.cxx/xxhgxx 系统发送请勿回复"
    ],
    "ann_valid_mspans": ["全美花", "春秋航空", "经济舱", "42B", "8月25日", "首都航空", "1921.13"],
    "ann_valid_dranges": [[0,6,9],[0,1,5],[0,27,30],[0,30,33],[0,13,18],[0,18,22],[0,41,48]],
    "ann_mspan2dranges": {
      "全美花#0_6_9#person": [[0, 6, 9]],
      "春秋航空#0_1_5#orderApp": [[0, 1, 5]],
      "经济舱#0_27_30#seatClass": [[0, 27, 30]],
      "42B#0_30_33#seatNumber": [[0, 30, 33]],
      "8月25日#0_13_18#startDate": [[0, 13, 18]],
      "首都航空#0_18_22#name": [[0, 18, 22]],
      "1921.13#0_41_48#price": [[0, 41, 48]]
    },
    "ann_mspan2guess_field": {
      "全美花#0_6_9#person": "person",
      "春秋航空#0_1_5#orderApp": "orderApp",
      "经济舱#0_27_30#seatClass": "seatClass",
      "42B#0_30_33#seatNumber": "seatNumber",
      "8月25日#0_13_18#startDate": "startDate",
      "首都航空#0_18_22#name": "name",
      "1921.13#0_41_48#price": "price"
    },
    "recguid_eventname_eventdict_list": [
      [0, "flight", {
        "person": "全美花",
        "orderApp": "春秋航空",
        "seatClass": "经济舱",
        "seatNumber": "42B",
        "startDate": "8月25日",
        "name": "首都航空",
        "price": "1921.13"
      }]
    ]
  }
]
```

**数据链路验证**：
- W2NER 输出（`mixed_data_full_output.json`）：2165 条样本，720 个唯一 doc_id
- ProcNet 格式（`procnet_format/mixed_data_with_queries/test.json`）：720 个文档
- doc_id 集合完全一致，来自同一源数据 `data_v1b/mixed_data_with_queries`

---

### 3. `procnet_entities` 里的 `b/e` 索引说明

```text
b/e 的索引类型：字符索引（char-level index），同时等于 token 索引
b/e 针对的目标序列："".join(sentence)，即句子字符串
```

**原因**：W2NER 的 `sentence` 是**字级别列表**（每个元素一个汉字），所以 token 索引和字符索引等价。

**验证**（doc_000007）：

| 实体 | b | e | sentence[b:e] | sentence_str[b:e] | 匹配 |
|------|---|---|---------------|-------------------|------|
| 春秋航空 | 1 | 5 | "春秋航空" | "春秋航空" | ✓ |
| 全美花 | 6 | 9 | "全美花" | "全美花" | ✓ |
| 8月25日 | 13 | 18 | "8月25日" | "8月25日" | ✓ |
| 首都航空机场 | 18 | 24 | "首都航空机场" | "首都航空机场" | ✓ |
| 经济舱 | 27 | 30 | "经济舱" | "经济舱" | ✓ |
| 42B | 30 | 33 | "42B" | "42B" | ✓ |
| 1921.13 | 41 | 48 | "1921.13" | "1921.13" | ✓ |

---

### 4. `end` 边界约定

```text
切片规则是否为 [start, end)：是（左闭右开）
```

**示例**（doc_000007 中实体"春秋航空"）：
```text
sentence_str = "【春秋航空】全美花女士..."
start = 1
end = 5
sentence_str[1:5] = "春秋航空"
```

**注意**：W2NER 的 `e` 字段等于 `token_indices[-1] + 1`（见 `utils.py:384`），即已经是左闭右开的 end 值。

---

### 5. `sentence` 粒度说明

```text
sentence 的粒度：字列表（char-level list）
示例 sentence：["【", "春", "秋", "航", "空", "】", "全", "美", "花", ...]
是否经过 tokenizer：否（原始字列表，来自 data_v1b 源数据）
BERT tokenizer 在 process_bert() 中单独应用，不影响 sentence 字段
```

**来源**：`data_v1b/mixed_data_with_queries/train.json`（Rasa NLU 格式）→ `regenerate_all_data_step2_procnet_to_w2ner.py` 按字符拆分 → `sentence` 字段为字列表。

---

### 6. ProcNet 读取配置

来源：`official_procnet/procnet/run.py`

```python
def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run_save_name", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--epoch", type=int, default=50)
    arg_parser.add_argument("--read_pseudo", type=str, default=False)
    args = arg_parser.parse_args(args=in_args)
    args.read_pseudo = UtilString.str_to_bool(args.read_pseudo)
    return args

def get_config(args) -> DocEEConfig:
    config = DocEEConfig()
    config.model_save_name = args.run_save_name
    config.node_size = 512
    config.proxy_slot_num = 16
    config.gradient_accumulation_steps = args.batch_size
    config.max_epochs = args.epoch
    config.data_loader_shuffle = True
    config.model_name = "hfl/chinese-roberta-wwm-ext"
    config.device = torch.device('cuda')
    return config
```

```text
当前是否启用了 pseudo dataset：否（默认 read_pseudo=False）
相关入口文件：official_procnet/procnet/run.py
相关配置文件：official_procnet/procnet/procnet/conf/DocEE_conf.py
数据路径：由 GlobalConfigManager.get_dataset_path() 决定
```

---

## 关键差异总结

| 维度 | W2NER 输出 | ProcNet 输入 |
|------|-----------|-------------|
| 粒度 | 句子级（每样本一句） | 文档级（每文档多句） |
| 实体 key | `procnet_entities` 列表 | `ann_mspan2dranges` 字典，key 格式 `"文本#sentIdx_start_end#类型"` |
| 类型名 | `type` 字段（小写，如 `orderapp`） | `ann_mspan2guess_field` 值（驼峰，如 `orderApp`） |
| 事件 | ❌ 不产出 | `recguid_eventname_eventdict_list`（必需） |
| 索引 | `b/e` 字索引（左闭右开） | `[sentIdx, start, end]` 字符索引（左闭右开） |

**索引口径一致**：W2NER 的 `b/e` 和 ProcNet 的 `start/end` 都是针对句子字符串的字符索引，且都是左闭右开 `[start, end)`。转换时只需按 `doc_id` 聚合句子、拼接实体 key 格式即可。

---

## 交付目标

基于以上信息，后续可继续做：

1. 检查 **W2NER 索引口径** 和 **ProcNet span 切片口径** 是否一致 → **已确认一致**
2. 检查 **实体类型 / field 语义** 是否错位 → **需注意大小写差异**（W2NER 小写 vs ProcNet 驼峰）
3. 直接给出一版更稳妥的 **W2NER → ProcNet 转换方案** 或转换脚本修订建议
