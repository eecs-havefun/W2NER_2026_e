# W2NER × ProcNet — 统一命名实体识别与事件抽取耦合框架

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

本项目是 [W2NER](https://github.com/ljynlp/W2NER)（AAAI 2022）的扩展分支，将其与 [ProcNet](https://github.com/xnyuwg/procnet) 耦合，实现从**命名实体识别**到**文档级多事件抽取**的统一流水线。

> W2NER 将统一 NER 建模为词-词关系分类，通过 Next-Neighboring-Word (NNW) 和 Tail-Head-Word (THW) 关系同时处理扁平、嵌套和不连续实体。
>
> 本扩展在 W2NER 的预测输出中增加 `procnet_entities` 字段，使其可直接作为 ProcNet 的 sidecar 输入，实现 W2NER → ProcNet 级联事件抽取。

---

## 目录

- [架构概览](#架构概览)
- [数据链路](#数据链路)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [训练与预测](#训练与预测)
- [ProcNet 耦合](#procnet-耦合)
- [项目结构](#项目结构)
- [已知问题](#已知问题)
- [引用](#引用)

---

## 架构概览

### 标签方案

<p align="center">
  <img src="./figures/scheme.PNG" width="400"/>
</p>

### 模型架构

<p align="center">
  <img src="./figures/architecture.PNG" width="600"/>
</p>

**W2NER 架构**：BERT → BiLSTM → Conditional LayerNorm → 多粒度 2D 卷积 → Biaffine 分类器

**耦合架构**：W2NER 实体预测 → sidecar JSONL → ProcNet DocEE 事件抽取

---

## 数据链路

```
data_v1b/ (RASA NLU 原始格式, 4,800 篇文档)
  │
  ▼ [脚本 1] convert_data_v1b_to_procnet.py  [ProcNet 仓库 scripts/]
  │   句子分割 → 实体位置映射 → 复合 key → 事件构建 → 70/15/15 划分
  │
  ▼ procnet_format/ (ProcNet 格式, 4,800 篇)
  │   train: 3,360 | dev: 720 | test: 720
  │
  ├──────────────────────────────────────────────┐
  │                                              │
  ▼ [导出 JSONL]                                 ▼ [脚本 2] convert_procnet_to_w2ner.py
sidecar_entities_gold/                    data/{dataset}/{split}.json
(Gold sidecar, 上限实验)                         │
                                                 ▼ data_loader.load_data_bert()
                                                 │  BERT tokenizer → piece_map → grid_labels
                                                 ▼
                                           Model.forward()
                                           BERT → BiLSTM → 2D conv → Biaffine
                                                 │
                                                 ▼ Trainer.predict() + decode_for_procnet()
                                                 │  输出: {doc_id, sent_id, sentence,
                                                 │        entity[], procnet_entities[]}
                                                 ▼
                                           predictions/{train,dev,test}.json
                                                 │
                                                 ▼ [脚本 3] export_doc_typed_entities.py [ProcNet 仓库]
                                                 │  按 doc_id 聚合 → 验证 span → 去重
                                                 ▼
                                           sidecar_entities/*.jsonl
                                           (W2NER 预测 sidecar)
                                                 │
                                                 ▼
                                           ProcNet 训练 (级联实验)
```

### 数据转换脚本

| # | 脚本 | 位置 | 功能 |
|---|------|------|------|
| 1 | `convert_data_v1b_to_procnet.py` | ProcNet 仓库 `scripts/` | RASA NLU → ProcNet 格式 |
| 2 | `convert_procnet_to_w2ner.py` | ProcNet 仓库 `scripts/` | ProcNet → W2NER 格式 |
| 3 | `export_doc_typed_entities.py` | ProcNet 仓库 `scripts/` | W2NER 预测 → sidecar JSONL |

#### 脚本 1：`convert_data_v1b_to_procnet.py`

将 RASA NLU 格式的源数据转换为 ProcNet 格式。

```bash
python scripts/convert_data_v1b_to_procnet.py \
  --input_dir ../data_v1b/mixed_data_with_queries \
  --output_dir ./procnet_format/mixed_data_with_queries \
  --split all
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | RASA NLU 源数据目录 | — |
| `--output_dir` | ProcNet 格式输出目录 | — |
| `--split` | 处理的分割（`all`/`train`/`test`） | `all` |
| `--event_mapping` | intent → 事件类型的 JSON 映射文件 | `None` |
| `--max_docs` | 最大转换文档数（测试用） | 全部 |

**输出**：`{output_dir}/{train,dev,test}.json`，每个文件为 `[[doc_id, {sentences, ann_mspan2dranges, ann_mspan2guess_field, recguid_eventname_eventdict_list}]]` 格式。

#### 脚本 2：`convert_procnet_to_w2ner.py`

将 ProcNet 格式转换为 W2NER 训练所需的句子级格式。

```bash
# 目录模式（推荐）
python scripts/convert_procnet_to_w2ner.py \
  --input_dir ./procnet_format/mixed_data_with_queries \
  --output_dir ./data/mixed_data_with_queries

# 单文件模式
python scripts/convert_procnet_to_w2ner.py \
  --input ./procnet_format/mixed_data_with_queries/train.json \
  --output ./data/mixed_data_with_queries/train.json
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` / `--input` | ProcNet 输入目录或文件 | — |
| `--output_dir` / `--output` | W2NER 输出目录或文件 | — |
| `--fold_role_types` | 折叠 startDate/endDate→date, startTime/endTime→time | `False` |
| `--no_strict_alignment` | 关闭 span 文本严格对齐检查 | `False` |
| `--keep_duplicates` | 保留同 span 多类型实体 | `True` |
| `--slim_entities` | 不保留原始类型信息 | `False` |
| `--write_manifest` | 写入转换清单文件 | `False` |

**输出**：`{output_dir}/{train,dev,test}.json`，每个文件为 `[{sample_id, doc_id, sent_id, text, sentence, ner, entities}]` 格式。

#### 脚本 3：`export_doc_typed_entities.py`

将 W2NER 句子级预测聚合为文档级 sidecar JSONL，供 ProcNet 使用。

```bash
python scripts/export_doc_typed_entities.py \
  --source_json ./data/mixed_data_with_queries/test.json \
  --pred_json ./predictions/test.json \
  --output_jsonl ./sidecar_entities/test_doc_typed_entities.jsonl \
  --report_json ./sidecar_entities/test_export_report.json
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source_json` | 源数据 JSON（用于对齐句子） | — |
| `--pred_json` | W2NER 预测输出 JSON | — |
| `--output_jsonl` | 输出的文档级 sidecar JSONL | — |
| `--report_json` | 导出报告（含验证统计） | — |
| `--no_strict_sentence_match` | 关闭句子文本严格匹配 | `False` |
| `--no_strict_range_check` | 关闭 span 范围严格检查 | `False` |

**输出**：每行一个文档的 JSONL 文件，格式为 `{"doc_id": "...", "typed_entities": [...]}`。

### 关键设计

| 设计 | 说明 |
|------|------|
| **复合 key** | `文本#sentIdx_start_end#类型`，天然区分同文本多位置/多类型 |
| **索引口径** | 字级、左闭右开 `[start, end)`，W2NER 与 ProcNet 一致 |
| **类型映射** | 28 个 W2NER type → 30 个 ProcNet field（驼峰归一化） |
| **文档聚合** | 按 `doc_id` 分组、按 `sent_id` 排序，防止数据泄露 |

---

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.4（GPU 训练）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 关键依赖版本

| 包 | 版本 |
|----|------|
| torch | 1.13.1 |
| transformers | 4.37.2 |
| numpy | 1.23.5 |
| scikit-learn | 1.3.2 |
| gensim | 4.1.2 |
| pandas | 1.3.4 |
| prettytable | 2.4.0 |

### BERT 模型

需要预训练的中文 BERT 模型，默认路径为 `../models/bert_base_chinese`。可在配置文件中的 `bert_name` 字段修改。

---

## 快速开始

### 1. 冒烟测试（1 epoch）

```bash
python main.py --config config/id_cards_with_queries.json --device 0
```

### 2. 全量训练（10 epochs）

```bash
python main.py --config config/mixed_data_with_queries.json --device 0
```

### 3. 查看预测结果

训练完成后，预测结果自动保存到 `predictions/{train,dev,test}.json`。

---

## 训练与预测

### 训练命令

```bash
python main.py --config config/<dataset>.json --device <GPU_ID>
```

### 可用配置

| 配置文件 | 数据集 | 领域 |
|----------|--------|------|
| `flight_orders_with_queries.json` | flight_orders_with_queries | 航班订单 |
| `hotel_orders_with_queries.json` | hotel_orders_with_queries | 酒店订单 |
| `id_cards_with_queries.json` | id_cards_with_queries | 身份证 |
| `train_orders_with_queries.json` | train_orders_with_queries | 火车票 |
| `mixed_data_with_queries.json` | mixed_data_with_queries | 混合（全部领域） |
| `smoke_test.json` | smoke_test | 冒烟测试（小样本） |

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `./config/conll03.json` |
| `--device` | GPU 设备 ID | `0` |
| `--save_path` | 模型保存路径 | 配置文件指定 |
| `--predict_path` | 预测输出路径 | 配置文件指定 |
| `--continuous_only` | 仅导出连续实体（ProcNet 耦合） | `1` |
| `--epochs` | 训练轮数 | 配置文件指定 |
| `--batch_size` | 批次大小 | 配置文件指定 |
| `--seed` | 随机种子 | 配置文件指定 |

### 训练流程

1. **训练阶段**：在每个 epoch 中，模型在 train set 上训练，在 dev/test set 上评估
2. **模型选择**：保存 dev set 上 Entity F1 最高的模型
3. **预测阶段**：加载最佳模型，对 train/dev/test 三个 split 分别生成预测
4. **输出格式**：每条预测包含 `doc_id`, `sent_id`, `sentence`, `entity`（W2NER 格式）, `procnet_entities`（ProcNet 格式）

---

## ProcNet 耦合

### 耦合范式

本项目采用 **Sidecar 范式（范式 B）**：保留 ProcNet 原始事件样本不变，将 W2NER 预测的实体作为 sidecar 输入注入 ProcNet 训练流程。

```
┌─────────────────────────────────────────────────────┐
│  范式 A（不采用）：把 W2NER 输出完全改写为 ProcNet 样本  │
│  范式 B（采用）：保留 ProcNet 原始样本，W2NER 实体作 sidecar │
└─────────────────────────────────────────────────────┘
```

这种设计的优势：
- ProcNet 的事件监督信号保持完整
- W2NER 只负责提供高质量的 mention/entity 候选层
- 事件类型/角色组装由 ProcNet 自行完成

### 耦合机制

W2NER 输出句子级实体预测，ProcNet 需要文档级实体+事件结构。耦合通过以下机制实现：

1. **`utils.decode_for_procnet()`**：从 N×N 网格预测中提取带分数的实体结构
2. **`utils.build_prediction_record()`**：构建同时包含 W2NER 和 ProcNet 格式的输出记录
3. **`export_doc_typed_entities.py`**（ProcNet 仓库）：按文档聚合句子级预测，生成 sidecar JSONL

### W2NER 输出格式

```json
{
  "doc_id": "doc_000007",
  "sent_id": 0,
  "sentence": ["【", "春", "秋", "航", "空", "】", "..."],
  "entity": [{"text": ["春", "秋", "航", "空"], "type": "orderApp"}],
  "procnet_entities": [
    {
      "b": 1,
      "e": 5,
      "type": "orderApp",
      "text": "春秋航空",
      "score": 0.94
    }
  ]
}
```

`procnet_entities` 中每个实体的字段：

| 字段 | 说明 |
|------|------|
| `b` | 起始字符索引（字级、左闭） |
| `e` | 结束字符索引（字级、右开） |
| `type` | 实体类型名（驼峰格式，如 `orderApp`） |
| `text` | 实体文本 |
| `score` | 预测置信度 |

### 两层耦合策略

W2NER 的预测结果在 ProcNet 中扮演 **mention/entity 候选层**，而非直接替代完整事件层：

```
┌──────────────────────────────────────────────┐
│  Mention 层（W2NER 直接提供）                   │
│  • 实体边界 (b/e)                              │
│  • 实体文本                                    │
│  • 实体类型候选                                 │
│  • 预测分数                                    │
├──────────────────────────────────────────────┤
│  Event 层（ProcNet / 规则组装）                  │
│  • 事件类型判定                                 │
│  • 角色分配                                    │
│  • 事件数量预测                                 │
│  • 多事件关系                                   │
└──────────────────────────────────────────────┘
```

**为什么不让 W2NER 直接替代事件层？**
- W2NER 是实体识别模型，不建模事件间的全局依赖关系
- 同一文本可能对应多个角色（如 "泰安" 同时是 departureCity 和 departureStation）
- ProcNet 的代理节点机制和 Hausdorff 距离最小化专门为此设计

### 类型映射

W2NER 的实体类型通过驼峰归一化映射到 ProcNet 的事件角色字段。当前模型已直接输出细粒度类型（`startdate`/`enddate`/`starttime`/`endtime`），不再需要泛化 `date`/`time`。

#### 稳定映射（直接驼峰归一化）

| W2NER type | ProcNet field | W2NER type | ProcNet field |
|------------|---------------|------------|---------------|
| `orderapp` | `orderApp` | `seatclass` | `seatClass` |
| `seatnumber` | `seatNumber` | `seattype` | `seatType` |
| `departurestation` | `departureStation` | `arrivalstation` | `arrivalStation` |
| `departurecity` | `departureCity` | `arrivalcity` | `arrivalCity` |
| `vehiclenumber` | `vehicleNumber` | `dateofbirth` | `dateOfBirth` |
| `cardnumber` | `cardNumber` | `cardaddress` | `cardAddress` |
| `ordernumber` | `orderNumber` | `ethnicgroup` | `ethnicGroup` |
| `validfrom` | `validFrom` | `validto` | `validTo` |
| `idnumber` | `idNumber` | `roomtype` | `roomType` |
| `ticketgate` | `ticketGate` | `person` | `person` |
| `name` | `name` | `price` | `price` |
| `status` | `status` | `address` | `address` |
| `city` | `city` | `gender` | `gender` |

#### 时间类型（已解决语义分裂）

| W2NER type | ProcNet field | 说明 |
|------------|---------------|------|
| `startdate` | `startDate` | ✅ 直接映射 |
| `enddate` | `endDate` | ✅ 直接映射 |
| `starttime` | `startTime` | ✅ 直接映射 |
| `endtime` | `endTime` | ✅ 直接映射 |

> **历史说明**：早期版本中 W2NER 将所有日期合并为泛化 `date`、时间合并为 `time`，导致无法区分 start/end。已通过不折叠策略修复，当前模型直接输出 4 个细粒度类型。

### 复合 Key 设计

ProcNet 使用复合 key 区分同文本多位置/多类型的实体：

```
文本#sentIdx_start_end#类型
```

例如：`泰安#0_21_23#departureCity` 和 `泰安#0_21_23#departureStation` 是两个独立的 entry。

这个设计由修改版 ProcNet 的 `DocEE_processor.py` 支持，通过正则 `^(.*)#(\d+)_(\d+)_(\d+)#([^#]+)$` 解析。

### 级联实验

```
实验 A（上限）: Gold sidecar → ProcNet
实验 B（级联）: W2NER 预测 → W2NER sidecar → ProcNet
```

两者对比量化 W2NER 预测误差对 ProcNet 事件抽取的影响。

---

## 项目结构

```
W2NER/
├── main.py                 # 入口，Trainer 类
├── model.py                # BERT + BiLSTM + 2D conv + biaffine
├── config.py               # Config 类，类型验证
├── data_loader.py          # Vocabulary, RelationDataset, process_bert(), collate_fn()
├── utils.py                # 日志, 序列化, decode(), decode_for_procnet(), build_prediction_record()
├── config/                 # 各数据集训练配置
│   ├── mixed_data_with_queries.json
│   ├── flight_orders_with_queries.json
│   └── ...
├── data/                   # 数据集
│   ├── data_w2ner_folded_with_dev/   # git 跟踪的数据
│   ├── mixed_data_with_queries/      # 当前活跃数据集
│   └── ...
├── predictions/            # 预测输出（按 split）
├── scripts_maybeuseful/    # 数据转换、验证、冒烟测试脚本
├── conversation_history/   # 设计决策记录
├── needs/                  # 耦合分析文档
├── figures/                # 架构图
├── requirements.txt
├── AGENTS.md               # AI 代理开发指南
└── README.md               # 本文件
```

---

## 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| `date`/`time` 类型折叠 | ✅ 已修复 | 重新转换数据，30 个细粒度类型全部恢复 |
| 随机种子未生效 | ⚠️ 待修复 | `main.py` 中随机种子代码被注释，需取消注释 |
| `run_all_smoke_tests.py` 导入顺序 | ⚠️ 已知 | `Path` 在 shebang 之前使用 |
| `Vocabulary.__len__` | ✅ 已修复 | 官方返回 `len(self.token2id)`（不存在），本 fork 修复为 `len(self.label2id)` |
| W2NER 全量预测 | ⚠️ 进行中 | 当前 `predict_final` 仅跑 test 集，需扩展到全量 4,800 篇 |

---

## 引用

如果使用了本项目的代码或模型，请引用原始论文：

```bibtex
@inproceedings{li2022unified,
  title={Unified named entity recognition as word-word relation classification},
  author={Li, Jingye and Fei, Hao and Liu, Jiang and Wu, Shengqiong and Zhang, Meishan and Teng, Chong and Ji, Donghong and Li, Fei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={10965--10973},
  year={2022}
}
```

---

## 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。
