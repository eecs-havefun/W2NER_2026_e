# AGENTS.md — W2NER × ProcNet 耦合框架

> 本文档为 AI 代理提供 W2NER 仓库的完整上下文：已完成工作、当前状态、下一阶段目标、代码规范与关键命令。

---

## 1. 项目概述

W2NER (AAAI 2022) 将统一 NER 建模为词-词关系分类。本 fork 将其与 ProcNet 耦合，形成 **W2NER → ProcNet 级联事件抽取流水线**：

```
RASA NLU 原始数据 → ProcNet 中间格式 → W2NER 训练数据 → W2NER 预测 → sidecar JSONL → ProcNet 事件抽取
```

**耦合范式**：Sidecar（范式 B）— W2NER 提供 mention/entity 候选层，ProcNet 负责事件类型/角色组装和全局依赖建模。

**远程仓库**：`git@github.com:eecs-havefun/W2NER_2026_e.git`

---

## 2. 已完成工作

### 2.1 数据管线（3 步转换）

| 步骤 | 脚本（ProcNet 仓库 `scripts/`） | 输入 → 输出 |
|------|--------------------------------|-------------|
| 1 | `convert_data_v1b_to_procnet.py` | RASA NLU → ProcNet 格式 |
| 2 | `convert_procnet_to_w2ner.py` | ProcNet → W2NER 句子级格式 |
| 3 | `export_doc_typed_entities.py` | W2NER 预测 → 文档级 sidecar JSONL |

- 数据规模：5 个领域（flight/hotel/train/id_card/mixed），共 4,800 篇文档（train=3,360, dev=720, test=720）
- 复合 key 设计：`文本#sentIdx_start_end#类型`，天然区分同文本多位置/多类型
- 类型映射：28 个 W2NER type → 30 个 ProcNet field（驼峰归一化），`date`/`time` 语义分裂问题已修复

### 2.2 W2NER 扩展

- `utils.decode_for_procnet()`：从 N×N 网格预测中提取带分数的实体结构
- `utils.build_prediction_record()`：同时输出 W2NER 和 ProcNet 格式
- `main.py` 预测路径已扩展为对 train/dev/test 三个 split 全部生成预测
- `Vocabulary.__len__` 已修复（官方返回不存在的 `len(self.token2id)`）

### 2.3 仓库清理

- 移除 ~600MB 旧数据、日志、缓存、冗余脚本
- 恢复误删的 `data/mydata/`（Resume NER 数据集）
- 原始 RASA NLU 数据已放入 `data_v1b_raw_rasa/`

### 2.4 级联实验结果（`w2ner_sidecar_exp1`）

| Epoch | DEV BIO Exact F1 | DEV Event Micro F1 | TEST Event Micro F1 |
|-------|-----------------|-------------------|--------------------|
| 3     | 95.26%          | 97.23%            | 97.26%             |

各子任务 TEST F1：flight=99.10%, id_card=98.82%, train=95.88%, hotel=95.31%

---

## 3. 仓库结构

```
W2NER/
├── main.py                     # 入口，Trainer 类（含 predict_final 全量预测）
├── model.py                    # BERT + BiLSTM + 2D conv + biaffine 分类器
├── config.py                   # Config 类，含 data_root/cache_dir 支持
├── data_loader.py              # Vocabulary, RelationDataset, process_bert(), collate_fn()
├── utils.py                    # 日志, 序列化, decode(), decode_for_procnet(), build_prediction_record()
│
├── config/                     # 各数据集训练配置（JSON）
│   ├── mixed_data_with_queries.json
│   ├── flight_orders_with_queries.json
│   ├── hotel_orders_with_queries.json
│   ├── train_orders_with_queries.json
│   ├── id_cards_with_queries.json
│   └── smoke_test.json
│
├── data/                       # W2NER 训练数据
│   ├── data_w2ner_folded_with_dev/   # git 跟踪的折叠数据
│   ├── mixed_data_with_queries/      # 当前活跃数据集
│   └── ...
│
├── data_v1b_raw_rasa/          # RASA NLU 原始数据（5 领域，4,800 篇）
│   └── data_v1b/
│       ├── flight_orders_with_queries/
│       ├── hotel_orders_with_queries/
│       ├── train_orders_with_queries/
│       ├── id_cards_with_queries/
│       └── mixed_data_with_queries/
│
├── predictions/                # 预测输出（按 split）
├── scripts_maybeuseful/        # 数据转换、验证、冒烟测试脚本
├── conversation_history/       # 设计决策记录
├── needs/                      # 耦合分析文档
├── figures/                    # 架构图（scheme.PNG, architecture.PNG）
├── wheelhouse/                 # 离线依赖包
│
├── requirements.txt
├── AGENTS.md                   # 本文件
└── README.md                   # 项目文档
```

---

## 4. 关键文件说明

| 文件 | 职责 | 注意事项 |
|------|------|----------|
| `main.py` | 入口，`Trainer` 类 | `self.model`/`self.config` 是实例属性（非全局）；随机种子代码被注释 |
| `model.py` | 模型定义 | BERT → BiLSTM → Conditional LayerNorm → 2D conv → Biaffine |
| `config.py` | 配置类 | 含类型验证，`data_root` 默认 `./data` |
| `data_loader.py` | 数据加载 | `Vocabulary`, `RelationDataset`, `process_bert()`, `collate_fn()` |
| `utils.py` | 工具函数 | `decode_for_procnet()` 和 `build_prediction_record()` 是 ProcNet 耦合的关键 |

---

## 5. 代码规范

### 导入顺序
标准库 → 第三方 → 本地，空行分隔。使用绝对导入，禁止通配符。

### 命名
- 类：`CamelCase`（`LayerNorm`, `Vocabulary`）
- 函数/方法：`snake_case`（`get_logger`, `decode_for_procnet`）
- 变量/属性：`snake_case`（`bert_inputs`, `grid_mask2d`）
- 常量：`UPPER_SNAKE_CASE`（`PAD`, `RANDOM_SEED`）
- 私有成员：`_` 前缀

### 格式
- 4 空格缩进，行宽 120 字符
- 类型提示：使用 `typing` 模块，`np.int64`/`np.bool_`（不用已废弃的 `np.int`/`np.bool`）

### 错误处理
- 显式 `try-except`，不静默捕获
- 日志用 `utils.get_logger(dataset)`
- `assert` 用于内部不变量检查，加 `###<name>_checker` 注释便于 grep

---

## 6. 已知问题与根因分析

### 6.1 同 Span 多角色在 Gold Label 中丢失 ❌ 未修复

**现象**：原始数据中 `"12月19日"` 同时标注为 `startDate` 和 `endDate`（train 共 1,124 处），但评估时只保留一个。

**根因链路**：
```
data_v1b_raw_rasa (✅ 保留)
  → convert_data_v1b_to_procnet.py → event_dict (✅ 保留，不同 key 可同 value)
  → DocEE_processor.parse_json_one (✅ 保留)
  → DocEE_preparer → event_label (❌ 丢失)
```

在 `DocEE_preparer.py:426-436` 中：
```python
event_label[tuple(v_id)] = self.event_role_relation_to_index[k]
```
`event_label` 以 `tuple(token_ids)` 为 key，Python dict 的 key 唯一性导致**同 span 只能保留最后一个 role**。

**影响**：约 969/3,360 个训练文档包含同 span 多类型实体，被覆盖的角色必然被记为 FN，**人为压低 F1**。

### 6.2 模型架构无法输出同 span 多角色 ❌ 未修复

`DocEE_proxy_node_model.py` 中 `span_tensor_span_to_index` 将每个 span 映射为唯一索引，评估逻辑 `DocEE_metric.py:137-145` 也强制一个 span 只选概率最高的 role。

### 6.3 `multi_event` 指标始终为 0 ℹ️ 设计限制

Fragment-level 切分后每个样本最多含 1 个事件，`multi_event` 桶为空。该指标在当前粒度下无意义。

### 6.4 随机种子未生效 ⚠️ 待修复

`main.py` 中随机种子代码被注释，需取消注释以支持可复现实验。

### 6.5 `run_all_smoke_tests.py` 导入顺序 ⚠️ 已知

`Path` 在 shebang 之前使用。

---

## 7. 下一阶段目标

### 7.1 修复同 Span 多角色问题（高优先级）

**方案 A**：修改 `DocEE_preparer.py` 的 `event_label` 构建逻辑，改用 `(tuple(token_ids), role_index)` 作为复合 key，或将 `event_label` 改为 list of tuples 而非 dict。

**方案 B**：在 ProcNet 侧引入 EPAL 论文的 **role-indexed slot filling** 机制 — 将解码单位从 "entity → one role" 改为 "role → one argument"，天然支持同 span 多角色。

### 7.2 EPAL 机制集成（中优先级）

根据 `epal_procnet_report_and_dialogue_summary.md` 的建议，分三阶段在 ProcNet 中引入 EPAL 思想：

| 阶段 | 内容 | 目标 |
|------|------|------|
| Stage 1 | Role-indexed slot filling | 解决同事件多角色问题 |
| Stage 2 | Event-conditioned argument library + contrastive loss | 解决跨事件实体复用混淆 |
| Stage 3 | Proxy-to-event alignment | 稳定多事件文档训练 |

**原则**：不替换 W2NER，不立即替换 ProcNet 的 proxy-node 框架，先增强解码阶段。

### 7.3 启用随机种子（低优先级）

取消 `main.py` 中随机种子代码的注释，确保实验可复现。

### 7.4 全量预测覆盖（低优先级）

当前 `predict_final` 已扩展为 train/dev/test 全量预测，需验证输出完整性并与 ProcNet sidecar 导出流程对接。

---

## 8. 常用命令

### 训练
```bash
python main.py --config config/<dataset>.json --device 0
```

### 冒烟测试
```bash
python scripts_maybeuseful/smoke_test.py          # 准备小样本
python main.py --config config/smoke_test.json    # 训练小样本
```

### 数据转换（ProcNet 仓库）
```bash
# RASA → ProcNet
python scripts/convert_data_v1b_to_procnet.py --input_dir ../data_v1b_raw_rasa/data_v1b/mixed_data_with_queries --output_dir ./procnet_format/mixed_data_with_queries

# ProcNet → W2NER
python scripts/convert_procnet_to_w2ner.py --input_dir ./procnet_format/mixed_data_with_queries --output_dir ./data/mixed_data_with_queries

# W2NER 预测 → sidecar
python scripts/export_doc_typed_entities.py --source_json ./data/mixed_data_with_queries/test.json --pred_json ./predictions/test.json --output_jsonl ./sidecar_entities/test_doc_typed_entities.jsonl
```

### 代码质量
```bash
pip install black isort flake8
black . --exclude=./data --exclude=./data_v1b_raw_rasa
isort .
flake8 . --max-line-length=120
```

---

## 9. Git 规范

- 提交信息：动词开头，描述性（如 "Fix decode graph traversal in utils.py"）
- **不提交**：`*.pt`, `*.bin`, `*.pth`, `log/`, `cache/`, `output.json`, `__pycache__/`
- **不提交**：`data/`（除 `data/data_w2ner_folded_with_dev/`）
- **不提交**：`scripts_maybeuseful/W2NER/`（冗余的 smoke_test 副本）
- **提交**：`data_v1b_raw_rasa/`（原始 RASA NLU 数据，17M，5 领域）
- 提交前与 `official_W2NER/` 和 `official_procnet/` 对比确认

---

## 10. 相关仓库

| 仓库 | 地址 | 职责 |
|------|------|------|
| W2NER（本仓库） | `github.com:eecs-havefun/W2NER_2026_e` | NER 模型训练、预测、sidecar 导出 |
| ProcNet | `github.com:eecs-havefun/procnet_2026_e` | 事件抽取、sidecar 消费、EPAL 集成目标 |

数据转换脚本位于 ProcNet 仓库的 `scripts/` 目录。EPAL 集成分析详见 ProcNet 仓库的 `epal_procnet_report_and_dialogue_summary.md`。
