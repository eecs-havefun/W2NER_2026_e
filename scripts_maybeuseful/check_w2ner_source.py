
# Import path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

#!/usr/bin/env python3
import json
from pathlib import Path

path = project_root / "W2NER" / "data" / "data_w2ner_folded_with_dev" / "flight_orders_with_queries" / "train.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"总样本数：{len(data)}")
print(f"\n第一个样本:")
print(json.dumps(data[0], indent=2, ensure_ascii=False)[:800])

print(f"\n\n最后一个样本:")
print(json.dumps(data[-1], indent=2, ensure_ascii=False)[:800])
