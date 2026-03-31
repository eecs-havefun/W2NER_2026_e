#!/usr/bin/env python3
import json
from pathlib import Path

path = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev/flight_orders_with_queries/train.json")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("第一个样本的所有字段:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))

print("\n\n第二个样本的所有字段:")
print(json.dumps(data[1], indent=2, ensure_ascii=False))
