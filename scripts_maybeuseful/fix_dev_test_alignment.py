#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix dev/test alignment issue.

Problem:
  - data_w2ner_folded and data_w2ner have identical dev.json and test.json
  - This is incorrect: dev and test should be independent splits

Solution:
  - Copy the correct dev/test from data_w2ner_folded_with_dev
  - to data_w2ner_folded and data_w2ner
"""

import json
import shutil
from pathlib import Path

DATASETS = [
    "flight_orders_with_queries",
    "hotel_orders_with_queries",
    "id_cards_with_queries",
    "mixed_data_with_queries",
    "train_orders_with_queries",
]

SOURCE_ROOT = Path("/home/mengfanrong/finaldesign/W2NERproject/W2NER/data/data_w2ner_folded_with_dev")
TARGET_ROOTS = [
    Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner_folded"),
    Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner"),
]


def main():
    print("=" * 60)
    print("Fix dev/test alignment")
    print("=" * 60)

    for dataset in DATASETS:
        print(f"\n--- {dataset} ---")

        # Load source dev/test
        src_dev_path = SOURCE_ROOT / dataset / "dev.json"
        src_test_path = SOURCE_ROOT / dataset / "test.json"

        with open(src_dev_path, "r", encoding="utf-8") as f:
            src_dev = json.load(f)
        with open(src_test_path, "r", encoding="utf-8") as f:
            src_test = json.load(f)

        print(f"Source dev: {len(src_dev)} samples")
        print(f"Source test: {len(src_test)} samples")

        for target_root in TARGET_ROOTS:
            tgt_dev_path = target_root / dataset / "dev.json"
            tgt_test_path = target_root / dataset / "test.json"

            # Backup existing files
            if tgt_dev_path.exists():
                backup_dev = tgt_dev_path.with_suffix(".json.backup")
                shutil.copy(tgt_dev_path, backup_dev)
                print(f"  Backed up {tgt_dev_path} -> {backup_dev}")

            if tgt_test_path.exists():
                backup_test = tgt_test_path.with_suffix(".json.backup")
                shutil.copy(tgt_test_path, backup_test)
                print(f"  Backed up {tgt_test_path} -> {backup_test}")

            # Copy new dev
            shutil.copy(src_dev_path, tgt_dev_path)
            print(f"  Copied dev to {tgt_dev_path}")

            # Copy new test
            shutil.copy(src_test_path, tgt_test_path)
            print(f"  Copied test to {tgt_test_path}")

            # Verify
            with open(tgt_dev_path, "r", encoding="utf-8") as f:
                new_dev = json.load(f)
            with open(tgt_test_path, "r", encoding="utf-8") as f:
                new_test = json.load(f)

            assert len(new_dev) == len(src_dev), f"dev count mismatch for {dataset}"
            assert len(new_test) == len(src_test), f"test count mismatch for {dataset}"

            # Verify dev != test
            dev_json = json.dumps(new_dev, sort_keys=True)
            test_json = json.dumps(new_test, sort_keys=True)
            assert dev_json != test_json, f"dev and test are still identical for {dataset}!"

            print(f"  ✓ Verified: dev={len(new_dev)}, test={len(new_test)}, different ✓")

    print("\n" + "=" * 60)
    print("Done! All dev/test files have been fixed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
