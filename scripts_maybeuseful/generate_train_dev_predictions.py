#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 train 和 dev 集生成 W2NER 预测输出

使用方法：
python generate_train_dev_predictions.py --dataset mixed_data --device 0
"""

import json
import argparse
import torch
import prettytable as pt
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import sys
import os
# 添加 W2NER 目录到路径
W2NER_DIR = Path(__file__).parent / "W2NER"
sys.path.insert(0, str(W2NER_DIR))
os.chdir(W2NER_DIR)  # 切换到 W2NER 目录

from config import Config
from model import Model
import data_loader
import utils


def load_split_data(data_root: str, dataset: str, split: str):
    """加载指定 split 的数据"""
    path = Path(data_root) / dataset / f"{split}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(config, model, data, output_path):
    """生成预测结果"""
    model.eval()
    result = []
    
    # 创建 data loader
    batch_size = config.batch_size
    datasets, ori_data, config.vocab = data_loader.read_dataset(config, data)
    data_loader.batch_iter(datasets, ori_data, config.vocab, batch_size, if_shuffle=False)
    
    batch_start = 0
    with torch.no_grad():
        for data_batch in data_loader.batch_iter(datasets, ori_data, config.vocab, batch_size, if_shuffle=False):
            sentence_batch = ori_data[batch_start: batch_start + len(data_batch[0])]
            entity_text = data_batch[-1]
            data_batch = [data.cuda() for data in data_batch[:-1]]
            
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
            
            logits = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            probs = torch.softmax(logits, dim=-1)
            outputs = torch.argmax(logits, -1)
            
            length = sent_length
            
            ent_c, ent_p, ent_r, decode_entities = utils.decode(
                outputs.cpu().numpy(),
                entity_text,
                length.cpu().numpy()
            )
            
            procnet_decoded = utils.decode_for_procnet(
                outputs.cpu().numpy(),
                probs.cpu().numpy(),
                length.cpu().numpy()
            )
            
            for local_idx, (ent_list, procnet_ent_list, sentence_record) in enumerate(
                zip(decode_entities, procnet_decoded, sentence_batch)
            ):
                instance = utils.build_prediction_record(
                    sentence_record=sentence_record,
                    decoded_entities=ent_list,
                    procnet_decoded_entities=procnet_ent_list,
                    vocab=config.vocab,
                    sample_idx=batch_start + local_idx,
                    continuous_only=bool(getattr(config, "continuous_only", 1))
                )
                result.append(instance)
            
            batch_start += len(sentence_batch)
    
    # 保存结果
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return len(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mixed_data_with_queries",
                        help="数据集名称，如 mixed_data_with_queries")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, 
                        default="/home/mengfanrong/finaldesign/W2NERproject/W2NER/predictions",
                        help="预测输出目录")
    parser.add_argument("--model_path", type=str, 
                        default="/home/mengfanrong/finaldesign/W2NERproject/W2NER/model.pt",
                        help="训练好的模型路径")
    parser.add_argument("--config_path", type=str,
                        default="/home/mengfanrong/finaldesign/W2NERproject/W2NER/config/data_w2ner_folded.json",
                        help="配置文件路径")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("为 train/dev 集生成 W2NER 预测")
    print("=" * 70)
    
    # 加载配置
    class Args:
        config = args.config_path
        device = args.device
    
    config = Config(Args())
    config.device = args.device
    torch.cuda.set_device(args.device)
    
    # 加载模型
    print(f"\n加载模型：{args.model_path}")
    model = Model(config).cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # 生成预测
    splits = ["train", "dev"]
    
    for split in splits:
        print(f"\n处理 {split} 集...")
        
        # 加载数据
        data_path = Path("/home/mengfanrong/finaldesign/W2NERproject/data_w2ner_folded") / args.dataset / f"{split}.json"
        if not data_path.exists():
            print(f"  ⚠️  数据文件不存在：{data_path}")
            continue
        
        print(f"  加载数据：{data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"  样本数：{len(data)}")
        
        # 生成预测
        output_path = Path(args.output_dir) / f"{args.dataset}_{split}_output.json"
        print(f"  保存至：{output_path}")
        
        num_preds = generate_predictions(config, model, data, output_path)
        print(f"  生成预测：{num_preds} 条")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    
    # 列出输出文件
    print(f"\n输出文件:")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        for f in output_dir.glob(f"{args.dataset}_*_output.json"):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
