import logging
import pickle
import time
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Any, Optional, Union
import numpy as np


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type_id):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type_id)
    return text


def convert_text_to_index(text):
    index, type_id = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type_id)


# ==================== 通用图构建函数 ====================

class RelationNode:
    """通用的关系图节点，支持带分数和不带分数两种模式"""
    def __init__(self, with_scores: bool = False):
        self.with_scores = with_scores
        if with_scores:
            self.THW = []  # [(tail, type_id, thw_score)]
            self.NNW = defaultdict(dict)  # {(head, tail): {next_index: edge_score}}
        else:
            self.THW = []  # [(tail, type_id)]
            self.NNW = defaultdict(set)  # {(head, tail): {next_index}}


def build_relation_graph(outputs, scores=None, length=None):
    """
    构建关系图，支持带分数和不带分数两种模式
    
    Args:
        outputs: 模型输出 [batch, seq_len, seq_len]
        scores: 概率分数 [batch, seq_len, seq_len, num_classes] 或 None
        length: 序列长度列表 [batch]
    
    Returns:
        nodes_list: 每个样本的节点列表
        with_scores: 是否包含分数
    """
    if length is None:
        # 如果没有提供length，假设 outputs 是单个样本
        if outputs.ndim == 2:
            outputs = [outputs]
            if scores is not None:
                scores = [scores]
            length = [outputs[0].shape[0]]
        else:
            raise ValueError("必须提供 length 参数")
    
    with_scores = scores is not None
    nodes_list = []
    
    for idx in range(len(outputs)):
        instance = outputs[idx]
        prob_instance = scores[idx] if with_scores else None
        # 处理各种类型的length参数
        if isinstance(length, (list, tuple, np.ndarray)) and hasattr(length, '__len__'):
            l = int(length[idx])  # 确保转换为整数
        else:
            l = int(length)  # 标量情况
        
        nodes = [RelationNode(with_scores) for _ in range(l)]
        
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW 关系: [tail, head] -> type_id
                if instance[cur, pre] > 1:
                    type_id = int(instance[cur, pre])
                    if with_scores:
                        thw_score = float(prob_instance[cur, pre, type_id])
                        nodes[pre].THW.append((cur, type_id, thw_score))
                    else:
                        nodes[pre].THW.append((cur, type_id))
                    heads.append(pre)
                
                # NNW 关系: [pre, cur] == 1
                if pre < cur and instance[pre, cur] == 1:
                    if with_scores:
                        nnw_score = float(prob_instance[pre, cur, 1])
                        
                        # 当前 pre 直接连接到 cur
                        for head in heads:
                            old_score = nodes[pre].NNW[(head, cur)].get(cur, -1.0)
                            if nnw_score > old_score:
                                nodes[pre].NNW[(head, cur)][cur] = nnw_score
                        
                        # 从 cur 继承可达路径
                        for head, tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                old_score = nodes[pre].NNW[(head, tail)].get(cur, -1.0)
                                if nnw_score > old_score:
                                    nodes[pre].NNW[(head, tail)][cur] = nnw_score
                    else:
                        # 不带分数的版本
                        for head in heads:
                            nodes[pre].NNW[(head, cur)].add(cur)
                        
                        for head, tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head, tail)].add(cur)
            
        nodes_list.append(nodes)
    
    return nodes_list, with_scores


def decode_from_graph_fixed(nodes_list, with_scores=False):
    """
    从关系图解码实体（不带分数的版本）
    与官方 W2NER decode() 逻辑一致：每一步从链尾节点查找 NNW 边。
    """
    decode_entities = []

    for nodes in nodes_list:
        predicts = []
        q = deque()

        for cur, node in enumerate(nodes):
            for tail, type_id in node.THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue

                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx in chains:
                            continue
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])

    return decode_entities


def decode_from_graph(nodes_list, with_scores=False):
    """
    从关系图解码实体（不带分数的版本）
    与 decode_from_graph_fixed 等价，保留别名以兼容旧调用。
    """
    return decode_from_graph_fixed(nodes_list, with_scores)


def decode_for_procnet_from_graph(nodes_list):
    """
    从关系图解码实体（带分数的版本，用于ProcNet）
    与官方 W2NER decode() 逻辑一致：每一步从链尾节点查找 NNW 边，含环路检测。
    
    Returns:
        decode_entities: 每个样本的解码实体列表 [dict, ...]
    """
    decode_entities = []

    for nodes in nodes_list:
        predicts = {}
        q = deque()

        for cur, node in enumerate(nodes):
            for tail, type_id, thw_score in node.THW:
                if cur == tail:
                    key = convert_index_to_text([cur], type_id)
                    old = predicts.get(key)
                    if old is None or thw_score > old["score"]:
                        predicts[key] = {
                            "token_indices": [cur],
                            "type_id": int(type_id),
                            "score": thw_score,
                            "head": cur,
                        }
                    continue

                q.clear()
                q.append(([cur], [thw_score]))

                while len(q) > 0:
                    chains, score_trace = q.pop()
                    for idx, edge_score in nodes[chains[-1]].NNW[(cur, tail)].items():
                        if idx in chains:
                            continue
                        new_chain = chains + [idx]
                        new_scores = score_trace + [edge_score]

                        if idx == tail:
                            key = convert_index_to_text(new_chain, type_id)
                            final_score = _mean(new_scores)
                            old = predicts.get(key)
                            if old is None or final_score > old["score"]:
                                predicts[key] = {
                                    "token_indices": list(new_chain),
                                    "type_id": int(type_id),
                                    "score": final_score,
                                    "head": int(new_chain[0]),
                                }
                        else:
                            q.append((new_chain, new_scores))

        decode_entities.append(list(predicts.values()))

    return decode_entities


# ==================== 原函数（保持接口兼容） ====================

def decode(outputs, entities, length):
    """
    解码实体并计算评估指标（保持原接口）
    
    Args:
        outputs: 模型输出 [batch, seq_len, seq_len]
        entities: 每个样本的实体集合
        length: 序列长度列表 [batch]
    
    Returns:
        ent_c: 正确实体数
        ent_p: 预测实体数
        ent_r: 真实实体数
        decode_entities: 解码的实体列表
    """
    # 使用通用图构建函数（不带分数）
    nodes_list, _ = build_relation_graph(outputs, scores=None, length=length)
    
    # 解码实体
    decode_entities = decode_from_graph_fixed(nodes_list, with_scores=False)
    
    # 计算评估指标
    ent_r, ent_p, ent_c = 0, 0, 0
    for ent_set, predicts in zip(entities, decode_entities):
        # 将解码实体转换为文本表示以便比较
        predict_texts = set(convert_index_to_text(idx, type_id) for idx, type_id in predicts)
        ent_set_texts = set(ent_set)  # entities已经是文本表示
        
        ent_r += len(ent_set_texts)
        ent_p += len(predict_texts)
        ent_c += len(predict_texts.intersection(ent_set_texts))
    
    return ent_c, ent_p, ent_r, decode_entities


def _mean(scores):
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def is_contiguous(indices):
    if not indices:
        return False
    return list(indices) == list(range(indices[0], indices[-1] + 1))


def build_entity_text(sentence_tokens, token_indices):
    pieces = [str(sentence_tokens[i]) for i in token_indices]
    if not pieces:
        return ""

    # 中文字符级或字粒度输入通常直接拼接；英文/词粒度更适合空格连接
    if all(len(piece) == 1 for piece in pieces):
        return "".join(pieces)
    return " ".join(pieces)


def _safe_meta_get(record, keys, default=None):
    if not isinstance(record, dict):
        return default
    for key in keys:
        if key in record:
            return record[key]
    return default


def decode_for_procnet(outputs, scores, length):
    """
    在原 decode 的基础上，额外导出 ProcNet 更需要的结构（使用通用图构建函数）
    
    Args:
        outputs: 模型输出 [batch, seq_len, seq_len]
        scores: 概率分数 [batch, seq_len, seq_len, num_classes]
        length: 序列长度列表 [batch]
    
    Returns:
        decode_entities: 每个样本的解码实体列表 [dict, ...]
    """
    # 使用通用图构建函数（带分数）
    nodes_list, _ = build_relation_graph(outputs, scores=scores, length=length)
    
    # 解码实体（带分数版本）
    decode_entities = decode_for_procnet_from_graph(nodes_list)
    
    return decode_entities


def build_prediction_record(
    sentence_record,
    decoded_entities,
    procnet_decoded_entities,
    vocab,
    sample_idx=0,
    continuous_only=True,
):
    """
    将官方 W2NER 的 predict 输出扩展为 ProcNet 更易消费的结构。
    不要求你现在就改 data_loader；只要原始 json 里有 doc_id / sent_id，就会被自动带出。
    """
    if isinstance(sentence_record, dict):
        sentence_tokens = sentence_record.get("sentence", [])
    else:
        sentence_tokens = sentence_record

    doc_id = _safe_meta_get(sentence_record, ["doc_id", "docid", "guid", "id", "doc_key"], None)
    sent_id = _safe_meta_get(sentence_record, ["sent_id", "sentence_id", "sid"], sample_idx)

    instance = {
        "doc_id": doc_id,
        "sent_id": sent_id,
        "sentence": sentence_tokens,
        "entity": [],
        "procnet_entities": [],
    }

    for ent in decoded_entities:
        token_indices, type_id = ent
        instance["entity"].append({
            "text": [sentence_tokens[x] for x in token_indices],
            "type": vocab.id_to_label(type_id),
        })

    for ent in procnet_decoded_entities:
        token_indices = list(ent["token_indices"])
        type_id = int(ent["type_id"])

        if continuous_only and (not is_contiguous(token_indices)):
            continue

        b = token_indices[0]
        e = token_indices[-1] + 1

        if continuous_only:
            key = [b, e, type_id]
            cluster_key = [b, e, type_id]
        else:
            key = [token_indices, type_id]
            cluster_key = [token_indices, type_id]

        instance["procnet_entities"].append({
            "key": key,
            "cluster_key": cluster_key,
            "token_indices": token_indices,
            "b": b,
            "e": e,
            "type_id": type_id,
            "type": vocab.id_to_label(type_id),
            "score": round(float(ent["score"]), 6),
            "head": int(ent["head"]),
            "text": build_entity_text(sentence_tokens, token_indices),
        })

    instance["procnet_entities"].sort(key=lambda x: (x["b"], x["e"], x["type_id"]))
    return instance


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
