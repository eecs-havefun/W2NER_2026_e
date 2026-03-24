import logging
import pickle
import time
from collections import defaultdict, deque


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


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []  # [(tail, type_id)]
            self.NNW = defaultdict(set)  # {(head, tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()

    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]

        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW
                if instance[cur, pre] > 1:
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)

                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)

            # entity
            for tail, type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue

                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])

        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))

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
    在原 decode 的基础上，额外导出 ProcNet 更需要的结构：
    - token_indices
    - type_id
    - score: 基于 THW + NNW 边的平均置信度
    - head: 默认取 token_indices[0]

    注意：
    1. 这里不依赖 gold entities，因此适合直接用于 test/predict 导出。
    2. 对不连续实体也会保留 token_indices；Phase 1 可在上层用 continuous_only 过滤。
    """
    class Node:
        def __init__(self):
            self.THW = []  # [(tail, type_id, thw_score)]
            self.NNW = defaultdict(dict)  # {(head, tail): {next_index: edge_score}}

    decode_entities = []
    q = deque()

    for instance, prob_instance, l in zip(outputs, scores, length):
        predicts = {}
        nodes = [Node() for _ in range(l)]

        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW: [tail, head] -> type_id
                if instance[cur, pre] > 1:
                    type_id = int(instance[cur, pre])
                    thw_score = float(prob_instance[cur, pre, type_id])
                    nodes[pre].THW.append((cur, type_id, thw_score))
                    heads.append(pre)

                # NNW: [pre, cur] == 1
                if pre < cur and instance[pre, cur] == 1:
                    nnw_score = float(prob_instance[pre, cur, 1])

                    # 当前 pre 直接连接到 cur
                    for head in heads:
                        old_score = nodes[pre].NNW[(head, cur)].get(cur, -1.0)
                        if nnw_score > old_score:
                            nodes[pre].NNW[(head, cur)][cur] = nnw_score

                    # 从 cur 继承“以 cur 作为下一跳”的可达路径
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            old_score = nodes[pre].NNW[(head, tail)].get(cur, -1.0)
                            if nnw_score > old_score:
                                nodes[pre].NNW[(head, tail)][cur] = nnw_score

            for tail, type_id, thw_score in nodes[cur].THW:
                if cur == tail:
                    key = convert_index_to_text([cur], type_id)
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
                    next_nodes = nodes[chains[-1]].NNW[(cur, tail)]

                    for idx, edge_score in next_nodes.items():
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
