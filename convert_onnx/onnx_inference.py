import sys

import onnxruntime

sys.path.append('..')
import os
import re
import json
import torch
from transformers import BertTokenizer
import time
from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm


class Dict2Obj(dict):
    """让字典可以使用.调用"""

    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value





def cut_sentences(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([，。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([，。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def onnx_inference_single(ort_session, text):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    tokens = [i for i in text]
    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=args.max_seq_len,
                                        padding="max_length",
                                        truncation="longest_first",
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")
    token_ids = encode_dict['input_ids']
    attention_masks = torch.tensor(encode_dict['attention_mask'], dtype=torch.uint8)
    token_ids = to_numpy(token_ids)
    attention_masks = to_numpy(attention_masks)
    s2 = time.time()
    output = ort_session.run(None, {'input_ids': token_ids, "attention_mask": attention_masks})
    e2 = time.time()
    print('onnx耗时：', (e2 - s2))
    # print(output)
    output = output[0][0]
    output = [id2label[str(i)] for i in output[1:1 + len(text)]]
    pred_entities = get_entities(output)
    print(pred_entities)
    res = {}
    res["text"] = text
    res["labels"] = []
    for ent in pred_entities:
        res["labels"].append(
            {"ent_type": ent[0],
             "ent_text": text[ent[1]:ent[2] + 1],
             "start": ent[1],
             "end": ent[2],
             }
        )
    return res


def predict(texts, ort_ssession, max_length=64):
    sent_list = cut_sentences(texts)
    tmp = []
    final_res = []
    for sen in sent_list:
        tmp.append(sen)
        if len("".join(tmp)) > max_length:
            if len(tmp) == 1:
                sub_tmp = tmp
                tmp = []
            else:
                sub_tmp = tmp[:-1]
                tmp = [tmp[-1]]
            res = onnx_inference_single(ort_ssession, "".join(sub_tmp))
            # print(res)
            final_res.append(res)
    if len(tmp) != 0:
        res = onnx_inference_single(ort_ssession, "".join(tmp))
        # print(res)
        final_res.append(res)
    return final_res


def merge(final_res):
    res = {}
    text_list = []
    final_labels = []
    pre_len = 0
    for fr in final_res:
        labels = fr["labels"]
        text = fr["text"]
        for ents in labels:
            ent_type = ents["ent_type"]
            ent_text = ents["ent_text"]
            start = ents["start"]
            end = ents["end"]
            real_start = start + pre_len
            real_end = end + pre_len
            final_labels.append(
                {"ent_type": ent_type,
                 "ent_text": ent_text,
                 "start": real_start,
                 "end": real_end,
                 }
            )
        text_list.append(text)
        pre_len += len(text)
    res["text"] = "".join(text_list)
    res["labels"] = final_labels
    # print(res)
    return res


def trie_ner_predict(trie, text, entity_dicts=None):
    res = {}
    res["text"] = text
    res["labels"] = []
    if entity_dicts is None:
        return res
    res["labels"] = trie(text)
    return res


def merge_trie_ner(model_final_res, trie_final_res):
    if len(trie_final_res["labels"]) == 0:
        return model_final_res
    if len(model_final_res["labels"]) == 0:
        return trie_final_res
    res = {}
    trie_res_labels = trie_final_res["labels"]
    model_res_labels = model_final_res["labels"]
    labels = model_res_labels
    for label in trie_res_labels:
        if label not in model_res_labels:
            labels.append(label)
    res["text"] = model_final_res["text"]
    res["labels"] = labels
    return res


def test(fp_16_save, dev_path):
    from seqeval.metrics import classification_report
    ort_session = onnxruntime.InferenceSession(fp_16_save)
    # onnx_inference(ort_session, texts)

    with open(dev_path, "r") as fp:
        data = fp.read().strip().split("\n")
    data = data[:10]
    data = [json.loads(d) for d in data]
    print(data)
    preds = []
    trues = []
    for d in tqdm(data, total=len(data)):
        text = d["text"]
        text = "".join(text)
        true_labels = d["labels"]
        f_res = predict(text, ort_session, max_length=256)
        model_final_res = merge(f_res)
        # print(model_final_res)
        predict_labels = model_final_res["labels"]
        logit = ["O"] * len(text)
        for lab in predict_labels:
            ent_type = lab["ent_type"]
            start = lab["start"]
            end = lab["end"]
            logit[start] = "B-" + ent_type
            for i in range(start + 1, end + 1):
                logit[i] = "I-" + ent_type
        preds.append(logit)
        trues.append(true_labels)

    report = classification_report(trues, preds)
    print(report)


if __name__ == "__main__":
    # 加载模型、配置及其它的一些
    # ===================================
    ckpt_path = "../checkpoint/dgre"
    with open(os.path.join(ckpt_path, "ner_args.json"), "r", encoding="utf-8") as fp:
        args = json.load(fp)
    args = Dict2Obj(args)
    print(args)
    id2label = args.id2label
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    # 转Onnx后的推理
    # ===================================
    # pip install numpy==1.20.0
    # pip install onnxruntime-gpu
    # pip install onnxmltools
    # pip install onnxconverter_common

    # 转换为fp16
    from onnx import load_model, save_model

    fp_16_save = os.path.join(ckpt_path, "model_fp16.onnx")
    ort_session = onnxruntime.InferenceSession(fp_16_save)
    texts = "套管渗油、油位异常现象：套管表面渗漏有油渍。套管油位异常下降或者升高。处理原则：套管严重渗漏或者外绝缘破裂，需要更换时，向值班调控人员申请停运处理。套管油位异常时，应利用红外测温装置等方法检测油位，确认套管发生内漏需要处理时，向值班调控人员申请停运处理。"
    onnx_inference_single(ort_session, texts)
    f_res = predict(texts, ort_session, max_length=256)
    model_final_res = merge(f_res)

    print(model_final_res)

    fp_16_save = os.path.join(ckpt_path, "model.onnx")
    dev_path = os.path.join(args.data_path.replace("./", "../"), "dev.txt")
    test(fp_16_save, dev_path)
