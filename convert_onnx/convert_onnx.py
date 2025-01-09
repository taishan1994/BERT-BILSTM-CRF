import sys

sys.path.append('..')
import os
import json
import logging
from pprint import pprint
import torch
import numpy as np
from transformers import BertTokenizer
import time
from seqeval.metrics.sequence_labeling import get_entities
from onnx_model import BertNer


class Dict2Obj(dict):
    """让字典可以使用.调用"""

    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value


class ConverttOnnx:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.device = "cuda:0"
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.id2label = args.id2label
        print(self.id2label)

    def inference(self, texts):
        self.model.eval()
        with torch.no_grad():
            tokens = [i for i in texts]
            encode_dict = self.tokenizer.encode_plus(text=tokens,
                                                     max_length=self.args.max_seq_len,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True,
                                                     return_attention_mask=True,
                                                     return_tensors="pt")
            token_ids = encode_dict['input_ids'].to(self.device)
            attention_masks = encode_dict['attention_mask'].bool().to(self.device)
            s1 = time.time()
            for i in range(NUM):
                logits = self.model(token_ids, attention_masks)
            # print(logits)
            e1 = time.time()
            print('原版耗时：', (e1 - s1) / NUM)
            output = logits
            output = output.detach().cpu().numpy()
            output = output[0]
            output = [self.id2label[str(i)] for i in output[1:1 + len(texts)]]
            pred_entities = get_entities(output)
            res = {}
            res["text"] = texts
            res["labels"] = []
            for ent in pred_entities:
                res["labels"].append(
                    {"ent_type": ent[0],
                     "ent_text": texts[ent[1]:ent[2] + 1],
                     "start": ent[1],
                     "end": ent[2],
                     }
                )
            print(res)

    def convert(self, save_path):
        self.model.eval()
        inputs = {'input_ids': torch.ones(1, args.max_seq_len, dtype=torch.long),
                  'attention_mask': torch.ones(1, args.max_seq_len, dtype=torch.uint8),
                  }

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (inputs["input_ids"].to(self.device),
                 inputs["attention_mask"].to(self.device)),
                save_path,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={'input_ids': symbolic_names,
                              'attention_mask': symbolic_names,
                              'logits': symbolic_names}
            )

    def onnx_inference(self, ort_session, texts):
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        tokens = [i for i in texts]
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
        for i in range(NUM):
            output = ort_session.run(None, {'input_ids': token_ids, "attention_mask": attention_masks})
        e2 = time.time()
        print('onnx耗时：', (e2 - s2) / NUM)
        # print(output)
        output = output[0][0]
        output = [self.id2label[str(i)] for i in output[1:1 + len(texts)]]
        pred_entities = get_entities(output)
        res = {}
        res["text"] = texts
        res["labels"] = []
        for ent in pred_entities:
            res["labels"].append(
                {"ent_type": ent[0],
                 "ent_text": texts[ent[1]:ent[2] + 1],
                 "start": ent[1],
                 "end": ent[2],
                 }
            )
        print(res)


if __name__ == "__main__":
    # 加载模型、配置及其它的一些
    # ===================================
    ckpt_path = "../checkpoint/dgre/"
    model_path = os.path.join(ckpt_path, "pytorch_model_ner.bin")
    with open(os.path.join(ckpt_path, "ner_args.json"), "r", encoding="utf-8") as fp:
        args = json.load(fp)
    print(args)
    NUM = 10
    args = Dict2Obj(args)
    print(args)
    model = BertNer(args)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    pprint('Load ckpt from {}'.format(ckpt_path))

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    # ===================================

    # 定义转换器
    # ===================================
    convertOnnx = ConverttOnnx(args, model, tokenizer)
    # ===================================

    texts = "套管渗油、油位异常现象：套管表面渗漏有油渍。套管油位异常下降或者升高。处理原则：套管严重渗漏或者外绝缘破裂，需要更换时，向值班调控人员申请停运处理。套管油位异常时，应利用红外测温装置等方法检测油位，确认套管发生内漏需要处理时，向值班调控人员申请停运处理。"

    # 一般的推理
    # ===================================
    convertOnnx.inference(texts)
    # ===================================
    #
    # 转换成onnx
    # ===================================
    save_path = os.path.join(ckpt_path, "model.onnx")
    convertOnnx.convert(save_path)
    # ===================================

    # 转Onnx后的推理
    # ===================================
    # pip install numpy==1.20.0
    # pip install onnxruntime-gpu
    # pip install onnxmltools
    # pip install onnxconverter_common
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(save_path)
    convertOnnx.onnx_inference(ort_session, texts)

    # 转换为fp16
    from onnx import load_model, save_model
    from onnxmltools.utils import float16_converter

    onnx_model = load_model(save_path)
    trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
    fp_16_save = os.path.join(ckpt_path,"model_fp16.onnx")
    save_model(trans_model, fp_16_save)

    ort_session = onnxruntime.InferenceSession(fp_16_save)
    convertOnnx.onnx_inference(ort_session, texts)
