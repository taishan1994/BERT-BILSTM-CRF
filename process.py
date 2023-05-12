# -*- coding: utf-8 -*-
import os
import re
import json
import codecs
import random
import codecs
from tqdm import tqdm
from collections import defaultdict


class ProcessDgreData:
    def __init__(self):
        self.data_path = "./data/dgre/"
        self.train_file = self.data_path + "ori_data/train.json"

    def get_ner_data(self):
        with codecs.open(self.train_file, 'r', encoding="utf-8", errors="replace") as fp:
            data = fp.readlines()
        res = []
        for did, d in enumerate(data):
            d = eval(d)
            tmp = {}
            text = d['text']
            tmp["id"] = d['ID']
            tmp['text'] = [i for i in text]
            tmp["labels"] = ["O"] * len(tmp['text'])
            for rel_id, spo in enumerate(d['spo_list']):
                h = spo['h']
                t = spo['t']
                h_start = h["pos"][0]
                h_end = h["pos"][1]
                t_start = t["pos"][0]
                t_end = t["pos"][1]
                tmp["labels"][h_start] = "B-故障设备"
                for i in range(h_start + 1, h_end):
                    tmp["labels"][i] = "I-故障设备"
                tmp["labels"][t_start] = "B-故障原因"
                for i in range(t_start + 1, t_end):
                    tmp["labels"][i] = "I-故障原因"
            res.append(tmp)
        train_ratio = 0.92
        train_num = int(len(res) * 0.92)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path + "ner_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "ner_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))

        # 这里标签一般从数据中处理得到，这里我们自定义
        labels = ["故障设备", "故障原因"]
        with open(self.data_path + "ner_data/labels.txt", "w") as fp:
            fp.write("\n".join(labels))

    def get_re_data(self):
        with codecs.open(self.train_file, 'r', encoding="utf-8", errors="replace") as fp:
            data = fp.readlines()
        res = []
        re_labels = set()
        for did, d in enumerate(tqdm(data)):
            d = eval(d)
            text = d['text']
            gzsbs = []  # 存储故障设备
            gzyys = []  # 存储故障原因
            sbj_obj = []  # 存储真实的故障设备-故障原因
            for rel_id, spo in enumerate(d['spo_list']):
                tmp = {}
                tmp['text'] = text
                tmp["labels"] = []
                h = spo['h']
                t = spo['t']
                h_name = h["name"]
                t_name = t["name"]
                relation = spo["relation"]
                tmp_rel_id = str(did) + "_" + str(rel_id)
                tmp["id"] = tmp_rel_id
                tmp["labels"] = [h_name, t_name, relation]
                re_labels.add(relation)
                res.append(tmp)
                if h_name not in gzsbs:
                    gzsbs.append(h_name)
                if t_name not in gzyys:
                    gzyys.append(t_name)
                sbj_obj.append((h_name, t_name))

            # 关键是怎么构造负样本
            # 如果不在sbj_obj里则视为没有关系
            tmp = {}
            tmp["text"] = text
            tmp["labels"] = []
            tmp["id"] = str(did) + "_" + "norel"
            if len(gzsbs) > 1 and len(gzyys) > 1:
                neg_total = 3
                neg_cur = 0
                for gzsb in gzsbs:
                    random.shuffle(gzyys)
                    print(gzyys)
                    for gzyy in enumerate(gzyys):
                        if (gzsb, gzyy[1]) not in sbj_obj:
                            # print([gzsb, gzyy[1], "没关系"])
                            tmp["labels"] = [gzsb, gzyy[1], "没关系"]
                            res.append(tmp)
                            neg_cur += 1
                        break
                    if neg_cur == neg_total:
                        break

        train_ratio = 0.92
        train_num = int(len(res) * 0.92)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path + "re_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "re_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))

        # 这里标签一般从数据中处理得到，这里我们自定义
        labels = list(re_labels) + ["没关系"]
        with open(self.data_path + "re_data/labels.txt", "w") as fp:
            fp.write("\n".join(labels))


class ProcessDuieData:
    def __init__(self):
        self.data_path = "./data/duie/"
        self.train_file = self.data_path + "ori_data/duie_train.json"
        self.dev_file = self.data_path + "ori_data/duie_dev.json"
        self.test_file = self.data_path + "ori_data/duie_test2.json"
        self.schema_file = self.data_path + "ori_data/duie_schema.json"

    def get_ents(self):
        ents = set()
        rels = defaultdict(list)
        with open(self.schema_file, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                data = eval(line)
                subject_type = data['subject_type']['@value'] if '@value' in data['subject_type'] else data[
                    'subject_type']
                object_type = data['object_type']['@value'] if '@value' in data['object_type'] else data['object_type']
                if "人物" in subject_type:
                    subject_type = "人物"
                if "人物" in object_type:
                    object_type = "人物"
                ents.add(subject_type)
                ents.add(object_type)
                predicate = data["predicate"]
                rels[subject_type + "_" + object_type].append(predicate)

        with open(self.data_path + "ner_data/labels.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(list(ents)))

        with open(self.data_path + "re_data/rels.txt", "w", encoding="utf-8") as fp:
            json.dump(rels, fp, ensure_ascii=False, indent=2)

    def get_ner_data(self, input_file, output_file):
        res = []
        with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
            lines = fp.read().strip().split("\n")
            for i, line in enumerate(tqdm(lines)):
                try:
                    line = eval(line)
                except Exception as e:
                    continue
                tmp = {}
                text = line['text']
                tmp['text'] = [i for i in text]
                tmp["labels"] = ["O"] * len(text)
                tmp['id'] = i
                spo_list = line['spo_list']
                for j, spo in enumerate(spo_list):
                    # 从句子里面找到实体的开始位置、结束位置
                    if spo['subject'] == "" or spo['object']['@value'] == "":
                        continue
                    try:
                        subject_re_res = re.finditer(re.escape(spo['subject']), line['text'])
                        subject_type = spo["subject_type"]
                        if "人物" in subject_type:
                            subject_type = "人物"
                    except Exception as e:
                        print(e)
                        print(spo['subject'].replace('+', '\+'), line['text'])
                        import sys
                        sys.exit(0)
                    for sbj in subject_re_res:
                        sbj_span = sbj.span()
                        sbj_start = sbj_span[0]
                        sbj_end = sbj_span[1]
                        tmp["labels"][sbj_start] = f"B-{subject_type}"
                        for j in range(sbj_start + 1, sbj_end):
                            tmp["labels"][j] = f"I-{subject_type}"
                    try:
                        object_re_res = re.finditer(
                            re.escape(spo['object']['@value']), line['text'])
                        object_type = spo['object_type']['@value']
                        if "人物" in object_type:
                            object_type = "人物"
                    except Exception as e:
                        print(e)
                        print(line)
                        print(spo['object']['@value'].replace('+', '\+').replace('(', ''), line['text'])
                        import sys
                        sys.exit(0)
                    for obj in object_re_res:
                        obj_span = obj.span()
                        obj_start = obj_span[0]
                        obj_end = obj_span[1]
                        tmp["labels"][obj_start] = f"B-{object_type}"
                        for j in range(obj_start + 1, obj_end):
                            tmp["labels"][j] = f"I-{object_type}"
                res.append(tmp)

        with open(output_file, 'w', encoding="utf-8") as fp:
            fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))


if __name__ == "__main__":
    processDgreData = ProcessDgreData()
    processDgreData.get_ner_data()

    processDuieData = ProcessDuieData()
    processDuieData.get_ents()
    processDuieData.get_ner_data(processDuieData.train_file,
                                os.path.join(processDuieData.data_path, "ner_data/train.txt"))
    processDuieData.get_ner_data(processDuieData.dev_file, os.path.join(processDuieData.data_path, "ner_data/dev.txt"))
