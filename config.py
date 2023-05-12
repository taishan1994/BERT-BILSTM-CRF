import os
import torch
import json

from transformers import BertTokenizer


class CommonConfig:
    bert_dir = "./model_hub/chinese-bert-wwm-ext/"
    output_dir = "./checkpoint/"
    data_dir = "./data/"


class NerConfig:
    def __init__(self, data_name):
        cf = CommonConfig()
        self.bert_dir = cf.bert_dir
        self.output_dir = cf.output_dir
        self.output_dir = os.path.join(self.output_dir, data_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = cf.data_dir

        self.data_path = os.path.join(os.path.join(self.data_dir, data_name), "ner_data")
        with open(os.path.join(self.data_path, "labels.txt"), "r") as fp:
            self.labels = fp.read().strip().split("\n")
        self.bio_labels = ["O"]
        for label in self.labels:
            self.bio_labels.append("B-{}".format(label))
            self.bio_labels.append("I-{}".format(label))
        print(self.bio_labels)
        self.num_labels = len(self.bio_labels)
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        print(self.label2id)
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}

        self.max_seq_len = 512
        self.epochs = 3
        self.train_batch_size = 12
        self.dev_batch_size = 12
        self.bert_learning_rate = 3e-5
        self.crf_learning_rate = 3e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500
