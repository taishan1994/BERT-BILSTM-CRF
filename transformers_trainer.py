import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import json
import torch
import torch.nn as nn
import numpy as np

from transformers import BertTokenizer, TrainingArguments
from torchcrf import CRF
from transformers import BertModel, BertConfig, Trainer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities


class NerDataset(Dataset):
    def __init__(self,
                 data,
                 label2id,
                 max_seq_len,
                 tokenizer,
                 ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        if isinstance(text, str):
            text = list(text)
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data


class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss


class BertNer(nn.Module):
    def __init__(self,
                 bert_dir,
                 max_seq_len,
                 num_labels):
        super(BertNer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_dir)
        self.bert_config = BertConfig.from_pretrained(bert_dir)
        hidden_size = self.bert_config.hidden_size
        self.lstm_hiden = 128
        self.max_seq_len = max_seq_len
        self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(self.lstm_hiden * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)
        seq_out, _ = self.bilstm(seq_out)
        seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
        seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
        seq_out = self.linear(seq_out)
        logits = self.crf.decode(seq_out, mask=attention_mask.bool())
        logits = torch.tensor([i + [-100] * (self.max_seq_len - len(i)) for i in logits], requires_grad=False).to(
            seq_out.device)
        loss = None
        if labels is not None:
            loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
        return loss, {"logits": logits, "labels": labels}


bert_dir = "hfl/chinese-bert-wwm-ext"
data_path = "./data/dgre/ner_data"
max_seq_len = 256
train_batch_size = 8
dev_batch_size = 8
num_train_epochs = 3

tokenizer = BertTokenizer.from_pretrained(bert_dir)
with open(os.path.join(data_path, "labels.txt"), "r", encoding="utf-8") as fp:
    labels = fp.read().strip().split("\n")
bio_labels = ["O"]
for label in labels:
    bio_labels.append("B-{}".format(label))
    bio_labels.append("I-{}".format(label))

num_labels = len(bio_labels)
label2id = {label: i for i, label in enumerate(bio_labels)}
id2label = {i: label for i, label in enumerate(bio_labels)}

print("label2id：", label2id)

with open(os.path.join(data_path, "train.txt"), "r", encoding="utf-8") as fp:
    train_data = fp.read().split("\n")
train_data = [json.loads(d) for d in train_data]

with open(os.path.join(data_path, "dev.txt"), "r", encoding="utf-8") as fp:
    dev_data = fp.read().split("\n")
dev_data = [json.loads(d) for d in dev_data]

print(train_data[0])
print(dev_data[0])

train_dataset = NerDataset(train_data, label2id, max_seq_len, tokenizer)
dev_dataset = NerDataset(dev_data, label2id, max_seq_len, tokenizer)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=2)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=dev_batch_size, num_workers=2)

model = BertNer(bert_dir,
                max_seq_len,
                num_labels)

# 解决ValueError: You are trying to save a non contiguous tensor: `bert.encoder.layer.0.attention.self.query.weight` which is not allowed. It either means you are trying to save tensors which are reference of each other in which case it's recommended to save only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to pack it before saving.
for param in model.parameters():
    param.data = param.data.contiguous()

training_args = TrainingArguments(
    output_dir='./results',  # output directory 结果输出地址
    num_train_epochs=num_train_epochs,  # total # of training epochs 训练总批次
    per_device_train_batch_size=train_batch_size,  # batch size per device during training 训练批大小
    per_device_eval_batch_size=dev_batch_size,  # batch size for evaluation 评估批大小
    logging_dir='./logs/',  # directory for storing logs 日志存储位置
    # learning_rate=3e-5,  # 学习率
    save_steps=False,  # 不保存检查点
    logging_strategy="steps",
    evaluation_strategy="steps",
    logging_steps=1,
    max_grad_norm=1,
    eval_steps=10,
    do_eval=True,
    do_train=True,
)


def compute_metrics(pred):
    preds = pred.predictions
    logits = preds[0]
    labels = preds[1]

    preds = []
    trues = []

    batch_size = logits.shape[0]

    for i in range(batch_size):
        length = sum(logits[i] != -100)

        logit = logits[i][1:length]
        logit = [id2label[i] for i in logit]
        label = labels[i][1:length]
        label = [id2label[i] for i in label]

        preds.append(logit)
        trues.append(label)

    report = classification_report(trues, preds)
    return {"report": report}


class BertBilstmCrfTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs[1]) if return_outputs else loss


module = (
    model.module if hasattr(model, "module") else model
)

# 差分学习率
no_decay = ["bias", "LayerNorm.weight"]
model_param = list(module.named_parameters())

bert_param_optimizer = []
other_param_optimizer = []

for name, para in model_param:
    space = name.split('.')
    # print(name)
    if space[0] == 'bert_module' or space[0] == "bert":
        bert_param_optimizer.append((name, para))
    else:
        other_param_optimizer.append((name, para))

optimizer_grouped_parameters = [
    # bert other module
    {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01, 'lr': 3e-5},
    {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0, 'lr': 3e-5},

    # 其他模块，差分学习率
    {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01, 'lr': 3e-3},
    {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0, 'lr': 3e-3},
]

t_total = len(train_loader) * num_train_epochs

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.01 * t_total), num_training_steps=t_total
)

trainer = BertBilstmCrfTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
    args=training_args,  # training arguments, defined above 训练参数
    train_dataset=train_dataset,  # training dataset 训练集
    eval_dataset=dev_dataset,  # evaluation dataset 测试集
    compute_metrics=compute_metrics,  # 计算指标方法
    optimizers=(optimizer, scheduler),
)

trainer.train()
trainer.save_model()
trainer.evaluate()


class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        # self.model.load_state_dict(
        #     torch.load("/data/gongoubo/BERT-BILSTM-CRF/results/checkpoint-66/model.safetensors", map_location="cpu"))
        self.model.to(self.device)
        self.id2label = id2label

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        _, output = self.model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output["logits"]
        logits = logits.detach().cpu().numpy().tolist()
        logits = logits[0][1:length - 1]
        logits = [self.id2label[i] for i in logits]
        entities = get_entities(logits)
        return entities


texts = [
    "492号汽车故障报告故障现象一辆车用户用水清洗发动机后，在正常行驶时突然产生铛铛异响，自行熄火",
    "故障现象：空调制冷效果差。",
    "原因分析：1、遥控器失效或数据丢失;2、ISU模块功能失效或工作不良;3、系统信号有干扰导致。处理方法、体会：1、检查该车发现，两把遥控器都不能工作，两把遥控器同时出现故障的可能几乎是不存在的，由此可以排除遥控器本身的故障。2、检查ISU的功能，受其控制的部分全部工作正常，排除了ISU系统出现故障的可能。3、怀疑是遥控器数据丢失，用诊断仪对系统进行重新匹配，发现遥控器匹配不能正常进行。此时拔掉ISU模块上的电源插头，使系统强制恢复出厂设置，再插上插头，发现系统恢复，可以进行遥控操作。但当车辆发动在熄火后，遥控又再次失效。4、查看线路图发现，在点火开关处安装有一钥匙行程开关，当钥匙插入在点火开关内，处于ON位时，该开关接通，向ISU发送一个信号，此时遥控器不能进行控制工作。当钥匙处于OFF位时，开关断开，遥控器恢复工作，可以对门锁进行控制。如果此开关出现故障，也会导致遥控器不能正常工作。同时该行程开关也控制天窗的自动回位功能。测试天窗发现不能自动回位。确认该开关出现故障",
    "原因分析：1、发动机点火系统不良;2、发动机系统油压不足;3、喷嘴故障;4、发动机缸压不足;5、水温传感器故障。",
]

predictor = Predictor()
for text in texts:
    tmp = []
    ner_result = predictor.ner_predict(text)
    for t in ner_result:
        tmp.append(("".join(list(text)[t[1]:t[2]+1]), t[0], t[1], t[2]))
    print("文本>>>>>：", text)
    print("实体>>>>>：", tmp)
    print("=" * 100)
