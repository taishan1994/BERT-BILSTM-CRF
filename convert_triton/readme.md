# 执行指令
```shell script
docker pull ramonjuicelabs/fastertransformer-triton:1

docker run -it --rm --gpus \"device=2\" --ipc=host -p 8000:8000 -p 8001:8001 -p 8002:8002  -v /data/gongoubo/BERT-BILSTM-CRF/convert_triton/model_repository:/models ramonjuicelabs/fastertransformer-triton:1 bash
```
进到镜像以后，安装一些依赖：
```shell script
pip install seqeval
pip install onnxruntime
pip install transformers
pip install torch
```
缺什么安装什么

# 启动服务
```shell script
tritonserver --model-repository=/models/triton_model/
```
成功后会看到：
```shell script
I0109 10:49:40.446142 935 grpc_server.cc:4819] Started GRPCInferenceService at 0.0.0.0:8001
I0109 10:49:40.446392 935 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
I0109 10:49:40.487963 935 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002
```
使用triton_inference.py调用：
```shell script
{"text": "套管渗油、油位异常现象：套管表面渗漏有油渍。套管油位异常下降或者升高。处理原则：套管严重渗漏或者外绝缘破裂，需要更换时，向值班调控人员申请停运处理。套管油位异常时，应利用红外测温装置等方法检测油位，确认套管发生内漏需要处理时，向值班调控人员申请停运处理。", "labels": [{"ent_type": "故障设备", "ent_text": "套管", "start": 0, "end": 1}, {"ent_type": "故障原因", "ent_text": "渗油", "start": 2, "end": 3}, {"ent_type": "故障设备", "ent_text": "油位", "start": 5, "end": 6}, {"ent_type": "故障原因", "ent_text": "异常", "start": 7, "end": 8}, {"ent_type": "故障设备", "ent_text": "套管", "start": 12, "end": 13}, {"ent_type": "故障原因", "ent_text": "渗漏有油渍", "start": 16, "end": 20}, {"ent_type": "故障设备", "ent_text": "套管", "start": 22, "end": 23}, {"ent_type": "故障原因", "ent_text": "油位异常下降", "start": 24, "end": 29}, {"ent_type": "故障原因", "ent_text": "升高", "start": 32, "end": 33}, {"ent_type": "故障设备", "ent_text": "套管", "start": 40, "end": 41}, {"ent_type": "故障原因", "ent_text": "渗漏", "start": 44, "end": 45}, {"ent_type": "故障设备", "ent_text": "外绝缘", "start": 48, "end": 50}, {"ent_type": "故障原因", "ent_text": "破裂", "start": 51, "end": 52}, {"ent_type": "故障设备", "ent_text": "套管", "start": 74, "end": 75}, {"ent_type": "故障原因", "ent_text": "油位异常", "start": 76, "end": 79}, {"ent_type": "故障设备", "ent_text": "套管", "start": 101, "end": 102}, {"ent_type": "故障原因", "ent_text": "内漏", "start": 105, "end": 106}]}
```