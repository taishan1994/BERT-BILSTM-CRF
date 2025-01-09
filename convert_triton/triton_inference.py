import codecs
import base64


def decode_unicode(data):
    if isinstance(data, str):
        return codecs.decode(data, 'unicode_escape')
    elif isinstance(data, list):
        return [decode_unicode(item) for item in data]
    elif isinstance(data, dict):
        return {key: decode_unicode(value) for key, value in data.items()}
    else:
        return data


if __name__ == "__main__":
    model_name = "onnx_model"
    model_version = "1"
    text = "套管渗油、油位异常现象：套管表面渗漏有油渍。套管油位异常下降或者升高。处理原则：套管严重渗漏或者外绝缘破裂，需要更换时，向值班调控人员申请停运处理。套管油位异常时，应利用红外测温装置等方法检测油位，确认套管发生内漏需要处理时，向值班调控人员申请停运处理。"
    encoded_data = base64.b64encode(text.encode("utf-8")).decode('utf-8')
    raw_data = {
        "inputs": [
            {
                "name": "text",
                "datatype": "BYTES",
                "shape": [1, 1],
                "data": [[encoded_data]]
            }
        ],
        "outputs": [
            {
                "name": "res"
            }
        ]
    }

    url = "http://192.168.16.6:8000/v2/models/onnx_model/infer"
    import requests
    import json

    data = json.dumps(raw_data, ensure_ascii=True)
    import  time
    t1 = time.time()
    response = requests.post(url=url,
                             data=data,
                             headers={"Content_Type": "application/json"},
                             timeout=2000)
    t2 = time.time()
    print("耗时：{}s".format(t2-t1))
    print(response.content)
    res = json.loads(response.text)
    res = res["outputs"]
    tmp = {}

    for r in res:
        d = r["data"]
        for t in d:
            tmp = decode_unicode(t)

    print(tmp)


