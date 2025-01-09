import os
import json
import logging
import base64
import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils

from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        logger.info(self.model_config)
        self.tokenizer = BertTokenizer.from_pretrained("/models/model_hub/chinese-bert-wwm-ext/")
        self.max_seq_len = 512
        # 加载您的 ONNX 模型
        self.session = ort.InferenceSession("/models/triton_model/onnx_model/1/model.onnx")
        self.id2label = {0: 'O', 1: 'B-故障设备', 2: 'I-故障设备', 3: 'B-故障原因', 4: 'I-故障原因'}

    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        # output0_dtype = self.output0_dtype
        # output1_dtype = self.output1_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "text")
            logger.info(in_0.as_numpy().tolist()[0][0])
            decoded_strings = base64.b64decode(in_0.as_numpy().tolist()[0][0]).decode()
            text = decoded_strings
            text_chars = list(text)
            logger.info(text_chars)
            text_chars = text_chars[:self.max_seq_len - 2]
            text_chars = ["[CLS]"] + text_chars + ["[SEP]"]
            tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text_chars)
            input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
            attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
            # model inferencece clent
            input_ids = np.array([input_ids], dtype=np.int64)
            attention_mask = np.array([attention_mask], dtype=np.uint8)
            logger.info(input_ids)
            logger.info(attention_mask)

            # inference_request = pb_utils.InferenceRequest(
            #     model_name='onnx_model',
            #     requested_output_names=["logits"],
            #     inputs=[pb_utils.Tensor("input_ids", input_ids),
            #             pb_utils.Tensor("attention_mask", attention_mask)])
            # # model forward
            # inference_response = inference_request.exec()
            # # 确保推理响应中包含数据
            # if inference_response.has_error():
            #     logger.error("Inference error: %s", inference_response.error().message())
            #
            # logger.info(inference_response)
            # # get output tensor
            # output = pb_utils.get_output_tensor_by_name(inference_response, 'logits')

            ort_outputs = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

            logger.info(ort_outputs)

            # 获取输出
            output = ort_outputs[0][0]  # 假设 logits 是第一个输出
            output = [self.id2label[i] for i in output[1:1 + len(text)]]

            pred_entities = get_entities(output)
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

            logger.info(res)
            res_json = json.dumps(res)
            res_np = np.array([res_json], dtype=object)

            # 创建输出张量
            output_tensor = pb_utils.Tensor("res", res_np)

            # to response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
