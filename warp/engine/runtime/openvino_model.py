import os
from dataclasses import dataclass

import openvino.runtime as ov
from transformers import AutoTokenizer
import torch

from warp.modeling.xtr import XTRTokenizer
from warp.engine.runtime.onnx_model import XTROnnxModel, XTROnnxConfig, XTROnnxQuantization

@dataclass
class XTROpenVinoConfig:
    batch_size: int = 1
    opset_version: int = 16
    quantization: XTROnnxQuantization = XTROnnxQuantization.NONE

    @property
    def onnx(self):
        return XTROnnxConfig(batch_size=self.batch_size, opset_version=self.opset_version,
                             quantization=self.quantization)


# TODO(jlscheerer) Support quantization here.
class XTROpenVinoModel:
    def __init__(self, config: XTROpenVinoConfig):
        assert config.batch_size == 1

        ONNX_DIR = os.environ["ONNX_MODEL_DIR"]
        XTROnnxModel._quantize_model_if_not_exists(ONNX_DIR, config.onnx)

        self.tokenizer = XTRTokenizer(
            AutoTokenizer.from_pretrained("google/xtr-base-en")
        )

        core = ov.Core()
        ONNX_MODEL_DIR = os.environ["ONNX_MODEL_DIR"]
        model_path = os.path.join(ONNX_MODEL_DIR, config.onnx.filename)

        model = core.read_model(model_path)
        ov_config = {"NUM_STREAMS": "1", "INFERENCE_NUM_THREADS": 1}
        compiled_model = core.compile_model(model, "CPU", ov_config)
        self.infer_request = compiled_model.create_infer_request()

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        self.infer_request.set_input_tensor(0, ov.Tensor(input_ids.numpy()))
        self.infer_request.set_input_tensor(1, ov.Tensor(attention_mask.numpy()))
        result = self.infer_request.infer()
        return torch.from_numpy(result[0])