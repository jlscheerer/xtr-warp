import os
from dataclasses import dataclass
from enum import Enum, auto

import torch
from transformers import AutoTokenizer

import onnxruntime as ort
from onnxruntime import quantization as ort_quantization
from onnxruntime.transformers import optimizer as transformers_optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType

from warp.modeling.xtr import QUERY_MAXLEN, build_xtr_model, XTRTokenizer



class XTROnnxQuantization(Enum):
    NONE = auto()
    PREPROCESS = auto()
    DYN_QUANTIZED_QINT8 = auto()
    QUANTIZED_QATTENTION = auto()


@dataclass
class XTROnnxConfig:
    batch_size: int = 1
    opset_version: int = 16
    quantization: XTROnnxQuantization = XTROnnxQuantization.NONE

    @property
    def base_name(self):
        return f"xtr.v={self.opset_version}.batch={self.batch_size}"

    @property
    def base_filename(self):
        return f"{self.base_name}.onnx"

    @property
    def filename(self):
        if self.quantization == XTROnnxQuantization.NONE:
            return self.base_filename
        return f"{self.base_name}.{self.quantization.name}.onnx"


class XTROnnxModel:
    def __init__(self, config: XTROnnxConfig):
        ONNX_DIR = os.environ["ONNX_MODEL_DIR"]
        XTROnnxModel._quantize_model_if_not_exists(ONNX_DIR, config)

        model_path = os.path.join(ONNX_DIR, config.filename)
        filesize = os.path.getsize(model_path) / (1024 * 1024)

        print(f"#> Loading XTR ONNX model from '{model_path}' ({round(filesize, 2)}MB)")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.model = ort.InferenceSession(
            model_path,
            sess_opts=sess_opts,
        )

        self.tokenizer = XTRTokenizer(
            AutoTokenizer.from_pretrained("google/xtr-base-en")
        )

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        return torch.from_numpy(
            self.model.run(
                ["Q"],
                {
                    "input_ids": input_ids.numpy(),
                    "attention_mask": attention_mask.numpy(),
                },
            )[0]
        )

    @staticmethod
    def _quantize_model_if_not_exists(root_dir, config: XTROnnxConfig):
        base_model_path = os.path.join(root_dir, config.base_filename)
        if not os.path.exists(base_model_path):
            print(f"#> Exporting XTR base model to .onnx at '{root_dir}'")
            os.makedirs(root_dir, exist_ok=True)
            base_model = build_xtr_model()
            device = torch.device("cpu")
            input_dim = (config.batch_size, QUERY_MAXLEN)
            attention_mask = torch.randint(
                low=1, high=1000, size=input_dim, dtype=torch.int64
            ).to(device)
            input_ids = torch.randint(
                low=1, high=1000, size=input_dim, dtype=torch.int64
            ).to(device)

            base_model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    base_model,
                    args=(input_ids, {"attention_mask": attention_mask}),
                    f=str(base_model_path),
                    opset_version=config.opset_version,
                    do_constant_folding=True,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["Q"],
                )

        if config.quantization == XTROnnxQuantization.NONE:
            return

        quantization = config.quantization

        config.quantization = XTROnnxQuantization.PREPROCESS
        preprocessed_model_path = os.path.join(root_dir, config.filename)
        config.quantization = quantization

        if not os.path.exists(preprocessed_model_path):
            print(
                f"#> Creating preprocessed XTR ONNX model at '{preprocessed_model_path}'"
            )
            ort_quantization.shape_inference.quant_pre_process(
                base_model_path, preprocessed_model_path, skip_symbolic_shape=False
            )

        model_path = os.path.join(root_dir, config.filename)
        if config.quantization == XTROnnxQuantization.PREPROCESS or os.path.exists(
            model_path
        ):
            return

        print(
            f"#> Creating quantized ({config.quantization.name}) XTR ONNX model at '{model_path}'"
        )
        if config.quantization == XTROnnxQuantization.DYN_QUANTIZED_QINT8:
            quantize_dynamic(
                preprocessed_model_path, model_path, weight_type=QuantType.QInt8
            )
            return

        if config.quantization == XTROnnxQuantization.QUANTIZED_QATTENTION:
            optimized_model = transformers_optimizer.optimize_model(
                preprocessed_model_path, "bert", num_heads=12, hidden_size=768
            )
            optimized_model.save_model_to_file(model_path)
            return

        assert False  # We should never reach this point.
