import os
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer
import openvino.runtime as ov
from optimum.intel.openvino import OVModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

from warp.modeling.xtr import XTRTokenizer, XTRLinear

OPENVINO_MODEL_FILENAME = "openvino_encoder_model.xml"

@dataclass
class XTROpenVinoConfig:
    batch_size: int = 1
    num_threads: int = 1

class XTROpenVinoModel:
    def __init__(self, config: XTROpenVinoConfig):
        assert config.batch_size == 1

        OPENVINO_DIR = os.environ["OPENVINO_MODEL_DIR"]
        XTROpenVinoModel._quantize_model_if_not_exists(OPENVINO_DIR, config)

        self.tokenizer = XTRTokenizer(
            AutoTokenizer.from_pretrained("google/xtr-base-en")
        )
        self.linear = XTRLinear().to(self.device)
        self.linear.load_state_dict(torch.load(XTROpenVinoModel._hf_download_to_dense()))

        core = ov.Core()
        model_path = os.path.join(OPENVINO_DIR, OPENVINO_MODEL_FILENAME)
        model = core.read_model(model_path)
        ov_config = {
            "NUM_STREAMS": 1,
            "INFERENCE_NUM_THREADS": config.num_threads,
        }
        compiled_model = core.compile_model(model, "CPU", ov_config)
        self.infer_request = compiled_model.create_infer_request()

    @staticmethod
    def _hf_download_to_dense():
        return hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")

    @staticmethod
    def _quantize_model_if_not_exists(root_dir, config):
        model_path = os.path.join(root_dir, OPENVINO_MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"#> Exporting OpenVINO model at '{root_dir}'")
            hf_model = OVModelForSeq2SeqLM.from_pretrained("google/xtr-base-en", export=True, use_cache=False)
            hf_model.save_pretrained(root_dir)
            XTROpenVinoModel._hf_download_to_dense()

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        mask = (input_ids != 0).unsqueeze(2).float()
        self.infer_request.set_input_tensor(0, ov.Tensor(input_ids.numpy()))
        self.infer_request.set_input_tensor(1, ov.Tensor(attention_mask.numpy()))
        Q = self.linear(torch.from_numpy(self.infer_request.infer()[0])) * mask
        return torch.nn.functional.normalize(Q, dim=2)