import os
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from warp.modeling.xtr import QUERY_MAXLEN, build_xtr_model, XTRTokenizer

@dataclass
class XTRTorchScriptConfig:
    batch_size: int = 1
    num_threads: int = 1

    @property
    def filename(self):
        return "xtr-base-en-torchscript.pt"

class XTRTorchScriptModel:
    def __init__(self, config: XTRTorchScriptConfig):
        print("#> Loading TorchScriptModel")
        ONNX_DIR = os.environ["TORCHSCRIPT_MODEL_DIR"]
        XTRTorchScriptModel._create_model_if_not_exists(ONNX_DIR, config)

        self.model = torch.jit.load(os.path.join(ONNX_DIR, config.filename))
        self.model.eval()

        self.tokenizer = XTRTokenizer(
            AutoTokenizer.from_pretrained("google/xtr-base-en")
        )

        assert config.num_threads == torch.torch.get_num_threads()

    @staticmethod
    def _create_model_if_not_exists(root_dir, config: XTRTorchScriptConfig):
        model_path = os.path.join(root_dir, config.filename)
        if not os.path.exists(model_path):
            print(f"#> Creating TorchScript XTR Model at '{root_dir}'")

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

            traced_model = torch.jit.trace(base_model, (input_ids, attention_mask))
            traced_model.save(model_path)

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        with torch.inference_mode():
            return self.model(input_ids, attention_mask)