import os
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

import coremltools as ct

from colbert.modeling.xtr import QUERY_MAXLEN, build_xtr_model, XTRTokenizer


@dataclass
class XTRCoreMLConfig:
    batch_size: int = 1

    @property
    def base_name(self):
        return "xtr"

    @property
    def base_filename(self):
        return f"{self.base_name}.mlpackage"

    @property
    def filename(self):
        return self.base_filename


class XTRCoreMLModel:
    def __init__(self, config: XTRCoreMLConfig):
        COREML_DIR = os.environ["COREML_MODEL_DIR"]
        XTRCoreMLModel._create_model_if_not_exists(COREML_DIR, config)

        model_path = os.path.join(COREML_DIR, config.filename)

        root_directory = Path(model_path)
        filesize = sum(
            f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
        ) / (1024 * 1024)

        print(
            f"#> Loading XTR CoreML model from '{model_path}' ({round(filesize, 2)}MB)"
        )
        self.model = ct.models.MLModel(model_path)

        self.tokenizer = XTRTokenizer(
            AutoTokenizer.from_pretrained("google/xtr-base-en")
        )

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        output = self.model.predict(
            {
                "input_ids": input_ids.numpy().astype(np.int32),
                "attention_mask": attention_mask.numpy().astype(np.int32),
            }
        )
        return torch.from_numpy(next(iter(output.values())))

    @staticmethod
    def _create_model_if_not_exists(root_dir, config: XTRCoreMLConfig):
        base_model_path = os.path.join(root_dir, config.base_filename)
        if not os.path.exists(base_model_path):
            print(f"#> Exporting XTR base model to .mlpackage at '{root_dir}'")
            os.makedirs(root_dir, exist_ok=True)

            base_model = build_xtr_model()
            base_model.eval()

            input_dim = (config.batch_size, QUERY_MAXLEN)
            device = torch.device("cpu")
            input_ids = torch.randint(
                low=1, high=1000, size=input_dim, dtype=torch.int32
            ).to(device)
            attention_mask = torch.randint(
                low=0, high=1, size=input_dim, dtype=torch.int32
            ).to(device)

            traced_model = torch.jit.trace(base_model, (input_ids, attention_mask))
            traced_model(input_ids, attention_mask)

            ct_input_ids_input = ct.TensorType(
                shape=ct.Shape(shape=(1, 32)), dtype=np.int32
            )
            ct_attention_mask_input = ct.TensorType(
                shape=ct.Shape(shape=(1, 32)), dtype=np.int32
            )

            ct_model = ct.convert(
                traced_model,
                convert_to="mlprogram",
                inputs=[ct_input_ids_input, ct_attention_mask_input],
                compute_precision=ct.precision.FLOAT32,
            )

            ct_model.save(base_model_path)
