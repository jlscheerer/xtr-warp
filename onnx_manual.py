from dotenv import load_dotenv

load_dotenv()

from colbert.modeling.xtr import XTRCheckpoint
from colbert.warp.config import WARPRunConfig
from colbert.warp.onnx_model import (
    XTROnnxQuantization,
    XTROnnxModel,
    XTROnnxConfig,
)

if __name__ == "__main__":
    config = XTROnnxConfig(quantization=XTROnnxQuantization.DYN_QUANTIZED_QINT8)
    model = XTROnnxModel(config)

    run_config = WARPRunConfig(
        nranks=4, dataset="lotte", collection="writing", datasplit="test", nbits=4
    )
    checkpoint = XTRCheckpoint(model, run_config.colbert())
    print(checkpoint.queryFromText(["hello world"]))
