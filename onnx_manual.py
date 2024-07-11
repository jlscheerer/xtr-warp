from dotenv import load_dotenv
from colbert.utilities.warp.onnx_export import (
    XTROnnxQuantization,
    XTROnnxModel,
    XTROnnxConfig,
)

if __name__ == "__main__":
    load_dotenv()
    config = XTROnnxConfig(quantization=XTROnnxQuantization.DYN_QUANTIZED_QINT8)
    model = XTROnnxModel(config)
