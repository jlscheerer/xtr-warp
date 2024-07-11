import os
from dotenv import load_dotenv

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

load_dotenv()

from colbert.warp.data.queries import WARPQueries
from colbert.warp.searcher import WARPSearcher
from colbert.warp.config import WARPRunConfig
from colbert.warp.onnx_model import (
    XTROnnxQuantization,
    XTROnnxConfig,
)

if __name__ == "__main__":
    onnx_config = XTROnnxConfig(quantization=XTROnnxQuantization.PREPROCESS)
    config = WARPRunConfig(
        nranks=4,
        type_="search",
        dataset="lotte",
        collection="writing",
        datasplit="test",
        nbits=4,
        onnx=onnx_config,
    )

    searcher = WARPSearcher(config)
    queries = WARPQueries(config)

    searcher.search_all(queries, k=5, batched=False)
