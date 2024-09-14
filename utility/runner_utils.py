import sys
import re

from warp.engine.config import WARPRunConfig

from warp.engine.runtime.onnx_model import XTROnnxQuantization, XTROnnxConfig
from warp.engine.runtime.openvino_model import XTROpenVinoConfig
from warp.engine.config import USE_CORE_ML
if USE_CORE_ML:
    from warp.engine.runtime.coreml_model import XTRCoreMLConfig

DEFAULT_K_VALUE = 1000
QUANTIZATION_TYPES = "|".join(["NONE", "PREPROCESS", "DYN_QUANTIZED_QINT8", "QUANTIZED_QATTENTION"])

def _make_runtime(runtime):
    if runtime is None:
        return runtime
    
    match = re.match(f"ONNX\\.({QUANTIZATION_TYPES})", runtime)
    if match is not None:
        quantization = XTROnnxQuantization[match[1]]
        return XTROnnxConfig(quantization=quantization)

    match = re.match(f"OpenVINO\\.({QUANTIZATION_TYPES})", runtime)
    if match is not None:
        quantization = XTROnnxQuantization[match[1]]
        return XTROpenVinoConfig(quantization=quantization)

    if USE_CORE_ML and runtime == "CoreML":
        return XTRCoreMLConfig()

    assert False

def make_run_config(config):
    collection, dataset, split = config["collection"], config["dataset"], config["split"]
    nbits, nprobe, t_prime, bound = config["nbits"], config["nprobe"], config["t_prime"], config["bound"]
    k = config["document_top_k"] or DEFAULT_K_VALUE

    return WARPRunConfig(
        collection=collection,
        dataset=dataset,
        type_="search" if collection == "lotte" else None,
        datasplit=split,
        nbits=nbits,
        nprobe=nprobe,
        t_prime=t_prime,
        k=k,
        runtime=_make_runtime(config["runtime"]),
        bound=bound
    )