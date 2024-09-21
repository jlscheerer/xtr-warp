import sys
import re

from warp.engine.config import WARPRunConfig
from warp.engine.config import USE_CORE_ML

from warp.engine.runtime.torchscript_model import XTRTorchScriptConfig, XTRTorchScriptModel
from warp.engine.runtime.onnx_model import XTROnnxQuantization, XTROnnxConfig, XTROnnxModel
from warp.engine.runtime.openvino_model import XTROpenVinoConfig, XTROpenVinoModel
if USE_CORE_ML:
    from warp.engine.runtime.coreml_model import XTRCoreMLConfig, XTRCoreMLModel

DEFAULT_K_VALUE = 1000
QUANTIZATION_TYPES = "|".join(["NONE", "PREPROCESS", "DYN_QUANTIZED_QINT8", "QUANTIZED_QATTENTION"])

def _make_runtime(runtime, num_threads=1):
    if runtime is None:
        return runtime
    
    if runtime == "TORCHSCRIPT":
        return XTRTorchScriptConfig(num_threads=num_threads)

    match = re.match(f"ONNX\\.({QUANTIZATION_TYPES})", runtime)
    if match is not None:
        quantization = XTROnnxQuantization[match[1]]
        return XTROnnxConfig(quantization=quantization, num_threads=num_threads)

    if runtime == "OPENVINO":
        return XTROpenVinoConfig(num_threads=num_threads)

    if USE_CORE_ML and runtime == "CORE_ML":
        return XTRCoreMLConfig(num_threads=num_threads)

    assert False

def make_run_config(config):
    collection, dataset, split = config["collection"], config["dataset"], config["split"]
    nbits, nprobe, t_prime, bound = config["nbits"], config["nprobe"], config["t_prime"], config["bound"]
    k = config["document_top_k"] or DEFAULT_K_VALUE

    num_threads = config["num_threads"]
    fused_ext = True
    if num_threads != 1:
        fused_ext = config["fused_ext"]
    return WARPRunConfig(
        collection=collection,
        dataset=dataset,
        type_="search" if collection == "lotte" else None,
        datasplit=split,
        nbits=nbits,
        nprobe=nprobe,
        t_prime=t_prime,
        k=k,
        runtime=_make_runtime(config["runtime"], num_threads=num_threads),
        bound=bound,
        fused_ext=fused_ext
    )