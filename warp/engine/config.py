import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

from warp.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from warp.infra import ColBERTConfig

USE_CORE_ML = False

from warp.engine.runtime.onnx_model import XTROnnxConfig

if USE_CORE_ML:
    from warp.engine.runtime.coreml_model import XTRCoreMLConfig
    RuntimeConfig = Union[XTROnnxConfig, XTRCoreMLConfig]
else:
    RuntimeConfig = XTROnnxConfig

@dataclass
class WARPRunConfig:
    nbits: int
    
    collection: Literal["beir", "lotte"]
    dataset: str
    datasplit: Literal["train", "dev", "test"]
    type_: Optional[Literal["search", "forum"]] = None

    k: int = 100
    nprobe: int = 16
    t_prime: Optional[int] = None

    # Use the fused decompression + merge candidate scores extension.
    # NOTE This option is only applicable with num_threads != 1
    fused_ext: bool = True

    # NOTE To be more efficient, we could also derive this from the dataset.
    #      For now we just set it to a sufficiently high constant value.
    bound: int = 128
    
    nranks: int = 1

    # runtime == None uses "default" PyTorch for inference.
    runtime: Optional[RuntimeConfig] = None

    ablation_params: Optional[dict] = None

    @property
    def index_root(self):
        INDEX_ROOT = os.environ["INDEX_ROOT"]
        return INDEX_ROOT

    @property
    def index_name(self):
        return f"{self.collection}-{self.dataset}.split={self.datasplit}.nbits={self.nbits}"

    @property
    def collection_path(self):
        BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
        LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]
        if self.collection == "beir":
            return f"{BEIR_COLLECTION_PATH}/{self.dataset}/collection.tsv"
        elif self.collection == "lotte":
            return f"{LOTTE_COLLECTION_PATH}/{self.dataset}/{self.datasplit}/collection.tsv"
        raise AssertionError

    @property
    def queries_path(self):
        BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
        LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]
        if self.collection == "beir":
            return f"{BEIR_COLLECTION_PATH}/{self.dataset}/questions.{self.datasplit}.tsv"
        elif self.collection == "lotte":
            return f"{LOTTE_COLLECTION_PATH}/{self.dataset}/{self.datasplit}/questions.{self.type_}.tsv"
        raise AssertionError

    @property
    def experiment_root(self):
        EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
        return EXPERIMENT_ROOT

    @property
    def experiment_name(self):
        return f"{self.collection}-{self.dataset}"

    def colbert(self):
        return ColBERTConfig(
            nbits=self.nbits,
            ncells=self.nprobe,
            doc_maxlen=DOC_MAXLEN,
            query_maxlen=QUERY_MAXLEN,
            index_path=f"{self.index_root}/{self.index_name}",
            root="./",
        )
