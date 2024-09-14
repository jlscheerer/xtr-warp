import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from colbert.infra import ColBERTConfig

USE_CORE_ML = False

from colbert.warp.onnx_model import XTROnnxConfig

if USE_CORE_ML:
    from colbert.warp.coreml_model import XTRCoreMLConfig
    OptimConfig = Union[XTROnnxConfig, XTRCoreMLConfig]
else:
    OptimConfig = XTROnnxConfig

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
    nranks: int = 1

    optim: Optional[OptimConfig] = None

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
