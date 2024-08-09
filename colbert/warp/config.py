import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from colbert.infra import ColBERTConfig

from colbert.warp.onnx_model import XTROnnxConfig
from colbert.warp.coreml_model import XTRCoreMLConfig


@dataclass
class WARPRunConfig:
    nranks: int
    dataset: Literal["beir", "lotte"]
    collection: str
    datasplit: Literal["train", "dev", "test"]
    nbits: int

    type_: Optional[Literal["search", "forum"]] = None
    k: int = 100

    optim: Optional[Union[XTROnnxConfig, XTRCoreMLConfig]] = None

    @property
    def index_root(self):
        INDEX_ROOT = os.environ["INDEX_ROOT"]
        return INDEX_ROOT

    @property
    def index_name(self):
        return f"{self.dataset}-{self.collection}.split={self.datasplit}.nbits={self.nbits}"

    @property
    def collection_path(self):
        BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
        LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]
        if self.dataset == "beir":
            return f"{BEIR_COLLECTION_PATH}/{self.collection}/collection.tsv"
        elif self.dataset == "lotte":
            return f"{LOTTE_COLLECTION_PATH}/{self.collection}/{self.datasplit}/collection.tsv"
        raise AssertionError

    @property
    def queries_path(self):
        BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
        LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]
        if self.dataset == "beir":
            return f"{BEIR_COLLECTION_PATH}/{self.collection}/questions.{self.datasplit}.tsv"
        elif self.dataset == "lotte":
            return f"{LOTTE_COLLECTION_PATH}/{self.collection}/{self.datasplit}/questions.{self.type_}.tsv"
        raise AssertionError

    @property
    def experiment_root(self):
        EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
        return EXPERIMENT_ROOT

    @property
    def experiment_name(self):
        return f"{self.dataset}-{self.collection}"

    def colbert(self):
        return ColBERTConfig(
            nbits=self.nbits,
            doc_maxlen=DOC_MAXLEN,
            query_maxlen=QUERY_MAXLEN,
            index_path=f"{self.index_root}/{self.index_name}",
            root="./",
        )
