import os
from dataclasses import dataclass
from typing import Literal, Optional

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from colbert.infra import ColBERTConfig

from colbert.warp.onnx_model import XTROnnxConfig

INDEX_ROOT = os.environ["INDEX_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]


@dataclass
class WARPRunConfig:
    nranks: int
    dataset: Literal["beir", "lotte"]
    collection: str
    datasplit: Literal["train", "dev", "test"]
    nbits: int

    type_: Optional[Literal["search", "forum"]] = None
    k: int = 100

    onnx: Optional[XTROnnxConfig] = None

    @property
    def index_root(self):
        return INDEX_ROOT

    @property
    def index_name(self):
        return f"{self.dataset}-{self.collection}.split={self.datasplit}.nbits={self.nbits}"

    @property
    def collection_path(self):
        if self.dataset == "beir":
            return f"{BEIR_COLLECTION_PATH}/{self.collection}/collection.tsv"
        elif self.dataset == "lotte":
            return f"{LOTTE_COLLECTION_PATH}/{self.collection}/{self.datasplit}/collection.tsv"
        raise AssertionError

    @property
    def queries_path(self):
        if self.dataset == "beir":
            return f"{BEIR_COLLECTION_PATH}/{self.collection}/questions.{self.datasplit}.tsv"
        elif self.dataset == "lotte":
            return f"{LOTTE_COLLECTION_PATH}/{self.collection}/{self.datasplit}/questions.{self.type_}.tsv"
        raise AssertionError

    @property
    def experiment_root(self):
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
