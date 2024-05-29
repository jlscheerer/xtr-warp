import os
import json
from dataclasses import dataclass
from typing import Literal, Optional
import argparse

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

INDEX_ROOT = "/future/u/scheerer/home/data/indexes/ColBERT-XTR"
EXPERIMENT_ROOT = "/lfs/1/scheerer/experiments/ColBERT-XTR"

BEIR_COLLECTION_PATH = "/lfs/1/scheerer/datasets/beir/datasets"
LOTTE_COLLECTION_PATH = "/lfs/1/scheerer/datasets/lotte/lotte"

@dataclass
class XTRRunConfig:
    nranks: int
    dataset: Literal["beir", "lotte"]
    collection: str
    datasplit: Literal["train", "dev", "test"]
    nbits: int

    type_: Optional[Literal["search", "forum"]] = None
    k: int = 100

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

def to_colbert_config(config: XTRRunConfig):
    return ColBERTConfig(
        nbits=config.nbits,
        doc_maxlen=DOC_MAXLEN,
        query_maxlen=QUERY_MAXLEN,
        index_path=f"{config.index_root}/{config.index_name}",
        root="./"
    )

def index(config: XTRRunConfig):
    with Run().context(RunConfig(nranks=config.nranks, experiment=config.experiment_name)):
        indexer = Indexer(checkpoint="google/xtr-base-en", config=to_colbert_config(config))
        indexer.index(name=config.index_name, collection=config.collection_path)

def search(config: XTRRunConfig, batch_queries=True):
    with Run().context(RunConfig(nranks=config.nranks, experiment=config.experiment_name)):
        searcher = Searcher(index=config.index_name, config=to_colbert_config(config),
                            index_root=config.index_root)
        queries = Queries(config.queries_path)

        if batch_queries:
            ranking = searcher.search_all(queries, k=config.k)
            collection_map_path = os.path.join(os.path.dirname(config.collection_path), "collection_map.json")
            if os.path.exists(collection_map_path):
                with open(collection_map_path, "r") as file:
                    collection_map = json.load(file)
                    collection_map = {int(key): int(value) for key, value in collection_map.items()}
                print(f"[WARNING] Applying collection_map found in {config.collection_path}")
                ranking.apply_collection_map(collection_map)
            return ranking.save(f"{config.index_name}.split={config.datasplit}.ranking.k={config.k}.tsv")
        else:
            ranking = searcher.search(queries[1], k=100)
            collection_map_path = os.path.join(os.path.dirname(config.collection_path), "collection_map.json")
            if os.path.exists(collection_map_path):
                with open(collection_map_path, "r") as file:
                    collection_map = json.load(file)
                    collection_map = {int(key): int(value) for key, value in collection_map.items()}
                print(f"[WARNING] Applying collection_map found in {config.collection_path}")
                ranking = ([collection_map[x] for x in ranking[0]], ranking[1], ranking[2])
            return ranking

if __name__=='__main__':
    parser = argparse.ArgumentParser("xtr_via_plaid")
    parser.add_argument("-c", "--collection", required=True)
    parser.add_argument("-s", "--split", required=True)
    parser.add_argument("-n", "--nbits", type=int, required=True)

    args = parser.parse_args()

    config = XTRRunConfig(nranks=4, dataset="lotte", collection=args.collection,
                        type_="search", datasplit=args.split, nbits=args.nbits, k=100)
    index(config)