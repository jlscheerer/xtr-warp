# TODO(jlscheerer) Eventually rename this file/class.
import os
import json
from dataclasses import dataclass
from typing import Literal, Optional
import argparse
from tqdm import tqdm

from colbert import Searcher, Indexer
from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from colbert.utils.tracker import ExecutionTracker

# TODO(jlscheerer) Fix this again.
# from index_converter import convert_index

INDEX_ROOT = os.environ["INDEX_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]


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
        root="./",
    )


def index(config: XTRRunConfig):
    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        indexer = Indexer(
            checkpoint="google/xtr-base-en", config=to_colbert_config(config)
        )
        indexer.index(name=config.index_name, collection=config.collection_path)


def search(config: XTRRunConfig, batch_queries=True):
    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        searcher = Searcher(
            index=config.index_name,
            config=to_colbert_config(config),
            index_root=config.index_root,
        )
        queries = Queries(config.queries_path)

        if batch_queries:
            ranking = searcher.search_all(queries, k=config.k)
            collection_map_path = os.path.join(
                os.path.dirname(config.collection_path), "collection_map.json"
            )
            if os.path.exists(collection_map_path):
                with open(collection_map_path, "r") as file:
                    collection_map = json.load(file)
                    collection_map = {
                        int(key): int(value) for key, value in collection_map.items()
                    }
                print(
                    f"[WARNING] Applying collection_map found in {config.collection_path}"
                )
                ranking.apply_collection_map(collection_map)
            return ranking.save(
                f"{config.index_name}.split={config.datasplit}.ranking.k={config.k}.tsv"
            )
        else:
            tracker = ExecutionTracker(
                "XTR/PLAID [baseline]",
                [
                    "Query Encoding",
                    "Candidate Generation",
                    "Filtering",
                    "Decompress Residuals",
                    "Scoring",
                    "Sorting",
                ],
            )
            for query_id, query_text in tqdm(queries):
                tracker.next_iteration()
                ranking = searcher.search(query_text, k=100, tracker=tracker)
                tracker.end_iteration()
                collection_map_path = os.path.join(
                    os.path.dirname(config.collection_path), "collection_map.json"
                )
                if os.path.exists(collection_map_path):
                    with open(collection_map_path, "r") as file:
                        collection_map = json.load(file)
                        collection_map = {
                            int(key): int(value)
                            for key, value in collection_map.items()
                        }
                    # print(f"[WARNING] Applying collection_map found in {config.collection_path}")
                    ranking = (
                        [collection_map[x] for x in ranking[0]],
                        ranking[1],
                        ranking[2],
                    )
            return tracker
