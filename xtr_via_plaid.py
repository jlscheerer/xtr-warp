import os
import json

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

INDEX_ROOT = "/lfs/1/scheerer/indexes/ColBERT-XTR"
INDEX_NAME = "beir-scifact.nbits=2"

COLLECTION_PATH = "/lfs/1/scheerer/datasets/beir/datasets/scifact/collection.tsv"
QUERIES_PATH = "/lfs/1/scheerer/datasets/beir/datasets/scifact/questions.test.tsv"

EXPERIMENT_ROOT = "/lfs/1/scheerer/experiments/ColBERT-XTR"
EXPERIMENT_NAME = "beir-scifact"

def construct_config():
    return ColBERTConfig(
        nbits=2,
        doc_maxlen=DOC_MAXLEN,
        query_maxlen=QUERY_MAXLEN,
        index_path=f"{INDEX_ROOT}/{INDEX_NAME}",
        root="./"
    )

def index():
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT_NAME)):
        indexer = Indexer(checkpoint="google/xtr-base-en", config=construct_config())
        indexer.index(name=INDEX_NAME, collection=COLLECTION_PATH)

def search(batch_queries=True):
    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT_NAME)):
        searcher = Searcher(index=INDEX_NAME, config=construct_config(),
                            index_root=INDEX_ROOT)
        queries = Queries(QUERIES_PATH)

        if batch_queries:
            ranking = searcher.search_all(queries, k=5)
            collection_map_path = os.path.join(os.path.dirname(COLLECTION_PATH), "collection_map.json")
            if os.path.exists(collection_map_path):
                with open(collection_map_path, "r") as file:
                    collection_map = json.load(file)
                    collection_map = {int(key): int(value) for key, value in collection_map.items()}
                print(f"[WARNING] Applying collection_map found in {COLLECTION_PATH}")
                ranking.apply_collection_map(collection_map)

            ranking.save(f"{INDEX_NAME}.ranking.tsv")
        else:
            ranking = searcher.search(queries[1], k=100)

            collection_map_path = os.path.join(os.path.dirname(COLLECTION_PATH), "collection_map.json")
            if os.path.exists(collection_map_path):
                with open(collection_map_path, "r") as file:
                    collection_map = json.load(file)
                    collection_map = {int(key): int(value) for key, value in collection_map.items()}
                print(f"[WARNING] Applying collection_map found in {COLLECTION_PATH}")
                ranking = ([collection_map[x] for x in ranking[0]], ranking[1], ranking[2])

if __name__=='__main__':
    search()