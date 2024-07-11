import os
from dotenv import load_dotenv

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

# Load ENVIRONMENT variables. Be sure to change the .env file!
load_dotenv()

import json

from colbert.warp.config import WARPRunConfig
from colbert.infra import Run, RunConfig
from colbert.data import Queries
from colbert import Searcher

from evaluation.evaluate_lotte_rankings import evaluate_dataset

if __name__ == "__main__":
    config = WARPRunConfig(
        nranks=4,
        dataset="lotte",
        collection="writing",
        type_="search",
        datasplit="test",
        nbits=4,
        k=100,
    )

    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        searcher = Searcher(
            index=config.index_name,
            config=config.colbert(),
            index_root=config.index_root,
            warp_engine=True,
        )
        queries = Queries(config.queries_path)

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
        print(f"[WARNING] Applying collection_map found in {config.collection_path}")
        ranking.apply_collection_map(collection_map)
    rankings_path = ranking.save(
        f"{config.index_name}.split={config.datasplit}.ranking.k={config.k}.tsv"
    )

    print(
        evaluate_dataset(rankings_path, config.collection, split=config.datasplit, k=5)[
            "metrics"
        ]["Success@5"]
    )

    os.remove(rankings_path)
    os.remove(f"{rankings_path}.meta")
