import os
import json

from tqdm import tqdm

from colbert.warp.config import WARPRunConfig
from colbert.warp.data.queries import WARPQueries
from colbert.warp.data.ranking import WARPRanking
from colbert.infra import Run, RunConfig
from colbert import Searcher

from colbert.warp.data.ranking import WARPRankingItem, WARPRankingItems


class WARPSearcher:
    def __init__(self, config: WARPRunConfig):
        self.config = config
        with Run().context(
            RunConfig(nranks=config.nranks, experiment=config.experiment_name)
        ):
            self.searcher = Searcher(
                index=config.index_name,
                config=config.colbert(),
                index_root=config.index_root,
                warp_engine=True,
            )

        collection_map_path = os.path.join(
            os.path.dirname(config.collection_path), "collection_map.json"
        )
        if os.path.exists(collection_map_path):
            with open(collection_map_path, "r") as file:
                collection_map = json.load(file)
                collection_map = {
                    int(key): int(value) for key, value in collection_map.items()
                }
            print(f"#> Loading collection_map found in {config.collection_path}")
            self.collection_map = collection_map
        else:
            self.collection_map = None

    def search_all(self, queries, k=None, batched=True):
        if batched:
            return self._search_all_batched(queries, k)
        return self._search_all_unbatched(queries, k)

    def _search_all_batched(self, queries, k=None):
        if k is None:
            k = self.config.k
        if isinstance(queries, WARPQueries):
            queries = queries.queries
        ranking = self.searcher.search_all(queries, k=k)
        if self.collection_map is not None:
            ranking.apply_collection_map(self.collection_map)
        return WARPRanking(ranking)

    def _search_all_unbatched(self, queries, k=None):
        if k is None:
            k = self.config.k
        results = WARPRankingItems()
        for qid, qtext in tqdm(queries):
            results += WARPRankingItem(qid=qid, results=self.search(qtext))
        return results.finalize(
            self, queries.provenance(source="Searcher::search", k=k)
        )

    def search(self, query, k=None):
        if k is None:
            k = self.config.k
        return self.searcher.search(query, k=k)
