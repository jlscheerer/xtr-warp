import os
import json

from tqdm import tqdm

from warp.data.queries import WARPQueries
from warp.data.ranking import WARPRanking
from warp.data.ranking import WARPRankingItem, WARPRankingItems

from warp.utils.tracker import NOPTracker
from warp.engine.config import WARPRunConfig

from warp.infra import Run, RunConfig
from warp import Searcher

class WARPSearcher:
    def __init__(self, config: WARPRunConfig):
        self.config = config
        with Run().context(
            RunConfig(nranks=config.nranks, experiment=config.experiment_name)
        ):
            self.searcher = Searcher(
                index=config.index_name,
                config=config,
                index_root=config.index_root,
                warp_engine=True,
                ablation_params=config.ablation_params,
            )

        collection_map_path = os.path.join(
            os.path.dirname(config.collection_path), "collection_map.json"
        )
        if os.path.exists(collection_map_path):
            with open(collection_map_path, "r") as file:
                collection_map = json.load(file)
                collection_map = {
                    int(key): value for key, value in collection_map.items()
                }
            print(f"#> Loading collection_map found in {config.collection_path}")
            self.collection_map = collection_map
        else:
            self.collection_map = None

    def search_all(self, queries, k=None, batched=True, tracker=NOPTracker(), show_progress=True):
        if batched and self.config.onnx is not None:
            print("[WARNING] Batched search_all not implemented for ONNX Configuration")
            print("[WARNING] Falling back to batched=False")
            batched = False
        if batched:
            return self._search_all_batched(queries, k, tracker, show_progress=show_progress)
        return self._search_all_unbatched(queries, k, tracker, show_progress=show_progress)

    def _search_all_batched(self, queries, k=None, tracker=NOPTracker(), show_progress=True):
        if k is None:
            k = self.config.k
        if isinstance(queries, WARPQueries):
            queries = queries.queries
        ranking = self.searcher.search_all(queries, k=k, tracker=tracker, show_progress=show_progress)
        if self.collection_map is not None:
            ranking.apply_collection_map(self.collection_map)
        return WARPRanking(ranking)

    def _search_all_unbatched(self, queries, k=None, tracker=NOPTracker(), show_progress=True):
        if k is None:
            k = self.config.k
        results = WARPRankingItems()
        for qid, qtext in tqdm(queries, disable=not show_progress):
            tracker.next_iteration()
            results += WARPRankingItem(
                qid=qid, results=self.search(qtext, k=k, tracker=tracker)
            )
            tracker.end_iteration()
        return results.finalize(
            self, queries.provenance(source="Searcher::search", k=k)
        )

    def search(self, query, k=None, tracker=NOPTracker()):
        if k is None:
            k = self.config.k
        return self.searcher.search(query, k=k, tracker=tracker)
