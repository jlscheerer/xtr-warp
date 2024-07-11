import os
import io

from contextlib import redirect_stdout
from dataclasses import dataclass

from colbert.data import Ranking
from colbert.infra.provenance import Provenance
from colbert.utilities.evaluation.evaluate_lotte_rankings import evaluate_dataset

from colbert.warp.data.queries import WARPQRels


class WARPRanking:
    def __init__(self, ranking: Ranking):
        self.ranking = ranking

    def evaluate(self, qrels: WARPQRels, k: int, cleanup: bool = True):
        _ = io.StringIO()
        with redirect_stdout(_):
            rankings_path = self.ranking.save(
                f"{qrels.config.index_name}.split={qrels.config.datasplit}.ranking.k={k}.tsv"
            )
            assert qrels.config.dataset == "lotte"
            results = evaluate_dataset(
                rankings_path,
                qrels.config.collection,
                split=qrels.config.datasplit,
                k=k,
            )
        del _
        if cleanup:
            os.remove(rankings_path)
            os.remove(f"{rankings_path}.meta")
        return results


@dataclass
class WARPRankingItem:
    qid: int
    results: list


class WARPRankingItems:
    def __init__(self):
        self.data = dict()

    def __iadd__(self, item):
        assert isinstance(item, WARPRankingItem)
        assert item.qid not in self.data
        self.data[item.qid] = list(zip(*item.results))
        return self

    def finalize(self, searcher, provenance: Provenance):
        ranking = Ranking(data=self.data, provenance=provenance)
        if searcher.collection_map is not None:
            ranking.apply_collection_map(searcher.collection_map)
        return WARPRanking(ranking)
