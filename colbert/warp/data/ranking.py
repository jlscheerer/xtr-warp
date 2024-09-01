import os
import io
import pytrec_eval

from contextlib import redirect_stdout
from dataclasses import dataclass

from colbert.data import Ranking
from colbert.infra.provenance import Provenance
from colbert.utilities.evaluation.evaluate_lotte_rankings import evaluate_dataset

from colbert.warp.data.queries import WARPQRels

def eval_metrics_beir(qrels, rankings):
    K_VALUES = [5, 10, 50, 100]
    METRIC_NAMES = ['ndcg_cut', 'map_cut', 'recall']

    measurements = []
    for metric_name in METRIC_NAMES:
        measurements.append(
            f"{metric_name}." + ",".join([str(k) for k in K_VALUES])
        )
    evaluator = pytrec_eval.RelevanceEvaluator(qrels.qrels, measurements)

    flat_rankings = rankings.ranking.flat_ranking
    dict_rankings = dict()
    for qidx, docid, _, score in flat_rankings:
        if str(qidx) not in dict_rankings:
            dict_rankings[str(qidx)] = dict()
        dict_rankings[str(qidx)][str(docid)] = score
    final_scores = evaluator.evaluate(dict_rankings)

    final_metrics = dict()
    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = 0.0

    for query_id in final_scores.keys():
        for metric_name in METRIC_NAMES:
            for k in K_VALUES:
                final_metrics[f"{metric_name}@{k}"] += final_scores[query_id][
                    f"{metric_name}_{k}"
                ]

    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = round(
                final_metrics[f"{metric_name}@{k}"] / len(final_scores), 5
            )

    return final_metrics

class WARPRanking:
    def __init__(self, ranking: Ranking):
        self.ranking = ranking

    def evaluate(self, qrels: WARPQRels, k: int, cleanup: bool = True):
        if qrels.config.dataset == "lotte":
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
        else:
            assert qrels.config.dataset == "beir"
            return eval_metrics_beir(qrels, self)


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
