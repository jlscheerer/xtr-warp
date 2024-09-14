import pytrec_eval
import sys

from dataclasses import dataclass

from warp.data import Ranking
from warp.infra.provenance import Provenance

from warp.engine.data.queries import WARPQRels

def _success_at_k_lotte(expected, rankings, k):
    num_total_qids, success = 0, 0
    for qid, answer_pids in expected.data.items():
        num_total_qids += 1
        if str(qid) not in rankings.data:
            print(f"WARNING: qid {qid} not found in {rankings}!", file=sys.stderr)
            continue
        qid_rankings = set(map(lambda x: x[0], rankings.data[str(qid)][:k]))
        if len(qid_rankings.intersection(answer_pids)) > 0:
            success += 1
    return success / num_total_qids


def _recall_at_k_lotte(expected, rankings, k):
    avg, num_relevant = 0, 0
    for qid, answer_pids in expected.data.items():
        if str(qid) not in rankings.data:
            print(f"WARNING: qid {qid} not found in {rankings}!", file=sys.stderr)
            continue
        relevant_count = len(answer_pids)
        if relevant_count == 0:
            continue
        num_relevant += 1
        qid_rankings = set(map(lambda x: x[0], rankings.data[str(qid)][:k]))
        correct_count = len(answer_pids & qid_rankings)
        avg += correct_count / relevant_count
    return avg / num_relevant

def eval_metrics_lotte(qas, rankings, k):
    K_VALUES = [5, 10, 100, 1000]
    final_metrics = dict()
    for k in K_VALUES:
        final_metrics[f"success@{k}"] = _success_at_k_lotte(expected=qas, rankings=rankings.ranking, k=k)
        final_metrics[f"recall@{k}"] = _recall_at_k_lotte(expected=qas, rankings=rankings.ranking, k=k)
    return {
        "metrics": final_metrics
    }

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

    def evaluate(self, qrels: WARPQRels, k: int):
        if qrels.config.collection == "lotte":
            return eval_metrics_lotte(qrels.qas, self, k=5)
        else:
            assert qrels.config.collection == "beir"
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
