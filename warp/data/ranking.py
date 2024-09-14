import os
import tqdm
import ujson
import sys
from dataclasses import dataclass

import pytrec_eval

from warp.infra.provenance import Provenance
from warp.infra.run import Run
from warp.utils.utils import print_message, groupby_first_item
from warp.data.queries import WARPQRels

from utility.utils.save_metadata import get_metadata_only

def numericize(v):
    if '.' in v:
        return float(v)

    return int(v)


def load_ranking(path):  # works with annotated and un-annotated ranked lists
    print_message("#> Loading the ranked lists from", path)

    with open(path) as f:
        return [list(map(numericize, line.strip().split('\t'))) for line in f]


class Ranking:
    def __init__(self, path=None, data=None, metrics=None, provenance=None):
        self.__provenance = provenance or path or Provenance()
        self.data = self._prepare_data(data or self._load_file(path))

    def apply_collection_map(self, collection_map):
        new_flat_ranking = []
        for query_id, passage_id, rank, score in self.flat_ranking:
            new_flat_ranking.append((query_id, collection_map[passage_id], rank, score))
        self.flat_ranking = new_flat_ranking

    def provenance(self):
        return self.__provenance
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def _prepare_data(self, data):
        # TODO: Handle list of lists???
        if isinstance(data, dict):
            self.flat_ranking = [(qid, *rest) for qid, subranking in data.items() for rest in subranking]
            return data

        self.flat_ranking = data
        return groupby_first_item(tqdm.tqdm(self.flat_ranking))

    def _load_file(self, path):
        return load_ranking(path)

    def todict(self):
        return dict(self.data)

    def tolist(self):
        return list(self.flat_ranking)

    def items(self):
        return self.data.items()

    def _load_tsv(self, path):
        raise NotImplementedError

    def _load_jsonl(self, path):
        raise NotImplementedError

    def save(self, new_path):
        assert 'tsv' in new_path.strip('/').split('/')[-1].split('.'), "TODO: Support .json[l] too."

        with Run().open(new_path, 'w') as f:
            for items in self.flat_ranking:
                line = '\t'.join(map(lambda x: str(int(x) if type(x) is bool else x), items)) + '\n'
                f.write(line)

            output_path = f.name
            print_message(f"#> Saved ranking of {len(self.data)} queries and {len(self.flat_ranking)} lines to {f.name}")
        
        with Run().open(f'{new_path}.meta', 'w') as f:
            d = {}
            d['metadata'] = get_metadata_only()
            d['provenance'] = self.provenance()
            line = ujson.dumps(d, indent=4)
            f.write(line)
        
        return output_path

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if isinstance(obj, dict) or isinstance(obj, list):
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


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
    return final_metrics

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