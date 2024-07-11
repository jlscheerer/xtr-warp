from collections import defaultdict
from enum import Enum
import argparse
import csv
import os

import pytrec_eval
import pandas as pd

K_VALUES = [5, 10, 50, 100]
METRIC_NAMES = ["ndcg_cut", "map_cut", "recall"]


# Source: https://github.com/google-deepmind/xtr
def eval_metrics(qrels, predictions):
    measurements = []
    for metric_name in METRIC_NAMES:
        measurements.append(f"{metric_name}." + ",".join([str(k) for k in K_VALUES]))
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measurements)
    final_scores = evaluator.evaluate(predictions)

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

    print("[Result]")
    for metric_name, metric_score in final_metrics.items():
        print(f"{metric_name}: {metric_score:.4f}")


class BEIR(Enum):
    SCIFACT = 1
    NFCORPUS = 2


def construct_dataset_path(queries_path, dataset, split):
    return os.path.join(queries_path, "beir", dataset, split)


def evaluate_beir_rankings(datapath, split, rankings):
    qrels_path = os.path.join(datapath, "qrels", f"{split}.tsv")
    qrels = {}
    with open(qrels_path, "r") as file:
        qrels_tsv = csv.reader(file, delimiter="\t")
        next(qrels_tsv, None) # skip the header
        for line in qrels_tsv:
            qid, did, judgement = line
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(judgement)

    rankings = pd.read_csv(
        rankings,
        sep="\t",
        names=["query_id", "passage_id", "rank", "score"],
        dtype={"query_id": int, "passage_id": int, "rank": int, "score": float},
        header=None,
    )

    results = defaultdict(dict)
    for _, row in rankings.iterrows():
        query_id = int(row["query_id"])
        passage_id = int(row["passage_id"])
        score = row["score"]

        results[str(query_id)][str(passage_id)] = score
    results = dict(results)

    eval_metrics(qrels, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_beir_rankings")
    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-s", "--split", required=True)
    parser.add_argument("-r", "--rankings", required=True)
    args = parser.parse_args()
    evaluate_beir_rankings(args.data, args.split, args.rankings)
