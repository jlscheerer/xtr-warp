from collections import defaultdict
import jsonlines
import os
import sys

LOTTE_COLLECTION_PATH = "/lfs/1/scheerer/datasets/lotte/lotte"

def evaluate_dataset(rankings_path, dataset, split, query_type="search", k=5, data_rootdir=LOTTE_COLLECTION_PATH):
    provenance = {'query_type': query_type, 'dataset': dataset}
    data_path = os.path.join(data_rootdir, dataset, split)
    if not os.path.exists(rankings_path):
        return {'provenance': provenance, 'metrics': {'Success@5': None}}
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            qid, pid, rank = items[:3]
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            rankings[qid].append(pid)
            assert rank == len(rankings[qid])

    success = 0
    qas_path = os.path.join(data_path, f"qas.{query_type}.jsonl")

    num_total_qids = 0
    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            num_total_qids += 1
            if qid not in rankings:
                print(f"WARNING: qid {qid} not found in {rankings_path}!", file=sys.stderr)
                continue
            answer_pids = set(line["answer_pids"])
            if len(set(rankings[qid][:k]).intersection(answer_pids)) > 0:
                success += 1
    return {'provenance': provenance, 'metrics': {'Success@5': success / num_total_qids}}