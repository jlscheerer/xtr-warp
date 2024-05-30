import os
import torch
from xtr_via_plaid import search, XTRRunConfig
from colbert.modeling.colbert import ColBERT
import itertools
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def setup_num_threads(num_threads):
    torch.set_num_threads(num_threads)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

def make_config(collection, datasplit, nbits, k):
    return XTRRunConfig(nranks=4, dataset="lotte", collection=collection,
                    type_="search", datasplit=datasplit, nbits=nbits, k=k)

if __name__ == "__main__":
    COLLECTIONS = ["lifestyle", "pooled", "recreation", "science", "technology", "writing"]
    DATASPLITS = ["dev", "test"]
    NBITS = [1, 2, 4]
    K = [100]
    THREADS = [1, 4, 8, 16, 24]

    ColBERT.try_load_torch_extensions(use_gpu=False)

    root_dir = "latency/xtr"
    os.makedirs(root_dir, exist_ok=True)

    for collection, split, nbits, k, threads in itertools.product(COLLECTIONS, DATASPLITS, NBITS, K, THREADS):
        filename = f"{collection}_split={split}_nbits={nbits}_k={k}_threads={threads}.json"
        if os.path.exists(os.path.join(root_dir, filename)):
            print(f"Skipping: {collection} {split} {nbits} {k} {threads}")
            continue
        setup_num_threads(threads)
        config = make_config(collection=collection, datasplit=split, nbits=nbits, k=k)
        tracker = search(config, batch_queries=False)

        result = {'config': {'collection': collection, 'split': split, 'nbits': nbits, 'k': k}}
        result['tracker'] = tracker.as_dict()
        # Note "hacky" way to change the name of the tracker
        result['tracker']['name'] = f"XTR/PLAID [baseline, {collection} ({split}, nbits={nbits}), k={k}, threads={threads}]"

        with open(os.path.join(root_dir, filename), "w") as file:
            file.write(json.dumps(result))