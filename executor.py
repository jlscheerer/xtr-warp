import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(1)

import colbert.warp.setup

from colbert.warp.config import WARPRunConfig
from colbert.warp.searcher import WARPSearcher
from colbert.warp.data.queries import WARPQueries

import argparse
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from tqdm import tqdm
import os
import sys
import io
import json

BEIR_DATASETS = ["nfcorpus", "scifact", "scidocs", "fiqa", "webis-touche2020", "quora"]
LOTTE_DATASETS = ["lifestyle", "writing", "recreation", "technology", "science", "pooled"]

def make_config(collection, dataset, nbits, nprobe, t_prime, bound=None, split="test"):
    assert collection in ["beir", "lotte"]
    if collection == "beir":
        assert dataset in BEIR_DATASETS
    else:
        assert dataset in LOTTE_DATASETS
    assert nbits in [2, 4]
    assert nprobe is None or isinstance(nprobe, int)
    assert t_prime is None or isinstance(t_prime, int)
    assert bound is None or isinstance(bound, int)
    assert split in ["dev", "test"]
    return {
        "collection": collection,
        "dataset": dataset,
        "nbits": nbits,
        "nprobe": nprobe,
        "t_prime": t_prime,
        "bound": bound,
        "split": split
    }

def expand_configs(datasets, nbits, nprobes, t_primes, bound=None, split="test"):
    if not isinstance(nbits, list):
        nbits = [nbits]
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(nprobes, list):
        nprobes = [nprobes]
    if not isinstance(t_primes, list):
        t_primes = [t_primes]
    configs = []
    for collection_dataset in datasets:
        collection, dataset = collection_dataset.split(".")
        for nbit in nbits:
            for nprobe in nprobes:
                for t_prime in t_primes:
                    configs.append(make_config(collection=collection, dataset=dataset, nbits=nbit,
                                               nprobe=nprobe, t_prime=t_prime, bound=bound, split=split))
    return configs

def execute_config(config):
    collection, dataset, split, nbits = config["collection"], config["dataset"], config["split"], config["nbits"]
    nprobe, t_prime, bound = config["nprobe"], config["t_prime"], config["bound"]

    # Configure WARP to use specified dataset & the unquantized model
    optim = None
    warp_config = WARPRunConfig(
        nranks=4,
        dataset=collection,
        collection=dataset,
        datasplit=split,
        type_="search" if collection == "lotte" else None,
        nbits=nbits,
        optim=optim,
    )
    searcher = WARPSearcher(warp_config)
    if bound is not None:
        searcher.searcher.ranker.bound = bound
    if nprobe is not None:
        searcher.searcher.ranker.nprobe = nprobe
    if t_prime is not None:
        searcher.searcher.ranker.t_prime = t_prime
    
    queries = WARPQueries(warp_config)
    
    if collection == "beir":
        rankings = searcher.search_all(queries, k=100, batched=False, show_progress=True)
        metrics = rankings.evaluate(queries.qrels, k=100)
    elif collection == "lotte":
        rankings = searcher.search_all(queries, k=5, batched=False, show_progress=True)
        metrics = rankings.evaluate(queries.qrels, k=5)["metrics"]
    else: assert False

    statistics = {
        "centroids": searcher.searcher.ranker.centroids.shape[0],
        "embeddings": searcher.searcher.ranker.residuals_repacked_compacted_df.shape[0],
        "median_size": searcher.searcher.ranker.sizes_compacted.median().item()
    }

    return {
        "metrics": metrics,
        "statistics": statistics,
        "provenance": config
    }

def init_proc(env_vars):
    for key, value in env_vars.items():
        os.environ[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='XTR/WARP Experiment [Executor/Platform]')
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers or psutil.cpu_count(logical=False)
    OVERWRITE = args.overwrite

    print(OVERWRITE)

    config_file = args.config
    with open(config_file, "r") as file:
        config_data = json.loads(file.read())

    OUTPUT_FILE = f"experiments/results/{config_data['name']}.json"
    DATASETS = config_data["datasets"]
    NBITS_VALUES = config_data["nbits"]
    NPROBE_VALUES = config_data["nprobe"]
    T_PRIME_VALUES = config_data["t_prime"]
    BOUND = config_data["bound"]
    DATASPLIT = config_data["datasplit"]

    if not OVERWRITE:
        assert not os.path.exists(OUTPUT_FILE)
    configs = expand_configs(
        datasets=DATASETS,
        nbits=NBITS_VALUES,
        nprobes=NPROBE_VALUES,
        t_primes=T_PRIME_VALUES,
        bound=BOUND,
        split=DATASPLIT
    )
    print("#> Running:")
    print(configs)
    results = []

    env_vars = dict(os.environ)
    progress = tqdm(total=len(configs))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_proc, initargs=(env_vars,)) as executor, redirect_stdout(
        io.StringIO()
    ) as redirect_stdout:
        futures = [executor.submit(execute_config, config) for config in configs]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            with open(OUTPUT_FILE, "w") as file:
                file.write(json.dumps(results, indent=3))
            
            sys.stdout = sys.__stdout__
            sys.stdout = redirect_stdout
            progress.update(1)
    progress.close()
    print("#> Done")