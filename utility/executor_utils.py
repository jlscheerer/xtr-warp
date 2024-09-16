import os
import json
import os
import io
import sys
import copy
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout


BEIR_DATASETS = ["nfcorpus", "scifact", "scidocs", "fiqa", "webis-touche2020", "quora"]
LOTTE_DATASETS = ["lifestyle", "writing", "recreation", "technology", "science", "pooled"]

def _make_config(collection, dataset, nbits, nprobe, t_prime, document_top_k=None, runtime=None, split="test", bound=None, num_threads=1, fused_ext=None):
    assert collection in ["beir", "lotte"]
    if collection == "beir":
        assert dataset in BEIR_DATASETS
    else:
        assert dataset in LOTTE_DATASETS
    assert nbits in [2, 4]
    assert nprobe is None or isinstance(nprobe, int)
    assert t_prime is None or isinstance(t_prime, int)
    assert document_top_k is None or isinstance(document_top_k, int)
    assert split in ["dev", "test"]
    assert bound is None or isinstance(bound, int)
    if fused_ext is None:
        fused_ext = True
    config = {
        "collection": collection,
        "dataset": dataset,
        "nbits": nbits,
        "nprobe": nprobe,
        "t_prime": t_prime,
        "document_top_k": document_top_k,
        "runtime": runtime,
        "split": split,
        "bound": bound,
        "num_threads": num_threads
    }
    if num_threads != 1:
        config["fused_ext"] = fused_ext
    return config

def _expand_configs(datasets, nbits, nprobes, t_primes, document_top_ks=None, runtimes=None, fused_exts=None, split="test", bound=None, num_threads=None):
    if num_threads is None:
        num_threads = 1
    if not isinstance(nbits, list):
        nbits = [nbits]
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(nprobes, list):
        nprobes = [nprobes]
    if not isinstance(t_primes, list):
        t_primes = [t_primes]
    if not isinstance(document_top_ks, list):
        document_top_ks = [document_top_ks]
    if not isinstance(runtimes, list):
        runtimes = [runtimes]
    if not isinstance(num_threads, list):
        num_threads = [num_threads]
    if not isinstance(fused_exts, list):
        fused_exts = [fused_exts]
    configs = []
    for collection_dataset in datasets:
        collection, dataset = collection_dataset.split(".")
        for nbit in nbits:
            for nprobe in nprobes:
                for t_prime in t_primes:
                    for document_top_k in document_top_ks:
                        for runtime in runtimes:
                            for threads in num_threads:
                                for fused_ext in fused_exts:
                                    configs.append(_make_config(collection=collection, dataset=dataset, nbits=nbit, nprobe=nprobe, t_prime=t_prime,
                                                                document_top_k=document_top_k, runtime=runtime, split=split, bound=bound, num_threads=threads,
                                                                fused_ext=fused_ext))
    return configs

def _get(config, key):
    if key in config:
        return config[key]
    return None

def _expand_configs_file(configuration_file):
    configs = configuration_file["configurations"]
    return _expand_configs(datasets=_get(configs, "datasets"), nbits=_get(configs, "nbits"),
                           nprobes=_get(configs, "nprobe"),t_primes=_get(configs, "t_prime"),
                           document_top_ks=_get(configs, "document_top_k"), runtimes=_get(configs, "runtime"),
                           split=_get(configs, "datasplit"), bound=_get(configs, "bound"), num_threads=_get(configs, "num_threads"),
                           fused_exts=_get(configs, "fused_ext"))

def _write_results(results_file, data):
    with open(results_file, "w") as file:
        file.write(json.dumps(data, indent=3))

def load_configuration(filename, info=False, overwrite=False):
    with open(filename, "r") as file:
        config_file = json.loads(file.read())
    name = config_file["name"]
    type_ = config_file["type"]
    params = _get(config_file, "parameters") or {}
    configs = _expand_configs_file(config_file)

    results_file = os.path.join("experiments/results", f"{name}.json")
    if not info:
        os.makedirs("experiments/results", exist_ok=True)
        assert not os.path.exists(results_file) or overwrite

        _write_results(results_file, [])
    return results_file, type_, params, configs

def _init_proc(env_vars):
    for key, value in env_vars.items():
        os.environ[key] = value

def _prepare_result(result):
    if "_update" not in result:
        return result
    # NOTE Used to update parameters determined at runtime.
    result = copy.deepcopy(result)
    update = result["_update"]
    for key, value in update.items():
        current = result["provenance"][key]
        assert current is None or current == value
        result["provenance"][key] = value
    del result["_update"]
    return result

def _execute_configs_parallel(configs, callback, type_, params, results_file, max_workers):
    env_vars = dict(os.environ)
    progress = tqdm(total=len(configs))
    results = []
    with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_proc, initargs=(env_vars,)
        ) as executor, redirect_stdout(
            io.StringIO()
        ) as rd_stdout:
        futures = {executor.submit(callback, config, params): config for config in configs}
        for future in as_completed(futures.keys()):
            result = future.result()
            config = futures[future]

            result["provenance"] = config
            result["provenance"]["type"] = type_
            result["provenance"]["parameters"] = params
            results.append(_prepare_result(result))
            _write_results(results_file=results_file, data=results)
            
            sys.stdout = sys.__stdout__
            sys.stdout = rd_stdout
            progress.update(1)
    progress.close()

def _execute_configs_sequential(configs, callback, type_, params, results_file):
    results = []
    for config in tqdm(configs):
        result = callback(config, params)
        result["provenance"] = config
        result["provenance"]["type"] = type_
        result["provenance"]["parameters"] = params
        results.append(_prepare_result(result))
        _write_results(results_file=results_file, data=results)

def execute_configs(exec_info, configs, results_file, type_, params, max_workers):
    exec_info = exec_info[type_]
    callback, parallelizable = exec_info["callback"], exec_info["parallelizable"]
    if parallelizable:
        _execute_configs_parallel(configs, callback, type_, params, results_file, max_workers=max_workers)
    else:
        _execute_configs_sequential(configs, callback, type_, params, results_file)

def read_subprocess_inputs():
    data = json.loads(input())
    return data["config"], data["params"]

def publish_subprocess_results(results):
    print("")
    print(json.dumps(results))
    print("#> Done")

def spawn_and_execute(script, config, params):
    process = subprocess.run(
        ["python", script],
        input=json.dumps({"config": config, "params": params}),
        stdout=subprocess.PIPE, 
        bufsize=1,
        text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()},
        cwd=os.getcwd()
    )
    response = process.stdout.strip().split("\n")
    if response[-1] != "#> Done" or process.returncode != 0:
        print(process.stderr, file=sys.stderr)
    assert response[-1] == "#> Done"
    return json.loads(response[-2])

def _strip_provenance(config, result):
    del result["parameters"]
    del result["type"]
    if "document_top_k" not in config:
        result["document_top_k"] = None
    if "num_threads" not in config:
        result["num_threads"] = 1
    return result

def check_execution(filename, configs, result_file):
    with open(filename, "r") as file:
        config_data = json.loads(file.read())
    with open(result_file, "r") as file:
        results = json.loads(file.read())
    result_configs = [_strip_provenance(config_data["configurations"], result["provenance"]) for result in results]
    missing = []
    for config in configs:
        if config not in result_configs:
            missing.append(config)
    print(len(missing), len(configs))
