import os
# Enforces CPU-only execution of torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import psutil

from utility.executor_utils import read_subprocess_inputs, publish_subprocess_results

if __name__ == "__main__":
    config, params = read_subprocess_inputs()

    num_threads = params.get("num_threads", 1)

    proc = psutil.Process()
    if "cpu_affinity" in params:
        # Set the cpu_affinity, e.g., [0, 1] for CPUs #0 and #1
        # Reference: https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_affinity
        proc.cpu_affinity(params["cpu_affinity"])

    # Configure environment to ensure *correct* number of threads.
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"]= str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_AFFINITY"] = "disabled"

    import torch
    torch.set_num_threads(num_threads)

    from utility.runner_utils import make_run_config

    from warp.engine.searcher import WARPSearcher
    from warp.data.queries import WARPQueries
    from warp.utils.tracker import ExecutionTracker

    run_config = make_run_config(config, num_threads=num_threads)

    searcher = WARPSearcher(run_config)
    queries = WARPQueries(run_config)
    steps = ["Query Encoding", "Candidate Generation", "top-k Precompute", "Decompression", "Build Matrix"]
    tracker = ExecutionTracker(name="XTR/WARP", steps=steps)

    k = config["document_top_k"]
    rankings = searcher.search_all(queries, k=k, batched=False, tracker=tracker, show_progress=True)
    metrics = rankings.evaluate(queries.qrels, k=k)

    ranker = searcher.searcher.ranker
    statistics = {
        "centroids": ranker.centroids.shape[0],
        "embeddings": ranker.residuals_compacted.shape[0],
        "median_size": ranker.sizes_compacted.median().item()
    }

    params = {
        "nprobe": ranker.nprobe,
        "t_prime": ranker.t_prime,
        "document_top_k": searcher.config.k,
        "bound": ranker.bound
    }

    publish_subprocess_results({
        "tracker": tracker.as_dict(),
        "metrics": metrics,
        "statistics": statistics,
        "_update": params,
    })