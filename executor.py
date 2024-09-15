import argparse
import psutil

from utility.executor_utils import load_configuration, execute_configs, spawn_and_execute, check_execution
from utility.runner_utils import make_run_config

from utility.index_sizes import safe_index_size, bytes_to_gib

def index_size(config, params):
    assert len(params) == 0
    run_config = make_run_config(config)
    index_size_bytes = safe_index_size(run_config)
    return {
        "index_size_bytes": index_size_bytes,
        "index_size_gib": bytes_to_gib(index_size_bytes)
    }

def latency(config, params):
    return {}

def metrics(config, params):
    run = spawn_and_execute("utility/latency_runner.py", config, params)
    return {
        "metrics": run["metrics"],
        "statistics": run["statistics"],
        "_update": run["_update"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='XTR/WARP Experiment [Executor/Platform]')
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-i", "--info", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    
    MAX_WORKERS = args.workers or psutil.cpu_count(logical=False)
    OVERWRITE = args.overwrite
    results_file, type_, params, configs = load_configuration(args.config, info=args.info, overwrite=OVERWRITE)

    if args.info:
        check_execution(args.config, configs, results_file)
        exit(0)
    
    EXEC_INFO = {
        "index_size": {"callback": index_size, "parallelizable": True},
        "latency": {"callback": latency, "parallelizable": False},
        "metrics": {"callback": metrics, "parallelizable": True}
    }
    execute_configs(EXEC_INFO, configs, results_file=results_file, type_=type_,
                    params=params, max_workers=MAX_WORKERS)