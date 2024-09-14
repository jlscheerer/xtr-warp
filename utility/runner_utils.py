from warp.engine.config import WARPRunConfig

DEFAULT_K_VALUE = 10

def _make_runtime(runtime):
    assert runtime is None
    return runtime

def make_run_config(config):
    collection, dataset, split = config["collection"], config["dataset"], config["split"]
    nbits, nprobe, t_prime, bound = config["nbits"], config["nprobe"], config["t_prime"], config["bound"]

    return WARPRunConfig(
        collection=collection,
        dataset=dataset,
        type_="search" if collection == "lotte" else None,
        datasplit=split,
        nbits=nbits,
        nprobe=nprobe,
        t_prime=t_prime,
        k=DEFAULT_K_VALUE,
        runtime=_make_runtime(config["runtime"]),
        bound=bound
    )