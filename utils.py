import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from colbert.warp.utils.index_converter import convert_index
from colbert.warp.utils.collection_indexer import index
from colbert.warp.config import WARPRunConfig


def parse_warp_run_config(collection, dataset, type_, split, nbits):
    if collection not in ["lotte", "beir"] or dataset is None or split is None or nbits is None:
        return None
    if collection == "lotte" and type_ is None:
        return None
    return WARPRunConfig(
        nranks=4,
        collection=collection,
        dataset=dataset,
        type_=type_,
        datasplit=split,
        nbits=nbits,
        k=100,
    )

def get_warp_run_config(parser, args):
    config = parse_warp_run_config(collection=args.collection, dataset=args.dataset, type_=args.type,
                                   split=args.split, nbits=args.nbits)
    if config is None:
        parser.error("Invalid warp run config specified.")
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="xtr-warp",
        description="Utilities for XTR/WARP index creation/evaluation"
    )

    parser.add_argument("mode", choices=["index"], nargs=1)
    parser.add_argument("-c", "--collection", choices=["beir", "lotte"])
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-t", "--type", choices=["search", "forum"])
    parser.add_argument("-s", "--split", choices=["train", "test", "dev"])
    parser.add_argument("-n", "--nbits", type=int, choices=[1, 2, 4, 8])
    args = parser.parse_args()

    assert len(args.mode) == 1
    mode = args.mode[0]

    if mode == "index":
        config = get_warp_run_config(parser, args)
        index(config)
        index_path = os.path.join(config.index_root, config.index_name)
        convert_index(index_path)
    else: raise AssertionError
