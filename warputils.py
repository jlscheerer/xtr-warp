import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from colbert.warp.utils.index_converter import convert_index
from colbert.warp.utils.collection_indexer import index
from colbert.xtr_run_config import XTRRunConfig


def convert(config: XTRRunConfig):
    index_path = os.path.join(config.index_root, config.index_name)
    convert_index(index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("warputils")
    parser.add_argument("mode")
    parser.add_argument("-c", "--collection", required=True)
    parser.add_argument("-s", "--split", required=True)
    parser.add_argument("-n", "--nbits", type=int, required=True)

    args = parser.parse_args()

    config = XTRRunConfig(
        nranks=4,
        dataset="lotte",
        collection=args.collection,
        type_="search",
        datasplit=args.split,
        nbits=args.nbits,
        k=100,
    )

    if args.mode == "search":
        # search(config, batch_queries=False)
        pass
    elif args.mode == "index":
        index(config)
    elif args.mode == "convert":
        convert(config)
    else:
        raise AssertionError
