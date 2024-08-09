import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from colbert.warp.utils.index_converter import convert_index
from colbert.warp.utils.collection_indexer import index
from colbert.warp.config import WARPRunConfig

if __name__ == "__main__":
    config = WARPRunConfig(
        nranks=4,
        dataset="beir",
        collection="scifact",
        datasplit="test",
        nbits=4,
        k=100,
    )

    index(config)
    index_path = os.path.join(config.index_root, config.index_name)
    convert_index(index_path)