import os
from dotenv import load_dotenv

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

# Load ENVIRONMENT variables. Be sure to change the .env file!
load_dotenv()

from colbert.xtr_run_config import XTRRunConfig, to_colbert_config
from colbert.infra import Run, RunConfig
from colbert.data import Queries
from colbert import Searcher

if __name__ == "__main__":
    config = XTRRunConfig(
        nranks=4,
        dataset="lotte",
        collection="writing",
        type_="search",
        datasplit="test",
        nbits=4,
        k=100,
    )

    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        searcher = Searcher(
            index=config.index_name,
            config=to_colbert_config(config),
            index_root=config.index_root,
            warp_engine=True,
        )
        queries = Queries(config.queries_path)

    ranking = searcher.search_all(queries, k=config.k)
