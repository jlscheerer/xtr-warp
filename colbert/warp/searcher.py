from colbert.warp.config import WARPRunConfig
from colbert.infra import Run, RunConfig
from colbert.data import Queries
from colbert import Searcher


def load_index_from_config(config: WARPRunConfig, load_queries: bool = False):
    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        searcher = Searcher(
            index=config.index_name,
            config=config.colbert(),
            index_root=config.index_root,
            warp_engine=True,
        )
        if not load_queries:
            return searcher
        return searcher, Queries(config.queries_path)
