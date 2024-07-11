from colbert.warp.config import WARPRunConfig
from colbert.infra import Run, RunConfig
from colbert import Indexer


def index(config: WARPRunConfig):
    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        indexer = Indexer(checkpoint="google/xtr-base-en", config=config.colbert())
        indexer.index(name=config.index_name, collection=config.collection_path)
