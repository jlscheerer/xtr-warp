from warp.engine.config import WARPRunConfig
from warp.infra import Run, RunConfig
from warp import Indexer


def index(config: WARPRunConfig):
    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        indexer = Indexer(checkpoint="google/xtr-base-en", config=config.colbert())
        indexer.index(name=config.index_name, collection=config.collection_path)
