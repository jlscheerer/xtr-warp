from colbert.xtr_run_config import XTRRunConfig, to_colbert_config
from colbert.infra import Run, RunConfig
from colbert import Indexer


def index(config: XTRRunConfig):
    with Run().context(
        RunConfig(nranks=config.nranks, experiment=config.experiment_name)
    ):
        indexer = Indexer(
            checkpoint="google/xtr-base-en", config=to_colbert_config(config)
        )
        indexer.index(name=config.index_name, collection=config.collection_path)
