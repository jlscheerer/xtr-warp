import os
from beir.datasets.data_loader import GenericDataLoader

from colbert.warp.config import WARPRunConfig
from colbert.infra import Run, RunConfig
from colbert.data import Queries
from colbert.infra.provenance import Provenance


class WARPQRels:
    def __init__(self, config):
        self.config = config
        if self.config.dataset == "beir":
            BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
            dataset_path = os.path.join(BEIR_COLLECTION_PATH, self.config.collection)
            corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
            self.qrels = qrels

class WARPQueries:
    def __init__(self, config: WARPRunConfig):
        self.config = config
        with Run().context(
            RunConfig(nranks=config.nranks, experiment=config.experiment_name)
        ):
            self.queries = Queries(config.queries_path)

    def provenance(self, source, k):
        provenance = Provenance()
        provenance.source = source
        provenance.queries = self.queries.provenance()
        provenance.config = self.config.colbert().export()
        provenance.k = k
        return provenance

    def __len__(self):
        return self.queries.__len__()

    def __iter__(self):
        return self.queries.__iter__()

    def __getitem__(self, key):
        return self.queries.__getitem__(key)

    @property
    def qrels(self):
        return WARPQRels(self.config)
