import os
from beir.datasets.data_loader import GenericDataLoader
import jsonlines
from collections import OrderedDict

from warp.infra import Run, RunConfig
from warp.data import Queries
from warp.infra.provenance import Provenance

from warp.engine.config import WARPRunConfig

class WARPQas:
    def __init__(self, num_total_qids, data):
        super().__init__()
        self.num_total_qids = num_total_qids
        self.data = data

def _load_qas_lotte(qas_path):
    qas = OrderedDict()
    num_total_qids = 0
    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            num_total_qids += 1
            answer_pids = set(line["answer_pids"])
            qas[qid] = answer_pids
    return WARPQas(num_total_qids=num_total_qids, data=dict(qas))

class WARPQRels:
    def __init__(self, config):
        self.config = config
        if self.config.collection == "beir":
            BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
            dataset_path = os.path.join(BEIR_COLLECTION_PATH, self.config.dataset)
            corpus, queries, qrels = GenericDataLoader(dataset_path).load(split=self.config.datasplit)
            self.qrels = qrels
        elif self.config.collection == "lotte":
            LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]
            dataset_path = os.path.join(LOTTE_COLLECTION_PATH, self.config.dataset, self.config.datasplit)
            qas_path = os.path.join(dataset_path, f"qas.{self.config.type_}.jsonl")
            self.qas = _load_qas_lotte(qas_path)

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
