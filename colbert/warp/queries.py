import os
import io

from contextlib import redirect_stdout

from colbert.warp.config import WARPRunConfig
from colbert.infra import Run, RunConfig
from colbert.data import Queries, Ranking

from colbert.utilities.evaluation.evaluate_lotte_rankings import evaluate_dataset


class WARPQRels:
    def __init__(self, config):
        self.config = config


class WARPQueries:
    def __init__(self, config: WARPRunConfig):
        self.config = config
        with Run().context(
            RunConfig(nranks=config.nranks, experiment=config.experiment_name)
        ):
            self.queries = Queries(config.queries_path)

    @property
    def qrels(self):
        return WARPQRels(self.config)


class WARPRanking:
    def __init__(self, ranking: Ranking):
        self.ranking = ranking

    def evaluate(self, qrels: WARPQRels, k: int, cleanup: bool = True):
        _ = io.StringIO()
        with redirect_stdout(_):
            rankings_path = self.ranking.save(
                f"{qrels.config.index_name}.split={qrels.config.datasplit}.ranking.k={k}.tsv"
            )
            assert qrels.config.dataset == "lotte"
            results = evaluate_dataset(
                rankings_path,
                qrels.config.collection,
                split=qrels.config.datasplit,
                k=k,
            )
        del _
        if cleanup:
            os.remove(rankings_path)
            os.remove(f"{rankings_path}.meta")
        return results
