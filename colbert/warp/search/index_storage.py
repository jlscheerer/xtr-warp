import os
import pathlib
import torch
import numpy as np
from itertools import product

from colbert.infra.config.config import ColBERTConfig
from colbert.utils.tracker import NOPTracker
from colbert.utils.utils import print_message
from colbert.warp.constants import T_PRIME_MAX

from torch.utils.cpp_extension import load

class IndexLoaderWARP:
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=True,
        load_index_with_mmap=False,
    ):
        assert not use_gpu
        assert not load_index_with_mmap

        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap

        (
            reversed_bit_map,
            decompression_lookup_table,
            bucket_weights,
        ) = self._load_buckets(config.nbits)

        self._load_codec()

        print_message(f"#> Loading repacked residuals...")
        self.residuals_repacked_compacted_df = torch.load(
            os.path.join(self.index_path, "residuals.repacked.compacted.pt")
        )

    def _load_buckets(self, nbits: int):
        print_message(f"#> Loading buckets...")

        bucket_weights = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_weights.npy"))
        )

        self.bucket_weights = bucket_weights

        reversed_bit_map = []
        mask = (1 << nbits) - 1
        for i in range(256):
            # The reversed byte
            z = 0
            for j in range(8, 0, -nbits):
                # Extract a subsequence of length n bits
                x = (i >> (j - nbits)) & mask

                # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
                y = 0
                for k in range(nbits - 1, -1, -1):
                    y += ((x >> (nbits - k - 1)) & 1) * (2**k)

                # Set the corresponding bits in the output byte
                z |= y
                if j > nbits:
                    z <<= nbits
            reversed_bit_map.append(z)
        reversed_bit_map = torch.tensor(reversed_bit_map).to(torch.uint8)

        # A table of all possible lookup orders into bucket_weights
        # given n bits per lookup
        keys_per_byte = 8 // nbits
        decompression_lookup_table = torch.tensor(
            list(product(list(range(len(bucket_weights))), repeat=keys_per_byte))
        ).to(torch.uint8)

        return reversed_bit_map, decompression_lookup_table, bucket_weights

    def _load_codec(self):
        print_message(f"#> Loading codec...")

        centroids = torch.from_numpy(
            np.load(os.path.join(self.index_path, "centroids.npy"))
        )
        sizes_compacted = torch.load(
            os.path.join(self.index_path, "sizes.compacted.pt")
        )
        codes_compacted = torch.load(
            os.path.join(self.index_path, "codes.compacted.pt")
        )

        residuals_compacted = torch.load(
            os.path.join(self.index_path, "residuals.compacted.pt")
        )

        ncentroids = centroids.shape[0]
        assert sizes_compacted.shape == (ncentroids,)

        nembeddings = residuals_compacted.shape[0]
        assert sizes_compacted.sum() == nembeddings
        assert codes_compacted.shape == (nembeddings,)

        self.sizes_compacted = sizes_compacted
        self.codes_compacted = codes_compacted

        offsets_compacted = torch.zeros((ncentroids + 1,), dtype=torch.long)
        torch.cumsum(sizes_compacted, dim=0, out=offsets_compacted[1:])
        self.offsets_compacted = offsets_compacted

        # TODO(jlscheerer) Make this more elegant by introducing a skip_mask
        # Hacky way to force low number of candidates for small entries
        self.kdummy_centroid = sizes_compacted.argmin().item()

        self.centroids = centroids
        print("#> Not averaging centroids.")

        return residuals_compacted


class IndexScorerWARP(IndexLoaderWARP):
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=False,
        load_index_with_mmap=False,
        t_prime=None
    ):
        assert not use_gpu
        assert not load_index_with_mmap

        super().__init__(
            index_path=index_path,
            config=config,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap,
        )

        IndexScorerWARP.try_load_torch_extensions(use_gpu)

        assert config.ncells is not None
        self.nprobe = config.ncells

        (ncentroids, _) = self.centroids.shape
        if t_prime is not None:
            self.t_prime = t_prime
        elif ncentroids <= 2**16:
            (nembeddings, _) = self.residuals_repacked_compacted_df.shape
            self.t_prime = int(np.sqrt(8 * nembeddings) / 1000) * 1000
        else: self.t_prime = T_PRIME_MAX

        assert config.nbits in [2, 4]
        self.nbits = config.nbits

        print("nprobe", self.nprobe, "t_prime", self.t_prime, "nbits", config.nbits)

        # TODO(jlscheerer) Determine the bound automatically
        # This should be set in accordance with the average number...
        self.bound = 128

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return
        cflags = [
            "-O3", "-mavx2", "-mfma", "-march=native", "-ffast-math", "-fno-math-errno", "-m64", "-fopenmp", "-std=c++17",
            "-funroll-loops", "-msse", "-msse2", "-msse3", "-msse4.1", "-mbmi2", "-mmmx", "-mavx", "-fomit-frame-pointer",
            "-fno-strict-aliasing"
        ]

        print_message(
            f"Loading warp_select_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.warp_select_centroids_cpp = load(
            name="warp_select_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "warp_select_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).warp_select_centroids_cpp

        print_message(
            f"Loading decompress_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.decompress_centroids_cpp = dict()
        decompress_centroids_cpp = load(
            name="decompress_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "decompress_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_centroids_cpp[2] = decompress_centroids_cpp.decompress_centroids_2_cpp
        cls.decompress_centroids_cpp[4] = decompress_centroids_cpp.decompress_centroids_4_cpp

        print_message(
            f"Loading merge_candidate_scores_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.merge_candidate_scores_cpp = load(
            name="merge_candidate_scores_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "merge_candidate_scores.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).merge_candidate_scores_cpp

        cls.loaded_extensions = True

    def rank(self, config, Q, k=100, filter_fn=None, pids=None, tracker=NOPTracker()):
        assert filter_fn is None
        assert pids is None

        nprobe = self.nprobe
        t_prime = self.t_prime
        with torch.inference_mode():
            tracker.begin("Candidate Generation")
            centroid_scores = Q.squeeze(0) @ self.centroids.T
            tracker.end("Candidate Generation")

            tracker.begin("top-k Precompute")
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            cells, centroid_scores, mse_estimates = self._warp_select_centroids(
                Q_mask, centroid_scores, nprobe, t_prime
            )
            tracker.end("top-k Precompute")

            tracker.begin("Decompression")
            capacities, candidate_sizes, candidate_pids, candidate_scores = self._decompress_centroids(
                Q.squeeze(0), cells, centroid_scores, nprobe
            )
            tracker.end("Decompression")

            # TODO(jlscheerer) Remove this again.
            tracker.begin("Lookup")
            tracker.end("Lookup")

            tracker.begin("Build Matrix")
            pids, scores = self._merge_candidate_scores(
                capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
            )
            tracker.end("Build Matrix")

            return pids, scores

    def _warp_select_centroids(self, Q_mask, centroid_scores, nprobe, t_prime):
        cells, scores, mse = IndexScorerWARP.warp_select_centroids_cpp(
            Q_mask, centroid_scores, self.sizes_compacted, nprobe, t_prime, self.bound
        )

        cells = cells.flatten().contiguous()
        scores = scores.flatten().contiguous()

        # NOTE Skip decompression of cells with a zero score centroid.
        # This means that the corresponding query tokens was 0. 
        cells[scores == 0] = self.kdummy_centroid

        return cells, scores, mse

    def _decompress_centroids(
        self, Q, centroid_ids, centroid_scores, nprobe
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores = IndexScorerWARP.decompress_centroids_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_repacked_compacted_df, self.bucket_weights, Q, nprobe
        )
        return capacities, sizes, pids, scores

    def _merge_candidate_scores(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
    ):
        pids, scores = IndexScorerWARP.merge_candidate_scores_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, self.nprobe, k
        )
        return pids.tolist(), scores.tolist()
