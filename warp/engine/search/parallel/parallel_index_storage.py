import os
import pathlib
import torch
import numpy as np

from warp.infra.config.config import ColBERTConfig
from warp.utils.tracker import NOPTracker
from warp.utils.utils import print_message
from warp.engine.constants import TPrimePolicy, T_PRIME_MAX

from torch.utils.cpp_extension import load

class ParallelIndexLoaderWARP:
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=True,
        load_index_with_mmap=False,
        fused_decompression_merge=True
    ):
        assert not use_gpu and not load_index_with_mmap

        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap
        self.fused_decompression_merge = fused_decompression_merge

        print_message(f"#> Loading buckets...")
        bucket_weights = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_weights.npy"))
        )
        self.bucket_weights = bucket_weights

        self._load_codec()
        print_message(f"#> Loading repacked residuals...")
        self.residuals_compacted = torch.load(
            os.path.join(self.index_path, "residuals.repacked.compacted.pt")
        )

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

        num_centroids = centroids.shape[0]
        assert sizes_compacted.shape == (num_centroids,)

        num_embeddings = residuals_compacted.shape[0]
        assert sizes_compacted.sum() == num_embeddings
        assert codes_compacted.shape == (num_embeddings,)

        self.sizes_compacted = sizes_compacted
        self.codes_compacted = codes_compacted

        offsets_compacted = torch.zeros((num_centroids + 1,), dtype=torch.long)
        torch.cumsum(sizes_compacted, dim=0, out=offsets_compacted[1:])
        self.offsets_compacted = offsets_compacted

        self.kdummy_centroid = sizes_compacted.argmin().item()

        self.centroids = centroids
        print("#> Not averaging centroids.")

        return residuals_compacted


class ParallelIndexScorerWARP(ParallelIndexLoaderWARP):
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=False,
        load_index_with_mmap=False,
        t_prime=None,
        bound=128,
        fused_decompression_merge=True
    ):
        assert not use_gpu
        assert not load_index_with_mmap

        super().__init__(
            index_path=index_path,
            config=config,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap,
            fused_decompression_merge=fused_decompression_merge
        )

        num_threads = torch.get_num_threads()
        assert num_threads != 1

        ParallelIndexScorerWARP.try_load_torch_extensions(use_gpu)

        assert config.ncells is not None
        self.nprobe = config.ncells

        (num_centroids, _) = self.centroids.shape
        if t_prime is not None:
            self.t_prime = TPrimePolicy(value=t_prime)
        elif num_centroids <= 2**16:
            (num_embeddings, _) = self.residuals_compacted.shape
            self.t_prime = TPrimePolicy(value=int(np.sqrt(8 * num_embeddings) / 1000) * 1000)
        else: self.t_prime = T_PRIME_MAX

        assert config.nbits in [2, 4]
        self.nbits = config.nbits

        self.centroid_idx = torch.stack(tuple([
            torch.arange(num_centroids, dtype=torch.int32) for _ in range(num_threads)
        ])).contiguous()

        print("nprobe", self.nprobe, "t_prime", self.t_prime, "nbits", config.nbits)
        self.bound = bound or 128

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
            f"Loading parallel_warp_select_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.warp_select_centroids_cpp = load(
            name="parallel_warp_select_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_warp_select_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).parallel_warp_select_centroids_cpp

        print_message(
            f"Loading parallel_decompress_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.decompress_centroids_cpp = dict()
        decompress_centroids_cpp = load(
            name="parallel_decompress_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_decompress_centroids.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_centroids_cpp[2] = decompress_centroids_cpp.parallel_decompress_centroids_2_cpp
        cls.decompress_centroids_cpp[4] = decompress_centroids_cpp.parallel_decompress_centroids_4_cpp

        print_message(
            f"Loading parallel_merge_candidate_scores_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.merge_candidate_scores_cpp = load(
            name="parallel_merge_candidate_scores_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_merge_candidate_scores.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).parallel_merge_candidate_scores_cpp

        print_message(
            f"Loading parallel_fused_decompress_merge extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.fused_decompress_merge_cpp = dict()
        fused_decompress_merge_cpp = load(
            name="parallel_fused_decompress_merge_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "parallel_fused_decompress_merge.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.fused_decompress_merge_cpp[2] = fused_decompress_merge_cpp.parallel_fused_decompress_merge_2_cpp
        cls.fused_decompress_merge_cpp[4] = fused_decompress_merge_cpp.parallel_fused_decompress_merge_4_cpp

        cls.loaded_extensions = True

    def rank(self, config, Q, k=100, filter_fn=None, pids=None, tracker=NOPTracker()):
        assert filter_fn is None
        assert pids is None

        with torch.inference_mode():
            tracker.begin("Candidate Generation")
            centroid_scores = Q.squeeze(0) @ self.centroids.T
            tracker.end("Candidate Generation")

            tracker.begin("top-k Precompute")
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            cells, centroid_scores, mse_estimates = self._warp_select_centroids(
                Q_mask, centroid_scores, self.nprobe, self.t_prime[k]
            )
            tracker.end("top-k Precompute")

            num_tokens = Q_mask.sum().item()
            if self.fused_decompression_merge:
                tracker.begin("Decompression")
                tracker.end("Decompression")

                tracker.begin("Build Matrix")
                pids, scores = self._fused_decompress_merge_scores(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe, num_tokens, mse_estimates, k
                )
                tracker.end("Build Matrix")
            else:
                tracker.begin("Decompression")
                capacities, candidate_sizes, candidate_pids, candidate_scores = self._decompress_centroids(
                    Q.squeeze(0), cells, centroid_scores, self.nprobe, num_tokens
                )
                tracker.end("Decompression")

                tracker.begin("Build Matrix")
                pids, scores = self._merge_candidate_scores(
                    capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k, num_tokens
                )
                tracker.end("Build Matrix")

            return pids, scores

    def _warp_select_centroids(self, Q_mask, centroid_scores, nprobe, t_prime):
        cells, scores, mse = ParallelIndexScorerWARP.warp_select_centroids_cpp(
            self.centroid_idx, Q_mask, centroid_scores, self.sizes_compacted, nprobe, t_prime, self.bound
        )

        cells = cells.flatten().contiguous()
        scores = scores.flatten().contiguous()

        # NOTE Skip decompression of cells with a zero score centroid.
        # This means that the corresponding query token was 0.0 (i.e., masked out). 
        cells[scores == 0] = self.kdummy_centroid

        return cells, scores, mse

    def _decompress_centroids(
        self, Q, centroid_ids, centroid_scores, nprobe, num_tokens
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores = ParallelIndexScorerWARP.decompress_centroids_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, num_tokens
        )
        return capacities, sizes, pids, scores

    def _merge_candidate_scores(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k, num_tokens
    ):
        pids, scores = ParallelIndexScorerWARP.merge_candidate_scores_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, self.nprobe, k, num_tokens
        )
        return pids.tolist(), scores.tolist()

    def _fused_decompress_merge_scores(
        self, Q, centroid_ids, centroid_scores, nprobe, num_tokens, mse_estimates, k
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        pids, scores = ParallelIndexScorerWARP.fused_decompress_merge_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe, num_tokens,
            mse_estimates, k
        )
        return pids.tolist(), scores.tolist()