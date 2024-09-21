import os
import pathlib
import torch
import numpy as np
from itertools import product

from warp.infra.config.config import ColBERTConfig
from warp.utils.tracker import NOPTracker
from warp.search.strided_tensor import StridedTensor
from warp.utils.utils import print_message
from warp.engine.constants import TPrimePolicy, QUERY_MAXLEN, TOKEN_EMBED_DIM, T_PRIME_MAX

from torch_scatter import scatter_max, scatter_min

from torch.utils.cpp_extension import load

class IndexLoaderWARP:
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=True,
        load_index_with_mmap=False,
    ):
        assert not use_gpu and not load_index_with_mmap

        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap

        print_message(f"#> Loading buckets...")
        bucket_weights = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_weights.npy"))
        )
        self.bucket_weights = bucket_weights

        # [ ========== [Ablation] ========== ]
        reversed_bit_map = []
        mask = (1 << config.nbits) - 1
        for i in range(256):
            # The reversed byte
            z = 0
            for j in range(8, 0, -config.nbits):
                # Extract a subsequence of length n bits
                x = (i >> (j - config.nbits)) & mask

                # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
                y = 0
                for k in range(config.nbits - 1, -1, -1):
                    y += ((x >> (config.nbits - k - 1)) & 1) * (2**k)

                # Set the corresponding bits in the output byte
                z |= y
                if j > config.nbits:
                    z <<= config.nbits
            reversed_bit_map.append(z)
        self.reversed_bit_map = torch.tensor(reversed_bit_map).to(torch.uint8)

        # A table of all possible lookup orders into bucket_weights
        # given n bits per lookup
        keys_per_byte = 8 // config.nbits
        self.decompression_lookup_table = torch.tensor(
            list(product(list(range(len(bucket_weights))), repeat=keys_per_byte))
        ).to(torch.uint8)
        # [ ========== [/Ablation] ========== ]

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

        # [ ========== [Ablation] ========== ]
        self.codes_strided = StridedTensor(
            codes_compacted, sizes_compacted, use_gpu=self.use_gpu
        )
        self.residuals_strided = StridedTensor(
            residuals_compacted, sizes_compacted, use_gpu=self.use_gpu
        )
        self.token_idx_expanded = torch.arange(TOKEN_EMBED_DIM).repeat(self.sizes_compacted.max())
        # [ ========== [/Ablation] ========== ]

        offsets_compacted = torch.zeros((num_centroids + 1,), dtype=torch.long)
        torch.cumsum(sizes_compacted, dim=0, out=offsets_compacted[1:])
        self.offsets_compacted = offsets_compacted

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
        t_prime=None,
        bound=128,
        ablation_params=None
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

        (num_centroids, _) = self.centroids.shape
        if t_prime is not None:
            self.t_prime = TPrimePolicy(value=t_prime)
        elif num_centroids <= 2**16:
            (num_embeddings, _) = self.residuals_compacted.shape
            self.t_prime = TPrimePolicy(value=int(np.sqrt(8 * num_embeddings) / 1000) * 1000)
        else: self.t_prime = T_PRIME_MAX

        assert config.nbits in [2, 4]
        self.nbits = config.nbits

        self.bound = bound or 128

        # [ ========== [Ablation] ========== ]
        assert ablation_params is not None

        # Configure which functions to use for evaluation
        self.centroid_selection_fn = {
            "warp_select_py": self._warp_select_centroids_python,
            "warp_select_cpp": self._warp_select_centroids
        }[ablation_params["selection_fn"]]

        self.centroid_decompression_fn = {
            "explicit_decompress_py": self._explicit_decompress_python,
            "score_decompress_py": self._score_decompress_python,
            "score_decompress_cpp": self._decompress_centroids,
        }[ablation_params["decompression_fn"]]

        # NOTE this is only applicable if centroid_decompression_fn == _explicit_decompress_python
        self.normalized_decompression = ablation_params["normalized_decompression"]

        self.candidate_aggregation_fn = {
            "matrix_aggregate_py": self._matrix_candidate_scores_python,
            "hash_aggregate_cpp": self._aggregate_candidate_scores,
            "merge_aggregate_cpp": self._merge_candidate_scores
        }[ablation_params["aggregate_fn"]]

        self.compute_mse_via_reduce = ablation_params["compute_mse_via_reduce"]
        # [ ========== [/Ablation] ========== ]

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

        # [ ========== [Ablation] ========== ]
        print_message(
            f"Loading aggregate_candidate_scores_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.aggregate_candidate_scores_cpp = load(
            name="aggregate_candidate_scores_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "aggregate_candidate_scores.cpp",
                ),
            ],
            extra_cflags=cflags,
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).aggregate_candidate_scores_cpp
        # [ ========== [/Ablation] ========== ]

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
            cells, centroid_scores, mse_estimates = self.centroid_selection_fn(
                Q_mask, centroid_scores, self.nprobe, self.t_prime[k], compute_mse=not self.compute_mse_via_reduce
            )
            tracker.end("top-k Precompute")

            tracker.begin("Decompression")
            capacities, candidate_sizes, candidate_pids, candidate_scores = self.centroid_decompression_fn(
                Q.squeeze(0), cells, centroid_scores, self.nprobe
            )
            tracker.end("Decompression")

            tracker.begin("MSE via Reduction")
            if self.compute_mse_via_reduce:
                mse_estimates = self._compute_mse_reduce(
                    Q_mask, capacities, candidate_pids, candidate_scores
                )
            tracker.end("MSE via Reduction")

            tracker.begin("Build Matrix")
            pids, scores = self.candidate_aggregation_fn(
                capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
            )
            tracker.end("Build Matrix")

            return pids, scores

    def _compute_mse_reduce(
        self, Q_mask, capacities, candidate_pids, candidate_scores
    ):
        idx_to_candidate_pid = torch.unique(candidate_pids.flatten(), sorted=True)
        num_candidates = idx_to_candidate_pid.shape[0]

        # Construct a tensor indicating the qtoken_idx for each candidate.
        candidate_qids = torch.arange(QUERY_MAXLEN).repeat_interleave(self.nprobe).repeat_interleave(capacities)

        non_zero_mask = candidate_scores != 0
        non_zero_scores = candidate_scores[non_zero_mask]
        non_zero_qids = candidate_qids[non_zero_mask]

        mse_estimates = torch.full((QUERY_MAXLEN,), fill_value=1, dtype=torch.float)

        # Populate the score matrix using the maximum value for each index, i.e., max-reduction phase
        scatter_min(non_zero_scores.flatten(), non_zero_qids, out=mse_estimates)
        return (mse_estimates * Q_mask.int()).contiguous()

    def _warp_select_centroids_python(self, Q_mask, centroid_scores, nprobe, t_prime, compute_mse=True):
        topk_bound = self.bound if compute_mse else nprobe
        topk = torch.topk(centroid_scores, k=topk_bound, sorted=True)

        if compute_mse:
            cluster_sizes = self.sizes_compacted[topk.indices]
            cummulative_cluster_sizes = torch.cumsum(cluster_sizes, dim=1)
            cluster_index_mask = cummulative_cluster_sizes >= t_prime

            # If there are multiple maximal values then the indices of the first maximal value are returned.
            # Source: https://pytorch.org/docs/stable/generated/torch.argmax.html
            t_prime_index = torch.argmax(cluster_index_mask.int(), dim=1)
            mse_estimates = topk.values[torch.arange(QUERY_MAXLEN), t_prime_index]
        else:
            mse_estimates = None

        cells = topk.indices[:, :nprobe].flatten().contiguous()
        scores = topk.values[:, :nprobe].flatten().contiguous()

        cells[scores == 0] = self.kdummy_centroid
        return cells, scores, mse_estimates

    def _warp_select_centroids(self, Q_mask, centroid_scores, nprobe, t_prime, compute_mse=True):
        cells, scores, mse = IndexScorerWARP.warp_select_centroids_cpp(
            Q_mask, centroid_scores, self.sizes_compacted, nprobe, t_prime, self.bound
        )

        cells = cells.flatten().contiguous()
        scores = scores.flatten().contiguous()

        # NOTE Skip decompression of cells with a zero score centroid.
        # This means that the corresponding query token was 0.0 (i.e., masked out). 
        cells[scores == 0] = self.kdummy_centroid

        return cells, scores, mse

    # "Naive" Decompression
    def _explicit_decompress_python(self, Q, centroid_ids, centroid_scores, nprobe):
        centroid_ids = centroid_ids.long()
        centroids = self.centroids[centroid_ids]

        # Decompress the residuals at `centroid_ids`
        residuals, candidate_sizes = self.residuals_strided.lookup(centroid_ids)
        residuals = self.reversed_bit_map[residuals.long()]
        residuals = self.decompression_lookup_table[residuals.long()]
        residuals = residuals.reshape(residuals.shape[0], -1)
        residuals = self.bucket_weights[residuals.long()]

        decompressed_embeddings = torch.repeat_interleave(centroids, candidate_sizes, dim=0)
        decompressed_embeddings.add_(residuals)
        if self.normalized_decompression:
            decompressed_embeddings = torch.nn.functional.normalize(decompressed_embeddings.to(torch.float32), p=2, dim=-1)

        # Retrieve the corresponding passage ids, i.e., pids
        pids, _ = self.codes_strided.lookup(centroid_ids)

        # Multiply the decompressed residuals with the corresponding query vectors
        qids_expanded = torch.repeat_interleave(torch.arange(32), nprobe)
        candidate_qids_strided = torch.repeat_interleave(qids_expanded, candidate_sizes)

        Q_strided = Q[candidate_qids_strided, :]
        scores = torch.bmm(decompressed_embeddings.unsqueeze(1), Q_strided.unsqueeze(2)).flatten()

        return candidate_sizes, candidate_sizes.to(torch.int32), pids.to(torch.int32), scores

    # Entire Decompression using a single constant-sized MM.
    def _score_decompress_python(self, Q, centroid_ids, centroid_scores, nprobe):
        centroid_ids = centroid_ids.long()

        # Decompress the residuals at `centroid_ids`
        residuals, candidate_sizes = self.residuals_strided.lookup(centroid_ids)
        residuals = self.reversed_bit_map[residuals.long()]
        residuals = self.decompression_lookup_table[residuals.long()]
        residuals = residuals.view(-1).long()

        # NOTE Use the bucket_weights to compute scores directly.
        bucket_scores = Q.unsqueeze(2) @ self.bucket_weights.unsqueeze(0)

        decompressed_scores = torch.zeros((residuals.numel() // TOKEN_EMBED_DIM,))
        qid_candidates_compacted = torch.zeros((QUERY_MAXLEN * nprobe + 1,), dtype=torch.long)
        torch.cumsum(candidate_sizes, dim=0, out=qid_candidates_compacted[1:])
        for cell_idx in range(nprobe * (Q.squeeze(0).count_nonzero(dim=1) != 0).sum().item()):
            begin, end = qid_candidates_compacted[cell_idx], qid_candidates_compacted[cell_idx + 1]
            flat_residual_stride = residuals[begin * TOKEN_EMBED_DIM:end * TOKEN_EMBED_DIM]
            decompressed_scores_stride = bucket_scores[
                cell_idx // nprobe, self.token_idx_expanded[:candidate_sizes[cell_idx] * TOKEN_EMBED_DIM], flat_residual_stride
            ].view(-1, TOKEN_EMBED_DIM).sum(dim=1)
            torch.add(centroid_scores[cell_idx], decompressed_scores_stride, out=decompressed_scores[begin:end])

        # Retrieve the corresponding passage ids, i.e., pids
        pids, _ = self.codes_strided.lookup(centroid_ids)

        return candidate_sizes, candidate_sizes.to(torch.int32), pids.to(torch.int32), decompressed_scores

    # CPP Extension
    def _decompress_centroids(
        self, Q, centroid_ids, centroid_scores, nprobe
    ):
        centroid_ids = centroid_ids.long()
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        capacities = ends - begins
        sizes, pids, scores = IndexScorerWARP.decompress_centroids_cpp[self.nbits](
            begins, ends, capacities, centroid_scores, self.codes_compacted,
            self.residuals_compacted, self.bucket_weights, Q, nprobe
        )
        return capacities, sizes, pids, scores

    # "Reduce" token-level score to documnent-level scores by explicitly building the matrix.
    def _matrix_candidate_scores_python(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
    ):
        idx_to_candidate_pid = torch.unique(candidate_pids.flatten(), sorted=True)
        num_candidates = idx_to_candidate_pid.shape[0]

        # Construct a tensor indicating the qtoken_idx for each candidate.
        candidate_qids = torch.arange(QUERY_MAXLEN).repeat_interleave(self.nprobe).repeat_interleave(capacities)

        # Construct a tensor indicating the index corresponding to each of the candidate passage ids.
        candidate_pids_idx = torch.searchsorted(idx_to_candidate_pid, candidate_pids)

        # *Exact* index into the flattened score matrix for (candidate_pid, token_idx)
        indices = candidate_pids_idx * QUERY_MAXLEN + candidate_qids

        # candidate_qids_strided = torch.repeat_interleave(torch.arange(QUERY_MAXLEN), capacities)
        score_matrix = torch.zeros((num_candidates, QUERY_MAXLEN), dtype=torch.float)

        # Populate the score matrix using the maximum value for each index, i.e., max-reduction phase
        flat_score_matrix, _ = scatter_max(candidate_scores.flatten(), indices, out=score_matrix.view(-1))
        score_matrix = flat_score_matrix.view(score_matrix.size())

        mse_matrix = mse_estimates.view(-1, QUERY_MAXLEN).expand_as(score_matrix)
        zero_mask = score_matrix == 0
        score_matrix[zero_mask] = mse_matrix[zero_mask]

        scores = score_matrix.sum(dim=1)

        scores, indices = torch.sort(scores, stable=True, descending=True)
        pids = idx_to_candidate_pid[indices]

        return pids[:k].tolist(), scores[:k].tolist()

    # CPP Extension (Hashmap)
    def _aggregate_candidate_scores(self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k):
        pids, scores = IndexScorerWARP.aggregate_candidate_scores_cpp(
            candidate_pids, candidate_scores, capacities, mse_estimates, self.nprobe, k
        )
        return pids.tolist(), scores.tolist()

    # CPP Extension (Merge)
    def _merge_candidate_scores(
        self, capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, k
    ):
        pids, scores = IndexScorerWARP.merge_candidate_scores_cpp(
            capacities, candidate_sizes, candidate_pids, candidate_scores, mse_estimates, self.nprobe, k
        )
        return pids.tolist(), scores.tolist()
