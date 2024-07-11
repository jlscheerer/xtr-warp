import os
import pathlib
import torch
import numpy as np
from itertools import product
from tqdm import tqdm

from colbert.infra.config.config import ColBERTConfig
from colbert.utils.tracker import NOPTracker
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message

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

        # TODO(jlscheerer) Just directly emit torch tensors during conversion.
        residuals_compacted = self._load_codec(
            reversed_bit_map, decompression_lookup_table, bucket_weights
        )

        # TODO(jlscheerer) This is a REALLY unncessarily expensive computation.
        # We should eventually move this into the index conversion.
        print_message(f"#> Loading repacked residuals...")
        self.residuals_repacked_compacted_df = torch.load(
            os.path.join(self.index_path, "residuals.repacked.compacted.pt")
        )
        # residuals_repacked_strided = StridedTensor(residuals_repacked_compacted_df, sizes_compacted, use_gpu=False)

    def _load_buckets(self, nbits: int):
        print_message(f"#> Loading buckets...")

        bucket_weights = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_weights.npy"))
        )

        # bucket_cutoffs = torch.from_numpy(
        #     np.load(os.path.join(self.index_path, "bucket_cutoffs.npy"))
        # )

        self.bucket_weights = bucket_weights

        # TODO(jlscheerer) We could just directly store this as part of the index.
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

    # TODO(jlscheerer) We really don't need all of these arguments once we directly emit the correct index.
    def _load_codec(self, reversed_bit_map, decompression_lookup_table, bucket_weights):
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
        self.codes_strided = StridedTensor(
            codes_compacted, sizes_compacted, use_gpu=self.use_gpu
        )

        offsets_compacted = torch.zeros((ncentroids + 1,), dtype=torch.long)
        torch.cumsum(sizes_compacted, dim=0, out=offsets_compacted[1:])
        self.offsets_compacted = offsets_compacted

        # TODO(jlscheerer) Make this more elegant by introducing a skip_mask
        # Hacky way to force low number of candidates for small entries
        self.kdummy_centroid = sizes_compacted.argmin().item()

        # TODO(jlscheerer) This is a REALLY unncessarily expensive computation.
        # We should eventually move this into the index conversion.
        kaverage_centroids = True

        if kaverage_centroids:
            self.avg_centroids = torch.load(
                os.path.join(self.index_path, "avg_centroids.pt")
            )
        else:
            self.centroids = centroids

        return residuals_compacted


class IndexScorerWARP(IndexLoaderWARP):
    def __init__(
        self,
        index_path,
        config: ColBERTConfig,
        use_gpu=False,
        load_index_with_mmap=False,
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

        # self.ivf_strided = StridedTensor(
        #     self.codes_compacted, self.sizes_compacted, use_gpu=self.use_gpu
        # )

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        # TODO(jlscheerer) Add un-optimized CPP/Python Implementations for comparison.
        print_message(
            f"Loading precompute_topk_centroids_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.precompute_topk_centroids_cpp = load(
            name="precompute_topk_centroids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "precompute_topk_centroids.cpp",
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).precompute_topk_centroids_cpp

        print_message(
            f"Loading decompress_centroid_embeds_strided_repacked_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.decompress_centroid_embeds_strided_repacked_cpp = load(
            name="decompress_centroid_embeds_strided_repacked_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "decompress_centroid_embeds_strided_repacked.cpp",
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).decompress_centroid_embeds_strided_repacked_cpp

        print_message(
            f"Loading compute_candidate_scores_cpp extension (set WARP_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        cls.compute_candidate_scores_cpp = load(
            name="compute_candidate_scores_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "compute_candidate_scores.cpp",
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("WARP_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        ).compute_candidate_scores_cpp

        cls.loaded_extensions = True

    def rank(self, config, Q, k=100, filter_fn=None, pids=None, tracker=NOPTracker()):
        assert filter_fn is None
        assert pids is None

        # TODO(jlscheerer) Move this back into the config.
        nprobe = 12  # config.ncells

        with torch.inference_mode():
            # Compute the MSE

            tracker.begin("Candidate Generation")
            # centroid_scores = self.centroids @ Q.squeeze(0).T
            centroid_scores = self.avg_centroids @ Q.squeeze(0).T

            tracker.end("Candidate Generation")

            fill_blank = 10_000
            Q_mask = Q.squeeze(0).count_nonzero(dim=1) != 0
            (
                cells,
                centroid_scores,
                mse_estimates,
            ) = self._precompute_topk_centroids_native(
                Q_mask, centroid_scores, nprobe, fill_blank, tracker
            )

            tracker.begin("Decompression")
            # Decompression
            # NOTE: This is a significant speed-up compared to the naive approach.
            (
                decompressed_candidate_scores_strided,
                decompressed_sizes,
            ) = self._decompress_centroid_embeds_native_strided_repacked(
                Q.squeeze(0), cells, centroid_scores, nprobe
            )
            tracker.end("Decompression")

            tracker.begin("Lookup")
            # TODO(jlscheerer) Investigate: this seems to be slower than the naive version!
            # TODO(jlscheerer) Eventually just merge both into something that can be mmapped.
            candidate_pids_strided, candidate_sizes = self.codes_strided.lookup(cells)
            tracker.end("Lookup")

            pids, scores = self._compute_candidate_scores_native(
                cells,
                candidate_pids_strided,
                decompressed_candidate_scores_strided,
                candidate_sizes,
                mse_estimates,
                nprobe,
                k,
                tracker,
            )

            return pids, scores

    def _precompute_topk_centroids_native(
        self, Q_mask, centroid_scores, nprobe, fill_blank, tracker
    ):
        # TODO(jlscheerer) Compute centroid_scores differently so we don't need to tranpose...
        # tracker.begin("MSE Computation")
        # tracker.end("MSE Computation")

        tracker.begin("top-k Precompute")
        cells, centroid_scores, mse = IndexScorerWARP.precompute_topk_centroids_cpp(
            Q_mask, centroid_scores.T, self.sizes_compacted, nprobe, fill_blank
        )

        cells = cells.flatten().contiguous()  # (32 * nprobe,)
        centroid_scores = centroid_scores.flatten().contiguous()

        # NOTE This SIGNIFICANTLY IMPROVES performance, because we don't unnecessarily do decompression
        # We just decompress the dummy centroid here! ~60it/s vs 35it/s, without loss of performance.
        # TODO(jlscheerer) Fix this and use a "skip_mask". This requires correct handling of strides with
        #                  a length of zero.
        cells[centroid_scores == 0] = self.kdummy_centroid
        tracker.end("top-k Precompute")

        return cells, centroid_scores, mse

    def _decompress_centroid_embeds_native_strided_repacked(
        self, Q, centroid_ids, centroid_scores, nprobe
    ):
        begins = self.offsets_compacted[centroid_ids]
        ends = self.offsets_compacted[centroid_ids + 1]

        sizes = ends - begins
        results = IndexScorerWARP.decompress_centroid_embeds_strided_repacked_cpp(
            begins,
            ends,
            sizes,
            centroid_scores,
            self.residuals_repacked_compacted_df,
            self.bucket_weights,
            Q,
            nprobe,
        )

        return results, sizes

    def _compute_candidate_scores_native(
        self,
        cells,
        candidate_pids_strided,
        decompressed_candidate_scores_strided,
        candidate_sizes,
        mse_estimates,
        nprobe,
        k,
        tracker,
    ):
        # tracker.begin("Prepare Matrix")
        # tracker.end("Prepare Matrix")

        tracker.begin("Build Matrix")
        pids, scores = IndexScorerWARP.compute_candidate_scores_cpp(
            candidate_pids_strided,
            decompressed_candidate_scores_strided,
            candidate_sizes,
            mse_estimates,
            nprobe,
            k,
        )
        tracker.end("Build Matrix")

        # tracker.begin("Sort")
        # tracker.end("Sort")
        return pids.tolist(), scores.tolist()
