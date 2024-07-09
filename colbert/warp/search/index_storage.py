import os
import pathlib
import torch
import numpy as np
from itertools import product

from colbert.infra.config.config import ColBERTConfig
from colbert.utils.tracker import NOPTracker

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

        decompression_lookup_table = self._load_buckets(config.nbits)

        # TODO(jlscheerer) Just directly emit torch tensors during conversion.
        self._load_codec()

        # TODO(jlscheerer) This is a REALLY unncessarily expensive computation.
        # We should eventually move this into the index conversion.
        print_message(f"#> Repacking residuals...")

    def _load_buckets(self, nbits: int):
        print_message(f"#> Loading buckets...")

        bucket_weights = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_weights.npy"))
        )

        # TODO(jlscheerer) We probably don't need to load the cutoffs anyways.
        bucket_cutoffs = torch.from_numpy(
            np.load(os.path.join(self.index_path, "bucket_cutoffs.npy"))
        )

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

        return decompression_lookup_table

    def _load_codec(self):
        print_message(f"#> Loading codec...")

        centroids = torch.from_numpy(
            np.load(os.path.join(self.index_path, "centroids.npy"))
        )
        sizes_compacted = torch.from_numpy(
            np.load(os.path.join(self.index_path, "sizes.compacted.npy"))
        )
        codes_compacted = torch.from_numpy(
            np.load(os.path.join(self.index_path, "codes.compacted.npy"))
        )
        residuals_compacted = torch.from_numpy(
            np.load(os.path.join(self.index_path, "residuals.compacted.npy"))
        )

        ncentroids = centroids.shape[0]
        assert sizes_compacted.shape == (ncentroids,)

        nembeddings = residuals_compacted.shape[0]
        assert sizes_compacted.sum() == nembeddings
        assert codes_compacted.shape == (nembeddings,)


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
        precompute_topk_centroids_cpp = load(
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
        decompress_centroid_embeds_strided_repacked_cpp = load(
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
        compute_candidate_scores_cpp = load(
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

    def rank(
        self,
        config,
        Q,
        filter_fn=None,
        pids=None,
        tracker=NOPTracker(),
    ):
        assert filter_fn is None
        assert pids is None

        raise NotImplementedError()
