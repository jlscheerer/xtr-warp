import os
import json

import torch
import numpy as np
from itertools import product

from tqdm import tqdm

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN


def segmented_index_cumsum(input_tensor, offsets):
    values, indices = input_tensor.sort(stable=True)
    unique_values, inverse_indices, counts_values = torch.unique(
        values, return_inverse=True, return_counts=True
    )
    offset_arange = torch.arange(1, len(unique_values) + 1)
    offset_count_indices = offset_arange[inverse_indices]

    offset_counts = torch.zeros(counts_values.shape[0] + 1, dtype=torch.long)
    offset_counts[1:] = torch.cumsum(counts_values, dim=0)

    counts = torch.zeros_like(input_tensor, dtype=torch.long)
    counts[indices] = (
        torch.arange(0, input_tensor.shape[0]) - offset_counts[offset_count_indices - 1]
    )

    return counts + offsets[input_tensor.long()], offsets + torch.bincount(
        input_tensor, minlength=offsets.shape[0]
    )


def convert_index(index_path, destination_path=None):
    if destination_path is None:
        destination_path = index_path
    print(f"Compacting index at '{index_path}' to '{destination_path}'")
    os.makedirs(destination_path, exist_ok=True)
    with open(os.path.join(index_path, "plan.json"), "r") as file:
        plan = json.load(file)

    config = plan["config"]

    checkpoint = config["checkpoint"]
    assert checkpoint == "google/xtr-base-en"

    dim = config["dim"]
    nbits = config["nbits"]

    query_maxlen = config["query_maxlen"]
    doc_maxlen = config["doc_maxlen"]

    assert query_maxlen == QUERY_MAXLEN
    assert doc_maxlen == DOC_MAXLEN

    num_chunks = plan["num_chunks"]
    num_partitions = plan["num_partitions"]  # i.e., num_centroids

    centroids = torch.load(os.path.join(index_path, "centroids.pt"), map_location="cpu")
    assert centroids.shape == (num_partitions, dim)

    # TODO(jlscheerer) Perhaps do this per centroid instead of globally.
    bucket_cutoffs, bucket_weights = torch.load(
        os.path.join(index_path, "buckets.pt"), map_location="cpu"
    )

    np.save(
        os.path.join(destination_path, "bucket_cutoffs.npy"),
        bucket_cutoffs.float().numpy(force=True),
    )
    np.save(
        os.path.join(destination_path, "bucket_weights.npy"),
        bucket_weights.float().numpy(force=True),
    )

    print("[INFO] centroids.dtype=", centroids.dtype)

    centroids = centroids.float()
    # TODO(jlscheerer) Path the centroids again.
    # NOTE ...but for now we use the centroids.pt file anyways
    # assert centroids.dtype == torch.float
    np.save(
        os.path.join(destination_path, "centroids.npy"),
        centroids.numpy(force=True).astype(np.float32),
    )

    ivf, ivf_lengths = torch.load(os.path.join(index_path, "ivf.pid.pt"))
    assert ivf_lengths.shape == (num_partitions,)
    assert ivf.shape == (ivf_lengths.sum(),)

    # np.save(os.path.join(destination_path, "ivf.npy"), ivf.numpy())
    # np.save(os.path.join(destination_path, "ivf.size.npy"), ivf_lengths.numpy())

    print("> Loading centroid information")
    centroid_sizes = torch.zeros((num_partitions,), dtype=torch.int64)
    for chunk in tqdm(range(num_chunks)):
        # NOTE codes describe the corresponding centroid for each embedding
        codes = torch.load(os.path.join(index_path, f"{chunk}.codes.pt"))
        # residuals = torch.load(os.path.join(index_path, f"{chunk}.residuals.pt"))
        centroid_sizes += torch.bincount(codes, minlength=num_partitions)
    num_residuals = centroid_sizes.sum().item()

    offsets = torch.zeros((num_partitions,), dtype=torch.int64)
    offsets[1:] = torch.cumsum(centroid_sizes[:-1], dim=0)

    residual_dim = (dim * nbits) // 8  # residuals are stored as uint8

    tensor_offsets = torch.zeros((num_partitions,), dtype=torch.int64)
    tensor_offsets[1:] = torch.cumsum(centroid_sizes[:-1], dim=0)

    tensor_offsets = torch.zeros((num_partitions,), dtype=torch.int64)
    tensor_offsets[1:] = torch.cumsum(centroid_sizes[:-1], dim=0)

    tensor_compacted_residuals = torch.zeros(
        (num_residuals, residual_dim), dtype=torch.uint8
    )
    tensor_compacted_codes = torch.zeros((num_residuals,), dtype=torch.int32)

    print("> Compacting index")
    passage_id = 0
    for chunk in tqdm(range(num_chunks)):
        with open(os.path.join(index_path, f"doclens.{chunk}.json"), "r") as file:
            doclens = json.load(file)
        codes = torch.load(os.path.join(index_path, f"{chunk}.codes.pt"))
        residuals = torch.load(os.path.join(index_path, f"{chunk}.residuals.pt"))

        doclens = torch.tensor(doclens)
        assert doclens.sum() == residuals.shape[0]

        passage_ids = (
            torch.repeat_interleave(torch.arange(doclens.shape[0]), doclens).int()
            + passage_id
        )

        tensor_idx, tensor_offsets = segmented_index_cumsum(codes, tensor_offsets)

        tensor_compacted_residuals[tensor_idx] = residuals
        tensor_compacted_codes[tensor_idx] = passage_ids

        passage_id += doclens.shape[0]

    print("> Saving compacted index")
    torch.save(
        centroid_sizes,
        os.path.join(destination_path, "sizes.compacted.pt"),
    )
    torch.save(
        tensor_compacted_residuals,
        os.path.join(destination_path, "residuals.compacted.pt"),
    )
    torch.save(
        tensor_compacted_codes,
        os.path.join(destination_path, "codes.compacted.pt"),
    )

    print("> Repacking residuals")

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

    # TODO(jlscheerer) NOTE This requires nbits=4!
    residuals_repacked_compacted = reversed_bit_map[tensor_compacted_residuals.long()]
    residuals_repacked_compacted_d = decompression_lookup_table[
        residuals_repacked_compacted.long()
    ]
    residuals_repacked_compacted_df = (
        2**4 * residuals_repacked_compacted_d[:, :, 0]
        + residuals_repacked_compacted_d[:, :, 1]
    )
    torch.save(
        residuals_repacked_compacted_df,
        os.path.join(destination_path, "residuals.repacked.compacted.pt"),
    )

    print("> Averaging centroids")
    offsets_compacted = torch.zeros((num_partitions + 1,), dtype=torch.long)
    torch.cumsum(centroid_sizes, dim=0, out=offsets_compacted[1:])

    def decompress_centroid_embeddings(centroid_id):
        centroid = centroids[centroid_id]
        size = centroid_sizes[centroid_id]
        begin, end = offsets_compacted[centroid_id : centroid_id + 2]
        codes = tensor_compacted_codes[begin:end]
        residuals = tensor_compacted_residuals[begin:end]
        assert codes.shape == (size,) and residuals.shape[0] == size

        residuals_ = reversed_bit_map[residuals.long()]
        residuals_ = decompression_lookup_table[residuals_.long()]
        residuals_ = residuals_.reshape(residuals_.shape[0], -1)
        residuals_ = bucket_weights[residuals_.long()]
        embeddings = centroid + residuals_
        return torch.nn.functional.normalize(embeddings.to(torch.float32), p=2, dim=-1)

    avg_centroids = torch.zeros_like(centroids)
    for i in tqdm(range(centroids.shape[0])):
        centroid = decompress_centroid_embeddings(i).mean(dim=0)
        avg_centroids[i] = centroid

    torch.save(avg_centroids, os.path.join(destination_path, "avg_centroids.pt"))
