import os
import json

import torch
import numpy as np
from tqdm import tqdm

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN

def convert_index(index_path, destination_path=None):
    if destination_path is None:
        destination_path = index_path
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

    centroids = torch.load(os.path.join(index_path, "centroids.pt"))
    assert centroids.shape == (num_partitions, dim)

    # TODO(jlscheerer) Perhaps do this per centroid instead of globally.
    bucket_cutoffs, bucket_weights = torch.load(os.path.join(index_path, "buckets.pt"))

    np.save(
        os.path.join(destination_path, "bucket_cutoffs.npy"),
        bucket_cutoffs.float().numpy(force=True),
    )
    np.save(
        os.path.join(destination_path, "bucket_weights.npy"),
        bucket_weights.float().numpy(force=True),
    )

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

    np.save(os.path.join(destination_path, "ivf.npy"), ivf.numpy())
    np.save(os.path.join(destination_path, "ivf.size.npy"), ivf_lengths.numpy())

    centroid_sizes = torch.zeros((num_partitions,), dtype=torch.int64)
    for chunk in range(num_chunks):
        # NOTE codes describe the corresponding centroid for each embedding
        codes = torch.load(os.path.join(index_path, f"{chunk}.codes.pt"))
        # residuals = torch.load(os.path.join(index_path, f"{chunk}.residuals.pt"))
        centroid_sizes += torch.bincount(codes, minlength=num_partitions)
    num_residuals = centroid_sizes.sum().item()

    offsets = torch.zeros((num_partitions,), dtype=torch.int64)
    offsets[1:] = torch.cumsum(centroid_sizes[:-1], dim=0)

    residual_dim = (dim * nbits) // 8  # residuals are stored as uint8

    compacted_residuals = torch.zeros((num_residuals, residual_dim), dtype=torch.uint8)
    compacted_codes = torch.zeros((num_residuals,), dtype=torch.int32)

    passage_id = 0
    for chunk in range(num_chunks):
        print(f"Compacting chunk {chunk + 1} / {num_chunks}...")
        with open(os.path.join(index_path, f"doclens.{chunk}.json"), "r") as file:
            doclens = json.load(file)
        codes = torch.load(os.path.join(index_path, f"{chunk}.codes.pt"))
        residuals = torch.load(os.path.join(index_path, f"{chunk}.residuals.pt"))

        doclens = torch.tensor(doclens)
        assert doclens.sum() == residuals.shape[0]

        current_offset = 0
        current_length = 0
        for code, residual in tqdm(zip(codes, residuals)):
            if current_length == doclens[current_offset]:
                current_offset += 1
                current_length = 0
                passage_id += 1

            current_length += 1

            ofx = offsets[code]

            compacted_residuals[ofx] = residual
            compacted_codes[ofx] = passage_id

            offsets[code] += 1

        assert current_offset + 1 == len(doclens)
        assert current_length == doclens[current_offset]

        # NOTE we need to "manually" increase the passage_id for the final embeddings.
        passage_id += 1

    np.save(
        os.path.join(destination_path, "sizes.compacted.npy"), centroid_sizes.numpy()
    )
    np.save(
        os.path.join(destination_path, "residuals.compacted.npy"),
        compacted_residuals.numpy(),
    )
    np.save(
        os.path.join(destination_path, "codes.compacted.npy"), compacted_codes.numpy()
    )