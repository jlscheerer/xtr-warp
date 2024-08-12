import os

from dotenv import load_dotenv
load_dotenv()

from colbert.warp.config import WARPRunConfig

DATASETS = ["beir.scifact", "beir.scidocs", "beir.fiqa", "lotte.lifestyle",
            "lotte.technology", "lotte.science", "lotte.pooled"]

WARP_FILES = [
    "bucket_weights.npy", "centroids.npy", "sizes.compacted.pt",
    "codes.compacted.pt", "residuals.repacked.compacted.pt"
]
IGNORED_FILES = [
    "residuals.compacted.pt", "avg_centroids.pt", "bucket_cutoffs.npy"
]
SHARED_FILES = ["ivf.pid.pt"]

def filesize(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(filesize(os.path.join(path, file)) for file in os.listdir(path))

def warp_index_size(index_path):
    total_size = 0
    for entry in WARP_FILES + SHARED_FILES:
        total_size += filesize(os.path.join(index_path, entry))
    return total_size

def plaid_index_size(index_path):
    total_size = 0
    for entry in os.listdir(index_path):
        if entry not in (WARP_FILES + IGNORED_FILES):
            total_size += filesize(os.path.join(index_path, entry))
    return total_size

def bytes_to_gib(size):
    return size / (1024 * 1024 * 1024)

for collection_dataset in DATASETS:
    collection, dataset = collection_dataset.split(".")
    # TODO(jlscheerer) Introduce consistent naming for collection/dataset
    config = WARPRunConfig(
        nranks=4,
        dataset=collection,
        collection=dataset,
        type_="search" if collection == "lotte" else None,
        datasplit="test",
        nbits=4,
        k=100,
        optim=None
    )
    index_path = config.colbert().index_path

    try:
        warp_size = bytes_to_gib(warp_index_size(index_path))
    except:
        warp_size = "-"

    try:
        plaid_size = bytes_to_gib(plaid_index_size(index_path))
    except:
        plaid_size = "-"

    print(collection_dataset, "XTR/WARP", warp_size, "GiB")
    print(collection_dataset, "ColBERTv2/PLAID", plaid_size, "GiB")