import os

from dotenv import load_dotenv
load_dotenv()

from warp.engine.config import WARPRunConfig

DATASETS = ["beir.nfcorpus", "beir.scifact", "beir.scidocs",
            "beir.fiqa", "beir.webis-touche2020", "beir.quora",
            "lotte.lifestyle", "lotte.recreation", "lotte.writing", 
            "lotte.technology", "lotte.science", "lotte.pooled"]
NBITS_VALUES = [2, 4]

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

def safe_index_size(config: WARPRunConfig):
    index_path = config.colbert().index_path
    try:
        return warp_index_size(index_path)
    except:
        return None